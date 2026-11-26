#!/usr/bin/env python3
"""
Small drafter (VRAM) + big judge (RAM) benchmark, using Ollama.

Setups measured on the same problems:
  1. 4B alone                      (drafter model, one answer)
  2. 14B alone                     (judge model, one answer)
  3. 4B candidate #1               (what the user sees first in the hybrid)
  4. Best-of-N + 14B judge         (final answer after the judge pass)
  Extras: oracle (any candidate right), random pick, majority vote (math only).

No third-party packages needed. Only the Python standard library.

WARNING: the code problems execute model-written Python on your machine
(in a subprocess with a timeout). Run it only on a machine you are fine with that on.

Usage examples:
  python run_benchmark.py --limit 10            # quick check
  python run_benchmark.py                       # all 50 problems
  python run_benchmark.py --resume              # continue an interrupted run
  python run_benchmark.py --judge-num-gpu 0     # force the 14B fully onto CPU/RAM
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import Counter

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not HOST.startswith("http"):
    HOST = "http://" + HOST


# ----------------------------------------------------------------------------
# Ollama call
# ----------------------------------------------------------------------------
def chat(model, prompt, temperature, num_predict, num_ctx, num_gpu=None):
    options = {"temperature": temperature, "num_predict": num_predict, "num_ctx": num_ctx}
    if num_gpu is not None and num_gpu >= 0:
        options["num_gpu"] = num_gpu
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,          # Qwen3 thinking mode off, same for every setup
        "keep_alive": "30m",
        "options": options,
    }).encode()
    req = urllib.request.Request(f"{HOST}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    wall = time.perf_counter() - t0
    text = d.get("message", {}).get("content", "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return {
        "text": text,
        "wall": wall,                                         # seconds, full request
        "load_s": d.get("load_duration", 0) / 1e9,            # model load (should be ~0 after warm-up)
        "prompt_tokens": d.get("prompt_eval_count", 0),
        "prompt_eval_s": d.get("prompt_eval_duration", 0) / 1e9,
        "gen_tokens": d.get("eval_count", 0),
        "eval_s": d.get("eval_duration", 0) / 1e9,
    }


def warmup(model, num_ctx, num_gpu=None):
    print(f"Warming up {model} ...")
    chat(model, "Say OK.", 0, 4, num_ctx, num_gpu)


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------
def make_prompt(p):
    if p["type"] == "math":
        return (p["question"] + "\n\nSolve it step by step, concisely. "
                "End your reply with a final line of the form:\nFinal answer: <number>")
    return (p["question"] + "\n\nReturn only the complete Python code in one ```python block. "
            "No explanation and no test code.")


def condense(p, text):
    """Shorten a candidate so the judge prompt stays small."""
    if p["type"] == "math":
        return text[-1200:]
    code = extract_code(text)
    return code[:1500]


def make_judge_prompt(p, cands):
    parts = [
        "You are a strict grader. Below is a task and several candidate answers. "
        "Pick the candidate most likely to be fully correct. "
        f"Reply with ONLY the candidate number (1 to {len(cands)}).\n",
        "TASK:\n" + p["question"] + "\n",
    ]
    for i, c in enumerate(cands, 1):
        parts.append(f"CANDIDATE {i}:\n{condense(p, c['text'])}\n")
    parts.append("Best candidate number:")
    return "\n".join(parts)


# ----------------------------------------------------------------------------
# Answer checking
# ----------------------------------------------------------------------------
NUM = r"-?\d[\d,]*\.?\d*|-?\.\d+"


def extract_number(text):
    m = re.findall(r"final answer\s*:?\s*[\*\$\\\(\s]*(" + NUM + ")", text, flags=re.I)
    if not m:
        m = re.findall("(" + NUM + ")", text)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def extract_code(text):
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S | re.I)
    return m.group(1) if m else text


def check_code(code, tests, timeout=10):
    program = code + "\n\n" + tests + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(program)
        try:
            r = subprocess.run([sys.executable, path], capture_output=True,
                               timeout=timeout, cwd=tmp)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def check(p, text):
    if p["type"] == "math":
        v = extract_number(text)
        return v is not None and abs(v - float(p["answer"])) < 0.01
    return check_code(extract_code(text), p["tests"])


def parse_pick(text, n):
    m = re.search(r"\d+", text)
    if m:
        k = int(m.group())
        if 1 <= k <= n:
            return k - 1
    return 0  # fall back to candidate 1


# ----------------------------------------------------------------------------
# Phases (grouped by model so Ollama does not keep swapping models in VRAM)
# ----------------------------------------------------------------------------
def save(results, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)


def phase_draft(problems, results, a):
    warmup(a.draft_model, a.draft_ctx)
    for i, p in enumerate(problems, 1):
        r = results.setdefault(p["id"], {"type": p["type"]})
        prompt = make_prompt(p)
        if "alone_small" not in r:
            o = chat(a.draft_model, prompt, 0.2, a.max_tokens, a.draft_ctx)
            o["correct"] = check(p, o["text"])
            r["alone_small"] = o
        if "cands" not in r or len(r["cands"]) < a.n:
            cands = []
            for _ in range(a.n):
                o = chat(a.draft_model, prompt, a.draft_temp, a.max_tokens, a.draft_ctx)
                o["correct"] = check(p, o["text"])
                cands.append(o)
            r["cands"] = cands
        save(results, a.out)
        ok = sum(c["correct"] for c in r["cands"])
        print(f"[draft {i}/{len(problems)}] {p['id']}  alone={'Y' if r['alone_small']['correct'] else 'N'}  "
              f"candidates correct={ok}/{a.n}")


def phase_big(problems, results, a):
    warmup(a.big_model, a.big_ctx, a.judge_num_gpu)
    for i, p in enumerate(problems, 1):
        r = results[p["id"]]
        if not a.skip_big_alone and "alone_big" not in r:
            o = chat(a.big_model, make_prompt(p), 0.2, a.max_tokens, a.big_ctx, a.judge_num_gpu)
            o["correct"] = check(p, o["text"])
            r["alone_big"] = o
        if "judge" not in r:
            o = chat(a.big_model, make_judge_prompt(p, r["cands"]), 0, 8, a.big_ctx, a.judge_num_gpu)
            pick = parse_pick(o["text"], len(r["cands"]))
            o["pick"] = pick
            o["correct"] = r["cands"][pick]["correct"]
            r["judge"] = o
        save(results, a.out)
        j = r["judge"]
        print(f"[big   {i}/{len(problems)}] {p['id']}  judge picked #{j['pick'] + 1} "
              f"({'correct' if j['correct'] else 'wrong'}) in {j['wall']:.1f}s"
              + (f"  big-alone={'Y' if r['alone_big']['correct'] else 'N'}" if "alone_big" in r else ""))


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(problems, results, a):
    by_id = {p["id"]: p for p in problems}
    groups = {"ALL": [p["id"] for p in problems],
              "MATH": [p["id"] for p in problems if p["type"] == "math"],
              "CODE": [p["id"] for p in problems if p["type"] == "code"]}
    lines = []
    for name, ids in groups.items():
        rs = [(i, results[i]) for i in ids if i in results]
        done = [(i, r) for i, r in rs if "judge" in r and "cands" in r]
        if not done:
            continue
        lines.append(f"\n=== {name}  ({len(done)} problems, N={a.n}) ===")
        lines.append(f"{'setup':<34}{'accuracy':>10}{'avg time (s)':>15}")

        def row(label, accs, times):
            lines.append(f"{label:<34}{100 * mean(accs):>9.1f}%{mean(times):>15.1f}")

        row(f"{a.draft_model} alone",
            [r["alone_small"]["correct"] for _, r in done],
            [r["alone_small"]["wall"] for _, r in done])
        big = [(i, r) for i, r in done if "alone_big" in r]
        if big:
            row(f"{a.big_model} alone",
                [r["alone_big"]["correct"] for _, r in big],
                [r["alone_big"]["wall"] for _, r in big])
        row("hybrid: first answer (cand #1)",
            [r["cands"][0]["correct"] for _, r in done],
            [r["cands"][0]["wall"] for _, r in done])
        row("best-of-N + judge: final",
            [r["judge"]["correct"] for _, r in done],
            [sum(c["wall"] for c in r["cands"]) + r["judge"]["wall"] for _, r in done])
        row("  random pick (no judge)",
            [mean([c["correct"] for c in r["cands"]]) for _, r in done],
            [sum(c["wall"] for c in r["cands"]) for _, r in done])
        lines.append(f"{'  oracle (any candidate right)':<34}"
                     f"{100 * mean([any(c['correct'] for c in r['cands']) for _, r in done]):>9.1f}%")
        if name != "CODE":
            maj = []
            for i, r in done:
                if by_id[i]["type"] != "math":
                    continue
                vals = [extract_number(c["text"]) for c in r["cands"]]
                vals = [round(v, 4) for v in vals if v is not None]
                if vals:
                    top = Counter(vals).most_common(1)[0][0]
                    maj.append(abs(top - float(by_id[i]["answer"])) < 0.01)
                else:
                    maj.append(False)
            if maj:
                lines.append(f"{'  majority vote (math, no judge)':<34}{100 * mean(maj):>9.1f}%")
        solvable = [r for _, r in done if any(c["correct"] for c in r["cands"])]
        if solvable:
            lines.append(f"{'  judge right when a right one exists':<34}"
                         f"{100 * mean([r['judge']['correct'] for r in solvable]):>9.1f}%")
        lines.append(f"  judge pass: avg {mean([r['judge']['wall'] for _, r in done]):.1f}s, "
                     f"avg prompt {mean([r['judge']['prompt_tokens'] for _, r in done]):.0f} tokens, "
                     f"prompt-eval {mean([r['judge']['prompt_eval_s'] for _, r in done]):.1f}s")
    text = "\n".join(lines)
    print(text)
    with open(os.path.splitext(a.out)[0] + "_summary.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="problems.json")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--draft-model", default="qwen3:4b")
    ap.add_argument("--big-model", default="qwen3:14b")
    ap.add_argument("--n", type=int, default=5, help="candidates per problem")
    ap.add_argument("--draft-temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--draft-ctx", type=int, default=4096)
    ap.add_argument("--big-ctx", type=int, default=8192)
    ap.add_argument("--judge-num-gpu", type=int, default=-1,
                    help="GPU layers for the big model (-1 = Ollama decides, 0 = CPU/RAM only)")
    ap.add_argument("--limit", type=int, default=0, help="only the first K problems of each type")
    ap.add_argument("--skip-big-alone", action="store_true", help="skip the slow 14B-alone baseline")
    ap.add_argument("--resume", action="store_true", help="continue from the existing results file")
    a = ap.parse_args()

    with open(a.problems, encoding="utf-8") as f:
        problems = json.load(f)
    if a.limit:
        keep = []
        for t in ("math", "code"):
            keep += [p for p in problems if p["type"] == t][:max(1, a.limit // 2)]
        problems = keep

    results = {}
    if a.resume and os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            results = json.load(f)

    print(f"{len(problems)} problems | draft={a.draft_model} big={a.big_model} N={a.n}")
    t0 = time.time()
    phase_draft(problems, results, a)
    phase_big(problems, results, a)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Raw data: {a.out}")
    summarize(problems, results, a)


if __name__ == "__main__":
    main()
