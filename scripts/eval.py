# scripts/eval.py
import argparse, json, io, sys
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore", message="Can't initialize NVML")
# import the existing answer() function
from scripts.answer import answer

def run_answer_capture(query: str, k: int, max_sents: int) -> str:
    """Run answer() but capture its printed output as text."""
    buf = io.StringIO()
    _stdout = sys.stdout
    try:
        sys.stdout = buf
        answer(query, k_docs=k, max_sents=max_sents)
    finally:
        sys.stdout = _stdout
    return buf.getvalue()

def passes(text: str, must_include_any: List[str]) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in must_include_any)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, required=True, help="Path to qas.jsonl")
    ap.add_argument("--k", type=int, default=8, help="# retrieved chunks")
    ap.add_argument("--max-sents", type=int, default=6, help="# composed sentences")
    ap.add_argument("--save", type=str, default="eval_results.jsonl", help="Output results")
    args = ap.parse_args()

    in_path = Path(args.file)
    out_path = Path(args.save)
    out = out_path.open("w", encoding="utf-8")

    total, hits = 0, 0
    print("\n=== Calo-RAG Quick Eval ===\n")
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item: Dict = json.loads(line)
            q = item["question"]
            must_any = item.get("must_include_any", [])
            total += 1

            text = run_answer_capture(q, k=args.k, max_sents=args.max_sents)
            ok = passes(text, must_any) if must_any else (len(text.strip()) > 0)
            hits += int(ok)

            # console summary
            badge = "✅" if ok else "❌"
            print(f"{badge} Q{total}: {q}")
            if not ok:
                print("    (Hint) Expected to see any of:", must_any)
            print("    --- snippet ---")
            snippet = " ".join(text.split())[:300]
            print(f"    {snippet}...\n")

            # write per-item record
            rec = {
                "question": q,
                "must_include_any": must_any,
                "passed": ok,
                "raw_answer": text
            }
            out.write(json.dumps(rec) + "\n")

    out.close()
    print(f"Score: {hits}/{total} = {hits/total:.2f}")
    print(f"Saved detailed results to {out_path}")

if __name__ == "__main__":
    main()
