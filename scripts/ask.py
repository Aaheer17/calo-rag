# scripts/ask.py
import argparse
from scripts.answer import answer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("query", type=str, help="Your question")
    p.add_argument("--k", type=int, default=6, help="# retrieved chunks")
    p.add_argument("--max-sents", type=int, default=6, help="# sentences to compose")
    args = p.parse_args()

    answer(args.query, k_docs=args.k, max_sents=args.max_sents)

if __name__ == "__main__":
    main()
