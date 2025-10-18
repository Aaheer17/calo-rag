# Calo-RAG: A tiny, local RAG for calorimeter papers

**What it is.**  
A minimal, CPU-only Retrieval-Augmented Generation (RAG) prototype tailored to calorimeter literature. It uses local PDF parsing, local embeddings, FAISS retrieval, and an extractive “answer composer” with inline citations—no external APIs.

## Features
- 📄 PDF → chunks (local)
- 🔎 FAISS retrieval with local embeddings
- 🧩 Extractive answer composer with inline `[file p.X]` citations
- 🧪 Tiny eval harness (JSONL) for quick, repeatable checks
- 💻 CPU-only by default



## Environment

### Option A — use an existing env
```bash
conda activate llm_policy
pip install -U "langchain>=0.3.0" "langchain-community>=0.3.0" \
  "sentence-transformers>=3.0.0" faiss-cpu pypdf python-dotenv tqdm rich
# If faiss-cpu wheel fails on your OS:
# conda install -c conda-forge faiss-cpu
```
### Option B - create a new environment using env.yml
```bash
conda env create -f env.yml
conda activate calo-rag
```
## Step 1: Add PDFs & build the index
```bash
mkdir -p data indexes
# Put 3–5 calorimeter PDFs in ./data

python scripts/ingest.py
# -> "Saved FAISS index with <N> chunks to indexes/calo_faiss"
```
## Step 2: Try retrieval (sanity check)
```bash
python -m scripts.retrieve
#Prints top passages with filenames and page numbers for a few demo queries.
```
## Step 3: Ask a question (end-to-end)
```bash
python -m scripts.ask "Which recent approaches are used for fast calorimeter shower generation?" --k 10 --max-sents 8
#Outputs a short, stitched paragraph with inline citations like [paper.pdf p.12] and a compact “Sources” list.
```
## Step 4: Quick eval (simple, keyword-based)
```bash
# Validate JSONL format (no output = OK):
python -c 'import json;[json.loads(l) for l in open("eval/qas.jsonl")]'

# Run eval:
python -m scripts.eval --file eval/qas.jsonl --k 8 --max-sents 6
# -> shows per-question pass/fail, a final score, and saves eval_results.jsonl
```

