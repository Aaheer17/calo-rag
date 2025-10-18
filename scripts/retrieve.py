# scripts/retrieve.py
from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_DIR = Path("indexes/calo_faiss")

def load_index():
    # Same embedding model as ingest.py
    embed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )
    # allow_dangerous_deserialization=True is common when loading FAISS built by another LC version
    vs = FAISS.load_local(
        str(INDEX_DIR),
        embeddings=embed,
        allow_dangerous_deserialization=True
    )
    return vs

def ask(query: str, k: int = 5):
    vs = load_index()
    docs = vs.similarity_search(query, k=k)

    print(f"\nQUERY: {query}")
    print(f"Top {k} results:\n" + "-"*60)
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "n/a")
        print(f"[{i}] {src}  (page {page})")
        # Show a compact snippet
        text = d.page_content.replace("\n", " ")
        if len(text) > 400:
            text = text[:400] + "..."
        print(text)
        print("-"*60)

if __name__ == "__main__":
    # Try a few seed queries relevant to calorimeters
    # demo_queries: List[str] = [
    #     "longitudinal and lateral shower shapes in sampling calorimeters",
    #     "energy deposition profile in ATLAS liquid-argon electromagnetic calorimeter",
    #     "pile-up mitigation or calibration constants for EM barrel",
    #     "hadronic vs electromagnetic shower fluctuations and leakage"
    # ]
    demo_queries: List[str]=[
        "The distribution of preprocessed voxel energies u has zero mean and unit variance",
        "The primary input to the network is the noisy representation of the shower"
        
        
    ]
    for q in demo_queries:
        ask(q, k=4)
