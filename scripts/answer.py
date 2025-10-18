# scripts/answer.py
from pathlib import Path
import math
import re
from typing import List, Tuple
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_DIR = Path("indexes/calo_faiss")

def load_index_and_embed():
    # embed = HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2",
    #     encode_kwargs={"normalize_embeddings": True}
    # )
    embed = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    vs = FAISS.load_local(
        str(INDEX_DIR), embeddings=embed, allow_dangerous_deserialization=True
    )
    return vs, embed

def split_sentences(text: str) -> List[str]:
    # lightweight splitter (no NLTK download)
    # keeps abbreviations a bit safer than plain ". "
    text = re.sub(r"\s+", " ", text.strip())
    # split on ., ?, ! but keep punctuation attached
    parts = re.split(r"(?<=[.!?])\s+", text)
    # filter short fragments
    return [p.strip() for p in parts if len(p.strip()) > 20]

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def mmr_select(
    query_vec: np.ndarray,
    sent_vecs: List[np.ndarray],
    sents: List[str],
    k: int = 6,
    lambda_diversity: float = 0.7,
):
    # Maximal Marginal Relevance to reduce redundancy
    chosen, chosen_idx = [], []
    candidates = list(range(len(sents)))
    sim_to_query = [cosine(query_vec, v) for v in sent_vecs]

    if not candidates:
        return chosen

    # pick the most relevant first
    first = int(np.argmax(sim_to_query))
    chosen.append(sents[first])
    chosen_idx.append(first)
    candidates.remove(first)

    while candidates and len(chosen) < k:
        best_score, best_i = -1e9, None
        for i in candidates:
            rep = 0.0
            if chosen_idx:
                rep = max(
                    cosine(sent_vecs[i], sent_vecs[j]) for j in chosen_idx
                )
            score = lambda_diversity * sim_to_query[i] - (1 - lambda_diversity) * rep
            if score > best_score:
                best_score, best_i = score, i
        chosen.append(sents[best_i])
        chosen_idx.append(best_i)
        candidates.remove(best_i)
    return chosen

def answer(query: str, k_docs: int = 6, max_sents: int = 6):
    vs, embed = load_index_and_embed()
    # retrieve candidate chunks
    docs = vs.similarity_search(query, k=k_docs)

    claim_keywords = [
    "outperform", "better than", "state-of-the-art", "SOTA",
    "improv", "lower", "higher", "best", "vs.", "versus",
    "significant", "compared to", "faster", "speedup", "accuracy",
    "FPD", "KPD", "PRDC", "AUC", "correlation", "CFD"
    ]

    def is_claimy(s: str) -> bool:
        t = s.lower()
        return any(k in t for k in claim_keywords)

    # build a sentence pool with provenance
    sentence_pool: List[Tuple[str, str]] = []  # (sentence, citation)
    #sentence_pool = [(s, c) for (s, c) in sentence_pool if is_claimy(s)]
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "n/a")
        citation = f"[{Path(src).name} p.{page}]"
        for s in split_sentences(d.page_content):
            sentence_pool.append((s, citation))

    # filter; fallback if empty
    filtered = [(s, c) for (s, c) in sentence_pool if is_claimy(s)]
    sentence_pool = filtered or sentence_pool
    if not sentence_pool:
        print("No evidence found. Try a broader query.")
        return

    # embed query + sentences
    q_vec = np.array(embed.embed_query(query), dtype=np.float32)
    sent_texts = [s for s, _ in sentence_pool]
    sent_vecs = np.array(embed.embed_documents(sent_texts), dtype=np.float32)

    # select diverse, highly relevant sentences
    selected = mmr_select(q_vec, list(sent_vecs), sent_texts, k=max_sents)

    # stitch into a compact paragraph with citations (attach citation of first occurrence)
    used = set()
    lines = []
    for s in selected:
        # find provenance for this exact sentence
        for (orig, cit) in sentence_pool:
            if orig == s and (orig, cit) not in used:
                lines.append(f"{orig} {cit}")
                used.add((orig, cit))
                break

    # print final
    print("\nQUERY:", query)
    print("-" * 80)
    print(" ".join(lines))
    print("\nSources:")
    # list distinct sources in the retrieved set
    seen_sources = []
    for d in docs:
        src = Path(d.metadata.get("source", "unknown")).name
        page = d.metadata.get("page", "n/a")
        item = f"- {src} (page {page})"
        if item not in seen_sources:
            seen_sources.append(item)
    print("\n".join(seen_sources))

if __name__ == "__main__":
    # Try your own question, or edit these examples:
    # queries = [
    #     "What generative AI models performs the best in calorimeter shower simulation?",
    #     "What affects lateral containment and leakage in the ATLAS LAr EM barrel?",
    #     "What are common calibration or pile-up mitigation strategies for LAr calorimeters?"
    # ]
    queries = [
        "On CaloChallenge 2022 DS2, which model reports lowest FPD/KPD?",
        "Which model shows best CFD / correlation fidelity on EM showers (with numbers)?",
        "Which approach is fastest at sampling (ms/shower) while maintaining similar accuracy?"
    ]
    for q in queries:
        answer(q, k_docs=10, max_sents=8)
