from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_DIR = Path("data")
INDEX_DIR = Path("indexes/calo_faiss")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def load_pdfs(data_dir: Path):
    docs = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())
    return docs

def main():
    # 1) Load
    raw_docs = load_pdfs(DATA_DIR)
    if not raw_docs:
        raise RuntimeError("No PDFs found in ./data. Add 3–5 calorimeter papers first.")

    # 2) Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True
    )
    chunks = splitter.split_documents(raw_docs)

    # 3) Local embeddings (fast CPU model)
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    # 4) Build FAISS and persist
    vs = FAISS.from_documents(chunks, embed_model)
    vs.save_local(str(INDEX_DIR))
    print(f"Saved FAISS index with {len(chunks)} chunks to {INDEX_DIR}")

if __name__ == "__main__":
    main()
