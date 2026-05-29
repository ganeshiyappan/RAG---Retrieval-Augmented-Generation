"""
RAG — Retrieval Augmented Generation

Uses:
    pdfplumber            -> PDF text extraction
    sentence-transformers -> embeddings
    FAISS                 -> vector search
    OpenAI                -> answer generation

"""

import os
import json
import pickle
import argparse
from pathlib import Path

import pdfplumber
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5

EMBED_MODEL = "all-MiniLM-L6-v2"

# Choose model:
# "gpt-4.1"
# "gpt-4o"
# "gpt-4.1-mini"

OPENAI_MODEL = "gpt-4.1"

INDEX_DIR = Path(".rag_index")


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(path: str) -> str:
    """Extract text from PDF using pdfplumber."""
    text_parts = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()

            if text and text.strip():
                text_parts.append(
                    f"[Page {i+1}]\n{text.strip()}"
                )

    return "\n\n".join(text_parts)


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping chunks."""

    words = text.split()

    chunks = []

    i = 0
    chunk_idx = 0

    while i < len(words):

        chunk_words = words[i:i + CHUNK_SIZE]

        chunks.append({
            "text": " ".join(chunk_words),
            "source": source,
            "chunk_idx": chunk_idx,
            "word_start": i
        })

        chunk_idx += 1
        i += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(pdf_paths: list[str]):

    INDEX_DIR.mkdir(exist_ok=True)

    print(f"Loading embedding model: {EMBED_MODEL}")

    model = SentenceTransformer(EMBED_MODEL)

    all_chunks = []

    for pdf_path in pdf_paths:

        print(f"Processing: {pdf_path}")

        text = extract_pdf(pdf_path)

        if not text.strip():
            print(
                f" No text found in {pdf_path}. "
                f"It may be scanned and require OCR."
            )
            continue

        chunks = chunk_text(text, pdf_path)

        all_chunks.extend(chunks)

        print(f"→ {len(chunks)} chunks created")

    if not all_chunks:
        print("No chunks available.")
        return

    print(f"\nEmbedding {len(all_chunks)} chunks...")

    texts = [c["text"] for c in all_chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    embeddings = embeddings.astype("float32")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    faiss.write_index(
        index,
        str(INDEX_DIR / "index.faiss")
    )

    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    meta = {
        "num_chunks": len(all_chunks),
        "dim": dimension,
        "embed_model": EMBED_MODEL
    }

    with open(INDEX_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"\n✓ Index saved "
        f"({len(all_chunks)} chunks)"
    )


# ============================================================
# LOAD INDEX
# ============================================================

def load_index():

    index_path = INDEX_DIR / "index.faiss"

    if not index_path.exists():
        raise FileNotFoundError(
            "No index found.\n"
            "Run:\n"
            "python rag.py index file.pdf"
        )

    index = faiss.read_index(str(index_path))

    with open(INDEX_DIR / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    with open(INDEX_DIR / "meta.json") as f:
        meta = json.load(f)

    model = SentenceTransformer(
        meta["embed_model"]
    )

    return index, chunks, model


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    query: str,
    index,
    chunks,
    embed_model,
    k=TOP_K
):

    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        chunk = dict(chunks[idx])

        chunk["score"] = float(score)

        results.append(chunk)

    return results


# ============================================================
# CONTEXT BUILDER
# ============================================================

def build_context(retrieved):

    parts = []

    for i, chunk in enumerate(retrieved, start=1):

        source = Path(chunk["source"]).name

        parts.append(
            f"[Excerpt {i} — {source}, chunk {chunk['chunk_idx']}]\n"
            f"{chunk['text']}"
        )

    return "\n\n---\n\n".join(parts)


# ============================================================
# OPENAI ANSWER GENERATION
# ============================================================

def answer(
    query: str,
    retrieved: list[dict],
    client: OpenAI
):

    context = build_context(retrieved)

    system_prompt = """
You are a precise research assistant.

Use ONLY the document excerpts provided.

Rules:
1. Cite supporting excerpts.
2. If information is missing, say so.
3. Do not hallucinate.
4. Keep answers factual and concise.
"""

    user_prompt = f"""
Document Excerpts:

{context}

--------------------------------

Question:
{query}
"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response.choices[0].message.content


# ============================================================
# CHAT LOOP
# ============================================================

def chat_loop(
    index,
    chunks,
    embed_model,
    client
):

    print("\n📚 RAG Ready\n")

    while True:

        try:
            query = input("You: ").strip()

        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue

        if query.lower() in {
            "quit",
            "exit",
            "q"
        }:
            print("Bye!")
            break

        retrieved = retrieve(
            query,
            index,
            chunks,
            embed_model
        )

        if not retrieved:
            print("No relevant chunks found.\n")
            continue

        print(
            f"\nRetrieved {len(retrieved)} chunks\n"
        )

        result = answer(
            query,
            retrieved,
            client
        )

        print("\nAssistant:")
        print(result)
        print("\n" + "-" * 60)


# ============================================================
# SINGLE QUERY
# ============================================================

def query_once(
    question,
    index,
    chunks,
    embed_model,
    client
):

    retrieved = retrieve(
        question,
        index,
        chunks,
        embed_model
    )

    if not retrieved:
        return "No relevant content found."

    return answer(
        question,
        retrieved,
        client
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="PDF RAG Pipeline using OpenAI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    index_parser = subparsers.add_parser(
        "index"
    )

    index_parser.add_argument(
        "pdfs",
        nargs="+"
    )

    subparsers.add_parser("chat")

    ask_parser = subparsers.add_parser(
        "ask"
    )

    ask_parser.add_argument(
        "question"
    )

    subparsers.add_parser("info")

    args = parser.parse_args()

    if args.command == "index":
        build_index(args.pdfs)
        return

    if args.command == "info":

        meta_path = INDEX_DIR / "meta.json"

        if not meta_path.exists():
            print("No index found.")
            return

        with open(meta_path) as f:
            meta = json.load(f)

        print("\nIndex Information")
        print("-----------------")
        print(
            f"Chunks: {meta['num_chunks']}"
        )
        print(
            f"Dimension: {meta['dim']}"
        )
        print(
            f"Embedding Model: {meta['embed_model']}"
        )

        return

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set.\n"
            "Example:\n"
            "export OPENAI_API_KEY=sk-..."
        )

    client = OpenAI(
        api_key=api_key
    )

    index, chunks, embed_model = load_index()

    if args.command == "chat":

        chat_loop(
            index,
            chunks,
            embed_model,
            client
        )

    elif args.command == "ask":

        result = query_once(
            args.question,
            index,
            chunks,
            embed_model,
            client
        )

        print(result)


if __name__ == "__main__":
    main()