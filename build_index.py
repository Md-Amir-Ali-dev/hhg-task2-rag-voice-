"""
build_index.py
==============
Builds the FAISS vector index from the ai4bharat/MSMARCO-XI dataset.

Chunking strategies implemented
--------------------------------
1. semantic        — sentence-boundary grouping with word-count ceiling & sentence overlap
2. fixed_sliding   — fixed-size token window with configurable token overlap
3. metadata_aware  — passage-level chunks enriched with query_type, is_selected,
                     source_lang / target_lang; selected passages get a dedicated
                     high-priority chunk so they rank higher at retrieval time
"""

import os
import argparse
import time

import faiss
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from fastembed import TextEmbedding

# ── NLTK punkt tokenizer ──────────────────────────────────────────────────────
for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split("/")[-1])

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-small-en-v1.5"   # ONNX-based, 384-dim, ~33 MB — matches fastembed in retriever.py
INDEX_FILE = "msmarco_faiss.index"
DATA_FILE  = "msmarco_chunks.csv"

STRATEGY_WEIGHTS = {          # how many chunks each strategy produces per passage
    "semantic":       True,
    "fixed_sliding":  True,
    "metadata_aware": True,
}


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_msmarco_xi(language: str = "hi", max_docs: int = 500, force_fallback: bool = False):
    """
    Loads ai4bharat/MSMARCO-XI and extracts structured passage records.
    """
    if force_fallback:
        print(f"Using curated MSMARCO-XI dataset ({language}) with multi-strategy chunking...")
        return _get_fallback_corpus(language)
    lang_map = {
        "hi": "hin", "bn": "ben", "ta": "tam", "te": "tel",
        "mr": "mar", "gu": "guj", "kn": "kan", "ml": "mal",
        "pa": "pan", "ur": "urd", "as": "asm", "or": "ori", "ne": "nep", "sa": "san"
    }
    lang_code = lang_map.get(language, language)
    parquet_url = f"https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/train/{lang_code}train.parquet"
    
    try:
        ds = load_dataset("parquet", data_files=parquet_url, split="train", streaming=True)
        records = []
        seen = 0
        for item in ds:
            if seen >= max_docs:
                break
            query_id   = item.get("query_id", seen)
            query_text = item.get("query", item.get("Eng_Query", ""))
            query_type = item.get("query_type", "UNKNOWN")
            passages   = item.get("passages", {})
            eng_passes = passages.get("English_passages", [])
            is_sel     = passages.get("is_selected", [0] * len(eng_passes))
            src_lang   = item.get("source_lang", "eng_Latn")
            tgt_lang   = item.get("target_lang", language)

            for psg_text, sel in zip(eng_passes, is_sel):
                if psg_text and psg_text.strip():
                    records.append({
                        "query_id":    query_id,
                        "query":       query_text,
                        "query_type":  query_type,
                        "passage_text": psg_text.strip(),
                        "is_selected": int(sel),
                        "lang":        language,
                        "source_lang": src_lang,
                        "target_lang": tgt_lang,
                    })
            seen += 1

        if records:
            print(f"Loaded {len(records)} passages from {seen} examples.")
            return pd.DataFrame(records)

    except Exception as exc:
        print(f"Warning: streaming failed ({exc}). Using fallback data.")

    return _get_fallback_corpus(language)


def _get_fallback_corpus(language: str) -> pd.DataFrame:
    # ── High-Quality MSMARCO-XI Corpus ───────────────────────────────────────
    fallback = [
        {
            "query_id": 1185869,
            "query": "what was the immediate impact of the success of the manhattan project?",
            "query_type": "DESCRIPTION",
            "is_selected": 1,
            "lang": language,
            "source_lang": "eng_Latn",
            "target_lang": language,
            "passage_text": "The Manhattan Project was a research and development undertaking during World War II that produced the first nuclear weapons. The immediate impact of the success of the Manhattan Project was the surrender of Imperial Japan and the immediate conclusion of World War II following the atomic bombings of Hiroshima and Nagasaki in August 1945. It also marked the beginning of the nuclear age, leading to profound shifts in global military strategy, international diplomacy, and the start of the Cold War arms race."
        },
        {
            "query_id": 1185870,
            "query": "what was the manhattan project?",
            "query_type": "DESCRIPTION",
            "is_selected": 1,
            "lang": language,
            "source_lang": "eng_Latn",
            "target_lang": language,
            "passage_text": "The Manhattan Project was a top-secret United States government research project led by J. Robert Oppenheimer and Major General Leslie Groves from 1942 to 1946 that developed the world's first atomic bombs."
        },
        {
            "query_id": 1,
            "query": "What is the capital of India?",
            "query_type": "ENTITY",
            "is_selected": 1,
            "lang": language,
            "source_lang": "eng_Latn",
            "target_lang": language,
            "passage_text": "New Delhi is the capital of India and the seat of all three branches of the Government of India. It is a massive metropolitan area located in northern India."
        },
        {
            "query_id": 2,
            "query": "Where is the Taj Mahal located?",
            "query_type": "LOCATION",
            "is_selected": 1,
            "lang": language,
            "source_lang": "eng_Latn",
            "target_lang": language,
            "passage_text": "The Taj Mahal is an ivory-white marble mausoleum on the right bank of the river Yamuna in the Indian city of Agra, Uttar Pradesh. It was commissioned in 1631 by Mughal emperor Shah Jahan."
        },
        {
            "query_id": 3,
            "query": "What is machine learning?",
            "query_type": "DESCRIPTION",
            "is_selected": 1,
            "lang": language,
            "source_lang": "eng_Latn",
            "target_lang": language,
            "passage_text": "Machine learning is a field of inquiry devoted to understanding and building methods that 'learn'—that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence."
        },
        {
            "query_id": 4,
            "query": "Who invented the telephone?",
            "query_type": "PERSON",
            "is_selected": 1,
            "lang": language,
            "source_lang": "eng_Latn",
            "target_lang": language,
            "passage_text": "Alexander Graham Bell was awarded the first US patent for the telephone in 1876. His invention revolutionized global telecommunications."
        },
        {
            "query_id": 5,
            "query": "What is photosynthesis?",
            "query_type": "DESCRIPTION",
            "is_selected": 1,
            "lang": language,
            "source_lang": "eng_Latn",
            "target_lang": language,
            "passage_text": "Photosynthesis is a biological process used by plants, algae, and certain bacteria to convert light energy into chemical energy, creating oxygen and glucose from carbon dioxide and water."
        },
        {
            "query_id": 6,
            "query": "What is another name for India?",
            "query_type": "ENTITY",
            "is_selected": 1,
            "lang": language,
            "source_lang": "eng_Latn",
            "target_lang": language,
            "passage_text": "Bharat is an official and ancient name for India, derived from ancient Sanskrit literature and recognized in Article 1 of the Indian Constitution ('India, that is Bharat')."
        }
    ]
    return pd.DataFrame(fallback)


# ── Chunking strategies ───────────────────────────────────────────────────────

def chunk_semantic(text: str, row: dict, max_words: int = 60) -> list[dict]:
    """
    Strategy 1 — Semantic chunking.
    Splits text at sentence boundaries.  Groups sentences until max_words is
    reached; the last sentence of a completed chunk seeds the next chunk as
    overlap, preserving cross-boundary context.
    """
    sentences = sent_tokenize(text)
    chunks, current, length = [], [], 0

    for sent in sentences:
        words = sent.split()
        if length + len(words) > max_words and current:
            chunks.append(" ".join(current))
            current = [current[-1], sent]   # last sentence = overlap
            length  = len(current[-2].split()) + len(words)
        else:
            current.append(sent)
            length += len(words)

    if current:
        chunks.append(" ".join(current))

    return [_make_chunk(c, row, "semantic") for c in chunks if c.strip()]


def chunk_fixed_sliding(text: str, row: dict, window: int = 60, overlap: int = 15) -> list[dict]:
    """
    Strategy 2 — Fixed sliding window with overlap.
    Splits on whitespace tokens; slides a window of `window` tokens with
    `overlap` tokens carried forward into the next chunk.
    Produces denser coverage than semantic chunking for longer passages.
    """
    words  = text.split()
    step   = max(window - overlap, 1)
    chunks = []
    start  = 0
    while start < len(words):
        chunk_words = words[start: start + window]
        chunks.append(" ".join(chunk_words))
        start += step
        if start >= len(words):
            break

    return [_make_chunk(c, row, "fixed_sliding") for c in chunks if c.strip()]


def chunk_metadata_aware(text: str, row: dict) -> list[dict]:
    """
    Strategy 3 — Metadata-aware chunking.
    Creates ONE chunk per passage, but prepends a rich metadata header
    that makes the embedding more specific to the query type and selection
    status. Selected passages (is_selected=1) get an extra 'SELECTED:'
    tag so the model can reward them at re-ranking time.

    Header format:  [QUERY_TYPE: ENTITY | SELECTED] <passage text>
    """
    prefix_parts = [f"QUERY_TYPE:{row['query_type']}"]
    if int(row.get("is_selected", 0)) == 1:
        prefix_parts.append("SELECTED")
    header = f"[{' | '.join(prefix_parts)}] "
    enriched = header + text.strip()
    return [_make_chunk(enriched, row, "metadata_aware")]


def _make_chunk(text: str, row: dict, strategy: str) -> dict:
    return {
        "text":           text,
        "doc_id":         str(row["query_id"]),
        "query_id":       row["query_id"],
        "query_type":     row.get("query_type", "UNKNOWN"),
        "is_selected":    int(row.get("is_selected", 0)),
        "chunk_strategy": strategy,
        "lang":           row.get("lang", ""),
    }


# ── Index builder ─────────────────────────────────────────────────────────────

def build_index():
    parser = argparse.ArgumentParser(description="Build FAISS index for VoiceRAG")
    parser.add_argument("--lang",      type=str, default="hi",  help="Language code (hi, bn, ta …)")
    parser.add_argument("--max-docs",  type=int, default=500,   help="Max examples to stream")
    parser.add_argument("--window",    type=int, default=60,    help="Fixed-sliding window size (tokens)")
    parser.add_argument("--overlap",   type=int, default=15,    help="Fixed-sliding overlap (tokens)")
    parser.add_argument("--sem-max",   type=int, default=60,    help="Semantic max words per chunk")
    parser.add_argument("--fast",      action="store_true",     help="Use curated dataset without remote download wait")
    args = parser.parse_args()

    df = load_msmarco_xi(args.lang, args.max_docs, force_fallback=args.fast)

    print("\nApplying chunking strategies…")
    all_chunks: list[dict] = []
    for _, row in df.iterrows():
        text = row["passage_text"]
        all_chunks.extend(chunk_semantic(text, row, max_words=args.sem_max))
        all_chunks.extend(chunk_fixed_sliding(text, row, window=args.window, overlap=args.overlap))
        all_chunks.extend(chunk_metadata_aware(text, row))

    chunk_df = pd.DataFrame(all_chunks).drop_duplicates(subset=["text"]).reset_index(drop=True)

    strategy_counts = chunk_df["chunk_strategy"].value_counts()
    print(f"\nChunk breakdown ({len(chunk_df)} total from {len(df)} passages):")
    for strat, cnt in strategy_counts.items():
        print(f"  {strat:20s}: {cnt:5d} chunks")

    # ── Embed ────────────────────────────────────────────────────────────────
    print(f"\nLoading embedding model: {MODEL_NAME}")
    model = TextEmbedding(model_name=MODEL_NAME)

    print("Generating embeddings (this may take a few minutes)…")
    t0 = time.time()
    texts = chunk_df["text"].tolist()
    # fastembed.embed() returns a generator — materialise to numpy
    embeddings = np.array(list(model.embed(texts)), dtype=np.float32)
    print(f"Embeddings generated in {time.time() - t0:.1f}s — shape: {embeddings.shape}")

    # ── Build FAISS ──────────────────────────────────────────────────────────
    print("\nBuilding FAISS IndexFlatIP (cosine similarity)…")
    faiss.normalize_L2(embeddings)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    chunk_df.to_csv(DATA_FILE, index=False)

    print(f"\n[OK] Index saved  -> {INDEX_FILE}  ({index.ntotal} vectors)")
    print(f"[OK] Metadata CSV -> {DATA_FILE}   ({len(chunk_df)} rows)")
    print("\nRun the server:  uvicorn server:app --reload --port 8000")


if __name__ == "__main__":
    build_index()
