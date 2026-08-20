"""
retriever.py
============
FAISS-backed semantic retriever with:
  - Metadata-aware re-ranking (boosts oracle / is_selected passages)
  - Similarity threshold guardrail (off-topic detection)
  - Unsafe-input guardrail (toxic / prompt-injection keyword blocklist)
  - Structured return type for clean downstream handling
"""

import re
import time
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field

# ── Unsafe-input blocklist ────────────────────────────────────────────────────
_UNSAFE_PATTERNS = re.compile(
    r"\b(ignore (all |previous |above )?instructions?|"
    r"jailbreak|prompt injection|"
    r"kill|bomb|suicide|self.harm|"
    r"how to (make|build|create) (a )?(bomb|weapon|poison|drug))\b",
    re.IGNORECASE,
)

# ── Similarity boost for selected passages ────────────────────────────────────
_SELECTED_BOOST = 0.05   # added to cosine score of is_selected == 1 chunks


@dataclass
class RetrievalResult:
    context:       str
    off_topic:     bool
    unsafe:        bool
    latency:       float
    best_score:    float
    chunk_sources: list[dict] = field(default_factory=list)   # metadata of top chunks


class FastRetriever:
    def __init__(
        self,
        index_path:  str = "msmarco_faiss.index",
        data_path:   str = "msmarco_chunks.csv",
        model_name:  str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.index_path = index_path
        self.data_path  = data_path

        print("Loading embedding model for retrieval…")
        self.model = SentenceTransformer(model_name)

        try:
            print("Loading FAISS index…")
            self.index = faiss.read_index(index_path)
            print("Loading chunk metadata…")
            self.df = pd.read_csv(data_path)
            self.ready = True
            print(f"Retriever ready — {self.index.ntotal} vectors, {len(self.df)} chunks")
        except Exception as exc:
            print(f"⚠ Could not load index ({exc}). Run build_index.py first!")
            self.ready = False

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:                str,
        top_k:                int   = 5,
        similarity_threshold: float = 0.35,
    ) -> RetrievalResult:
        """
        Main retrieval entry point.

        Pipeline:
          1. Unsafe-input guardrail  → immediate reject if triggered
          2. Embed query
          3. FAISS cosine search (top_k * 3 candidates for re-ranking)
          4. Metadata-aware re-ranking (boost is_selected passages)
          5. Off-topic guardrail      → reject if best score < threshold
          6. Return top_k context chunks with metadata
        """
        t0 = time.time()

        # ── 1. Unsafe input check ─────────────────────────────────────────────
        if _UNSAFE_PATTERNS.search(query):
            return RetrievalResult(
                context="", off_topic=False, unsafe=True,
                latency=time.time() - t0, best_score=0.0,
            )

        if not self.ready:
            return RetrievalResult(
                context="", off_topic=True, unsafe=False,
                latency=time.time() - t0, best_score=0.0,
            )

        # ── 2. Embed ──────────────────────────────────────────────────────────
        query_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)

        # ── 3. FAISS search (fetch 3× candidates for re-ranking headroom) ─────
        fetch_k = min(top_k * 3, self.index.ntotal)
        raw_scores, raw_indices = self.index.search(query_emb, fetch_k)
        raw_scores  = raw_scores[0]
        raw_indices = raw_indices[0]

        # ── 4. Metadata-aware re-ranking ─────────────────────────────────────
        boosted = []
        for score, idx in zip(raw_scores, raw_indices):
            if idx == -1 or idx >= len(self.df):
                continue
            meta  = self.df.iloc[idx]
            boost = _SELECTED_BOOST if int(meta.get("is_selected", 0)) == 1 else 0.0
            boosted.append((score + boost, idx, meta))

        boosted.sort(key=lambda x: x[0], reverse=True)
        top = boosted[:top_k]

        if not top:
            return RetrievalResult(
                context="", off_topic=True, unsafe=False,
                latency=time.time() - t0, best_score=0.0,
            )

        best_score = top[0][0]

        # ── 5. Off-topic guardrail ────────────────────────────────────────────
        if best_score < similarity_threshold:
            return RetrievalResult(
                context="", off_topic=True, unsafe=False,
                latency=time.time() - t0, best_score=float(best_score),
            )

        # ── 6. Compile context + chunk metadata ───────────────────────────────
        context_parts  = []
        chunk_sources  = []
        seen_texts     = set()

        for score, idx, meta in top:
            text = str(meta.get("text", "")).strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            context_parts.append(text)
            chunk_sources.append({
                "score":          round(float(score), 4),
                "chunk_strategy": str(meta.get("chunk_strategy", "?")),
                "query_type":     str(meta.get("query_type", "?")),
                "is_selected":    int(meta.get("is_selected", 0)),
            })

        return RetrievalResult(
            context      = "\n---\n".join(context_parts),
            off_topic    = False,
            unsafe       = False,
            latency      = time.time() - t0,
            best_score   = float(best_score),
            chunk_sources = chunk_sources,
        )
