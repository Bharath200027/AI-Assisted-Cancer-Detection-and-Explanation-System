from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Evidence:
    source: str
    score: float
    snippet: str

def _read_kb_files(kb_dir: Path) -> List[Tuple[str, str]]:
    files = sorted([p for p in kb_dir.glob("**/*") if p.is_file() and p.suffix.lower() in {".md", ".txt"}])
    out = []
    for p in files:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        out.append((str(p), txt))
    return out

def _chunk_text(text: str, max_chars: int = 800) -> List[str]:
    # simple paragraph chunking
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for para in paras:
        if len(cur) + len(para) + 2 <= max_chars:
            cur = (cur + "\n\n" + para).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = para
    if cur:
        chunks.append(cur)
    return chunks

class TfidfRAG:
    def __init__(self, kb_dir: str, top_k: int = 4):
        self.kb_dir = Path(kb_dir)
        self.top_k = top_k
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)

        pairs = _read_kb_files(self.kb_dir)
        self._docs = []   # (source, chunk)
        for src, txt in pairs:
            for chunk in _chunk_text(txt):
                self._docs.append((src, chunk))

        self._X = self._vectorizer.fit_transform([c for _, c in self._docs]) if self._docs else None

    def search(self, query: str) -> List[Evidence]:
        if not self._docs or self._X is None:
            return []
        q = self._vectorizer.transform([query])
        sims = cosine_similarity(q, self._X).ravel()
        idx = sims.argsort()[::-1][: self.top_k]
        out = []
        for i in idx:
            out.append(Evidence(source=self._docs[i][0], score=float(sims[i]), snippet=self._docs[i][1]))
        return out
