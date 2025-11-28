import os
import glob
from rank_bm25 import BM25Okapi
import re

class LocalRetriever:
    def __init__(self, docs_path="docs/"):
        self.chunks = []
        self.chunk_ids = []
        self.corpus = []
        self.load_docs(docs_path)
        
        # Tokenize for BM25
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def load_docs(self, path):
        """Loads MD files and chunks them by headers or paragraphs."""
        files = glob.glob(os.path.join(path, "*.md"))
        for f in files:
            filename = os.path.basename(f).replace(".md", "")
            with open(f, 'r') as file:
                content = file.read()
                # Simple split by double newline for chunks
                raw_chunks = content.split("\n\n")
                for i, chunk in enumerate(raw_chunks):
                    if chunk.strip():
                        chunk_id = f"{filename}::chunk{i}"
                        self.chunks.append(chunk.strip())
                        self.chunk_ids.append(chunk_id)
                        self.corpus.append(chunk.strip())

    def search(self, query, top_k=3):
        """Returns top_k chunks with their IDs."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_n = self.bm25.get_top_n(tokenized_query, self.corpus, n=top_k)
        
        results = []
        # Find indices of top_n to get IDs (naive approach for this scale)
        for text in top_n:
            idx = self.corpus.index(text)
            results.append({
                "id": self.chunk_ids[idx],
                "text": text,
                "score": scores[idx]
            })
        return results