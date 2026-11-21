"""Document ranking from persistent passage vectors and BM25; no reuse judgment."""

import numpy as np
from app.retrieval.generations import GenerationStore, IndexSpec, IndexUnavailable
from app.retrieval.local_model import local_encoder

QUERY_BATCH_SIZE = 32


class HybridRetriever:
    def __init__(self, corpus, vectors=None, encoder=None):
        self.corpus = corpus
        self.encoder = encoder or local_encoder()
        self.generation_id = None
        if vectors is None:
            generation, error = GenerationStore(corpus.retrieval_index_path).active(
                corpus, IndexSpec.local()
            )
            if generation is None or error:
                raise IndexUnavailable(
                    "A compatible local index must be built before analysis"
                )
            self.generation_id = generation.id
            vectors = generation.vectors
        self.vectors = np.asarray(vectors, dtype=np.float32)
        if (
            self.vectors.shape != (len(corpus.documents), 384)
            or not np.isfinite(self.vectors).all()
        ):
            raise IndexUnavailable("Invalid local retrieval matrix")
        self.owners = {}
        self.indices = {d.source_id: i for i, d in enumerate(corpus.documents)}
        self.by_passage = {d.passage_id: i for i, d in enumerate(corpus.documents)}
        for i, d in enumerate(corpus.documents):
            for owner in d.source_occurrences:
                self.owners.setdefault(owner.source_document_id, owner)
        if not self.owners:
            raise IndexUnavailable("No grounded source versions in index")

    def _dense_queries(self, queries):
        # Keep only one embedding/similarity batch, never a document-wide
        # query-by-corpus matrix. Every query retains its original unit index.
        for start in range(0, len(queries), QUERY_BATCH_SIZE):
            batch = queries[start : start + QUERY_BATCH_SIZE]
            vectors = np.asarray(self.encoder.encode(batch), dtype=np.float32)
            if vectors.shape != (len(batch), 384) or not np.isfinite(vectors).all():
                raise IndexUnavailable("Invalid query embeddings")
            dense = vectors @ self.vectors.T
            for offset, query in enumerate(batch):
                yield start + offset, query, dense[offset]

    def rank(self, units):
        queries = [u.text if hasattr(u, "text") else u for u in units]
        if not queries:
            return {"rankings": {}, "documents": [], "related": []}
        lexical_scores = {}
        dense_scores = {}
        related = []
        for qi, query, dense_row in self._dense_queries(queries):
            for match in self.corpus.search(query, limit=len(self.corpus.documents)):
                for owner in match.source_occurrences:
                    sid = owner.source_document_id
                    lexical_scores[sid] = max(lexical_scores.get(sid, 0), match.score)
            best = np.argsort(-dense_row, kind="stable")
            for i in best:
                document = self.corpus.documents[int(i)]
                for owner in document.source_occurrences:
                    sid = owner.source_document_id
                    dense_scores[sid] = max(
                        dense_scores.get(sid, -1), float(dense_row[i])
                    )
            # Automatically nominate a bounded passage pool per submitted unit.
            # No cosine cutoff is treated as a reuse classifier.
            count = 0
            for index in best:
                i = int(index)
                document = self.corpus.documents[i]
                for owner in document.source_occurrences:
                    if count >= 8:
                        break
                    count += 1
                    related.append(
                        {
                            "unit_index": qi,
                            "source_document_id": owner.source_document_id,
                            "source_start": owner.source_start_char,
                            "source_end": owner.source_end_char,
                            "source_excerpt": document.text,
                            "source_title": owner.title,
                            "source_url": owner.url,
                            "source_revision": owner.revision,
                            "source_document_sha256": owner.source_document_sha256,
                            "source_passage_id": document.passage_id,
                            "source_attribution": {
                                k: getattr(owner, k)
                                for k in (
                                    "authors",
                                    "collection",
                                    "license_url",
                                    "license_name",
                                    "rights_status",
                                    "attribution_text",
                                    "attribution_url",
                                )
                            },
                            "minilm_rank": count,
                            "score": float(dense_row[i]),
                            "category": "POTENTIAL_SEMANTIC_CORRESPONDENCE",
                            "score_eligible": False,
                            "boundary_granularity": "retrieved_passage",
                            "uncertainty": "Relatedness only; reuse is not established.",
                        }
                    )
                if count >= 8:
                    break
        lexical = sorted(lexical_scores, key=lambda s: (-lexical_scores[s], s))
        dense_rank = sorted(dense_scores, key=lambda s: (-dense_scores[s], s))
        fusion = {sid: 0.0 for sid in self.owners}
        for ranking in (lexical, dense_rank):
            for rank, sid in enumerate(ranking, 1):
                fusion[sid] += 1 / (60 + rank)
        fused = sorted(fusion, key=lambda s: (-fusion[s], s))
        return {
            "rankings": {"bm25": lexical, "minilm": dense_rank, "rrf": fused},
            "documents": fused,
            "related": related,
            "query_count": len(queries),
            "query_batch_size": QUERY_BATCH_SIZE,
            "generation_id": self.generation_id,
            "method": "document-max BM25 + MiniLM; RRF k=60",
        }
