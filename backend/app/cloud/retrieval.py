"""Bounded Pinecone passage retrieval and server-side BM25 with existing RRF."""

from collections import Counter
import math

from app.cloud.client import CloudUnavailable, connect
from app.cloud.pinecone import Pinecone
from app.cloud.records import jsonb
from app.cloud.sources import occurrence
from app.cloud.libraries import library_scope
import os
from app.retrieval.bm25 import _tokenize
from app.retrieval.local_model import local_encoder
from app.storage.source_corpus import digest


class CloudRetriever:
    def __init__(self, corpus):
        self.corpus = corpus
        self.owner = corpus.store._owner()
        self.mapping = corpus.version_libraries
        self.pines = {key: Pinecone(key) for key in sorted(set(self.mapping.values()))}
        self.pine = next(iter(self.pines.values()), Pinecone(self.owner))
        self.generation_id = "pinecone-" + self.pine.space
        with connect() as c:
            rows = c.execute(
                """SELECT DISTINCT ON(o.version_id) o.*,v.metadata,v.text_sha256
                FROM st_occurrences o JOIN st_versions v USING(owner_id,version_id)
                JOIN jsonb_each_text(%s) s ON o.version_id=s.key AND o.owner_id=s.value::uuid
                ORDER BY o.version_id,o.source_id,o.start_char""",
                (jsonb(self.mapping),),
            ).fetchall()
        self.owners = {r["version_id"]: occurrence(r) for r in rows}

    def bm25(self, query):
        terms = Counter(_tokenize(query))
        if not terms:
            return {}
        # Same tokenizer, IDF, length normalization, rounding and document-max
        # aggregation as LocalCorpus. No corpus-sized token index in Python.
        with connect() as c:
            rows = c.execute(
                """WITH eligible AS (
                  SELECT o.owner_id,o.passage_id,o.version_id FROM st_occurrences o
                  JOIN jsonb_each_text(%s) s ON o.version_id=s.key AND o.owner_id=s.value::uuid
                ), active AS (
                  SELECT DISTINCT ON(passage_id) owner_id,passage_id FROM eligible ORDER BY passage_id,owner_id
                ), stats AS MATERIALIZED (
                  SELECT count(*)::double precision AS n,greatest(coalesce(avg(p.token_count),0),1) AS av
                  FROM active a JOIN st_passages p USING(owner_id,passage_id)
                ), terms AS (SELECT key AS term,value::integer AS qf FROM jsonb_each_text(%s)),
                hits AS (
                  SELECT t.passage_id,t.term,t.frequency,p.token_count,q.qf,
                    count(*) OVER(PARTITION BY t.term)::double precision AS df
                  FROM active a JOIN st_terms t USING(owner_id,passage_id) JOIN terms q USING(term)
                  JOIN st_passages p USING(owner_id,passage_id)
                ), scores AS (
                  SELECT passage_id,round(sum(qf*ln(1+(s.n-df+0.5)/(df+0.5))*
                    (frequency*2.0)/(frequency+1.5*(0.25+0.75*token_count/s.av)))::numeric,4)::double precision AS score
                  FROM hits CROSS JOIN stats s GROUP BY passage_id
                ) SELECT e.version_id,max(score) AS score FROM scores JOIN eligible e USING(passage_id)
                  GROUP BY e.version_id ORDER BY score DESC,e.version_id""",
                (jsonb(self.mapping), jsonb(dict(terms))),
            ).fetchall()
        return {r["version_id"]: r["score"] for r in rows}

    def grounded_matches(self, matches, library=None):
        library = library or self.owner
        if library in self.mapping.values() and not matches:
            raise CloudUnavailable(
                "The vector index is not ready for this source snapshot; no empty substitute result was used"
            )
        ids = [m.get("id") for m in matches]
        if any(not isinstance(i, str) or len(i) != 64 for i in ids) or len(
            set(ids)
        ) != len(ids):
            raise CloudUnavailable("Invalid vector identities")
        with connect() as c:
            rows = c.execute(
                """SELECT o.*,p.text,v.metadata,v.text_sha256,v.source_id AS source_family_id
              FROM st_occurrences o JOIN st_passages p USING(owner_id,passage_id)
              JOIN st_versions v USING(owner_id,version_id)
              JOIN jsonb_each_text(%s) s ON o.version_id=s.key AND o.owner_id=s.value::uuid
              WHERE o.owner_id=%s AND o.occurrence_id=ANY(%s) AND o.indexed""",
                (jsonb(self.mapping), library, ids),
            ).fetchall()
        by_id = {r["occurrence_id"]: r for r in rows}
        result = []
        for match in matches:
            r = by_id.get(match["id"])
            meta = match.get("metadata", {})
            score = match.get("score")
            if r is None:
                raise CloudUnavailable("Vector/source index consistency check failed")
            if (
                meta.get("space") != self.pine.space
                or meta.get("version_id") != r["version_id"]
                or meta.get("passage_id") != r["passage_id"]
                or meta.get("source_id") != r["source_family_id"]
                or digest(r["text"]) != r["passage_id"]
            ):
                raise CloudUnavailable("Vector/source provenance mismatch")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not -1.001 <= score <= 1.001
            ):
                raise CloudUnavailable("Invalid cosine score")
            result.append((r, float(score)))
        return sorted(
            result,
            key=lambda x: (
                -x[1],
                x[0]["source_id"],
                x[0]["start_char"],
                x[0]["passage_id"],
            ),
        )

    def rank(self, units):
        queries = [u.text if hasattr(u, "text") else u for u in units]
        lexical_scores = {}
        dense_scores = {}
        related = []
        for start in range(0, len(queries), 8):
            queries_batch = queries[start : start + 8]
            embeddings = local_encoder().encode(queries_batch)
            for offset, (query, vector) in enumerate(zip(queries_batch, embeddings)):
                for sid, score in self.bm25(query).items():
                    lexical_scores[sid] = max(lexical_scores.get(sid, 0), score)
                merged = {}
                for library, pine in self.pines.items():
                    versions = [v for v, key in self.mapping.items() if key == library]
                    for row, score in self.grounded_matches(
                        pine.query(vector, versions), library
                    ):
                        identity = (
                            row["version_id"],
                            row["start_char"],
                            row["end_char"],
                            row["passage_id"],
                        )
                        if identity not in merged or score > merged[identity][1]:
                            merged[identity] = (row, score)
                rows = sorted(
                    merged.values(),
                    key=lambda x: (
                        -x[1],
                        x[0]["source_id"],
                        x[0]["start_char"],
                        x[0]["passage_id"],
                    ),
                )
                rows = rows[: int(os.getenv("SEMATRACE_PINECONE_TOP_K", "256"))]
                for row, score in rows:
                    dense_scores[row["version_id"]] = max(
                        dense_scores.get(row["version_id"], -1), score
                    )
                for rank, (row, score) in enumerate(rows[:8], 1):
                    o = occurrence(row)
                    related.append(
                        dict(
                            unit_index=start + offset,
                            source_document_id=o.source_document_id,
                            source_start=o.source_start_char,
                            source_end=o.source_end_char,
                            source_excerpt=row["text"],
                            source_title=o.title,
                            source_url=o.url,
                            source_revision=o.revision,
                            source_document_sha256=o.source_document_sha256,
                            source_passage_id=row["passage_id"],
                            source_library_scope=library_scope(row["owner_id"]),
                            source_attribution={
                                k: getattr(o, k)
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
                            minilm_rank=rank,
                            score=score,
                            category="POTENTIAL_SEMANTIC_CORRESPONDENCE",
                            score_eligible=False,
                            boundary_granularity="retrieved_passage",
                            uncertainty="Relatedness only; reuse is not established.",
                        )
                    )
        lexical = sorted(lexical_scores, key=lambda s: (-lexical_scores[s], s))
        dense = sorted(dense_scores, key=lambda s: (-dense_scores[s], s))
        fusion = {sid: 0.0 for sid in self.owners}
        for ranking in (lexical, dense):
            for rank, sid in enumerate(ranking, 1):
                fusion[sid] += 1 / (60 + rank)
        fused = sorted(fusion, key=lambda s: (-fusion[s], s))
        return dict(
            rankings=dict(bm25=lexical, minilm=dense, rrf=fused),
            documents=fused,
            related=related,
            query_count=len(queries),
            query_batch_size=8,
            generation_id=self.generation_id,
            method="Eligible global/private BM25 + merged Pinecone cosine; RRF k=60",
        )
