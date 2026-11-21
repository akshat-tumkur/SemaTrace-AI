"""Explicit local retrieval administration; hash builds are offline by default."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app import config
from app.retrieval.generations import GenerationStore, IndexSpec, IndexUnavailable
from app.storage.source_corpus import SourceCorpusStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "inspect", "activate"))
    parser.add_argument("--corpus-db", type=Path)
    parser.add_argument("--index-db", type=Path)
    parser.add_argument("--corpus-version")
    parser.add_argument(
        "--provider", choices=("hash", "openai", "local-onnx"), default="hash"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--generation")
    parser.add_argument("--max-requests", type=int, default=64)
    parser.add_argument(
        "--allow-paid",
        action="store_true",
        help="Explicit permission for this OpenAI build only",
    )
    args = parser.parse_args(argv)
    if args.command == "build" and args.provider == "openai" and not args.allow_paid:
        parser.error(
            "OpenAI builds require --allow-paid and separately approved API use."
        )
    if (
        args.command == "build"
        and args.provider == "openai"
        and not config.OPENAI_API_KEY
    ):
        parser.error("OpenAI key unavailable; no live build ran.")
    corpus = SourceCorpusStore(args.corpus_db).snapshot(version_id=args.corpus_version)
    store = GenerationStore(args.index_db or corpus.retrieval_index_path)
    spec = (
        IndexSpec.hash(256 if args.dimensions is None else args.dimensions)
        if args.provider == "hash"
        else IndexSpec.openai(args.model, args.revision, args.dimensions)
    )
    if args.provider == "local-onnx":
        spec = IndexSpec.local()
        if (
            (args.model is not None and args.model != spec.model)
            or (args.revision is not None and args.revision != spec.revision)
            or (args.dimensions is not None and args.dimensions != spec.dimensions)
        ):
            parser.error(
                "Local model/revision/dimensions must match the verified MiniLM lock."
            )
        if args.batch_size > 64:
            parser.error("Local inference batches must be at most 64.")
    error = None
    try:
        if args.command == "build":
            generation, metrics = store.build(
                corpus,
                spec,
                api_key=config.OPENAI_API_KEY,
                max_requests=args.max_requests,
                batch_size=args.batch_size,
            )
        elif args.command == "activate":
            if not args.generation:
                parser.error(
                    "Activation requires --generation and the matching pinned corpus/spec."
                )
            store.activate(args.generation, corpus, spec)
            generation = store.load(args.generation, corpus, expected_spec=spec)
            metrics = {"physical_embedding_requests": 0}
        else:
            if args.generation:
                generation = store.load(args.generation, corpus, expected_spec=spec)
            else:
                generation, error = store.active(corpus, spec)
                if generation is None:
                    raise IndexUnavailable("compatible_generation_missing")
            metrics = {"physical_embedding_requests": 0}
        print(
            json.dumps(
                dict(
                    generation_id=generation.id,
                    corpus_version=generation.manifest["corpus_version"],
                    passage_count=len(generation.vectors),
                    spec=generation.manifest["spec"],
                    space=generation.manifest["space"],
                    index_error=error,
                    metrics=metrics,
                )
            )
        )
    except IndexUnavailable as e:
        parser.exit(1, str(e) + "\n")


if __name__ == "__main__":
    main()
