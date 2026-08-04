"""Thin command-line interface for explicit MarketRank lifecycle operations."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from market_rank.artifacts import ArtifactError
from market_rank.config import ConfigError, load_config
from market_rank.data.download import DownloadPolicy, download_validate_esci
from market_rank.data.esci_raw import RawDataError, RawDataValidationError, load_release_manifest
from market_rank.data.foundation import DataFoundationError, build_esci_foundation
from market_rank.data.profiles import ProfileBuildError, build_esci_profiles
from market_rank.evaluation.retrieval import (
    RetrievalEvaluationError,
    build_retrieval_evaluation,
)
from market_rank.retrieval.dense import (
    DenseRetrievalError,
    build_dense_index,
    cache_dense_model,
)
from market_rank.retrieval.sparse import SparseRetrievalError, build_sparse_index

DEFAULT_ESCI_MANIFEST = Path("configs/data/esci-release-7916cdf6ab75.json")
DEFAULT_CONFIG = Path("configs/base.yaml")


class CodeRevisionError(RawDataError):
    """Raised when the CLI cannot record a repository revision."""


def _find_repository_root(start: Path) -> Path:
    candidate = start.resolve(strict=False)
    for directory in (candidate, *candidate.parents):
        if (directory / ".git").exists():
            return directory
    raise CodeRevisionError("cannot locate a Git repository; pass --code-revision explicitly")


def detect_code_revision(repository_root: Path) -> str:
    """Return HEAD with an explicit dirty suffix for reproducible lineage."""
    try:
        head = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise CodeRevisionError(
            f"cannot determine code revision; pass --code-revision explicitly: {exc}"
        ) from exc
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise CodeRevisionError("Git returned an invalid HEAD revision")
    return f"{head}-dirty" if status else head


def build_parser() -> argparse.ArgumentParser:
    """Build the stable MarketRank CLI grammar without performing I/O."""
    parser = argparse.ArgumentParser(prog="market-rank")
    command_parsers = parser.add_subparsers(dest="command", required=True)

    data_parser = command_parsers.add_parser("data", help="raw dataset lifecycle commands")
    data_commands = data_parser.add_subparsers(dest="data_command", required=True)
    download_parser = data_commands.add_parser(
        "download-esci",
        help="explicitly download, verify, validate, and publish pinned ESCI data",
    )
    download_parser.add_argument("--manifest", type=Path, default=DEFAULT_ESCI_MANIFEST)
    download_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    download_parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="override configured data/raw/esci destination",
    )
    download_parser.add_argument(
        "--code-revision",
        default=None,
        help="explicit lineage revision when Git discovery is unavailable",
    )
    download_parser.add_argument("--connect-timeout", type=float, default=15.0)
    download_parser.add_argument("--read-timeout", type=float, default=60.0)
    download_parser.add_argument("--attempts", type=int, default=3)

    profiles_parser = data_commands.add_parser(
        "build-esci-profiles",
        help="build leakage-safe Task-1 US splits and nested query profiles",
    )
    profiles_parser.add_argument("--manifest", type=Path, default=DEFAULT_ESCI_MANIFEST)
    profiles_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    profiles_parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="override configured data/raw/esci source",
    )
    profiles_parser.add_argument(
        "--code-revision",
        default=None,
        help="explicit lineage revision when Git discovery is unavailable",
    )

    foundation_parser = data_commands.add_parser(
        "build-esci-foundation",
        help="build canonical tables, fixed catalog, judged pools, and product documents",
    )
    foundation_parser.add_argument("--manifest", type=Path, default=DEFAULT_ESCI_MANIFEST)
    foundation_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    foundation_parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="override configured data/raw/esci source",
    )
    foundation_parser.add_argument(
        "--code-revision",
        default=None,
        help="explicit lineage revision when Git discovery is unavailable",
    )

    retrieval_parser = command_parsers.add_parser(
        "retrieval", help="persisted candidate-retrieval lifecycle commands"
    )
    retrieval_commands = retrieval_parser.add_subparsers(dest="retrieval_command", required=True)
    sparse_parser = retrieval_commands.add_parser(
        "build-bm25",
        help="build the deterministic persisted BM25 fixed-catalog index",
    )
    sparse_parser.add_argument("--manifest", type=Path, default=DEFAULT_ESCI_MANIFEST)
    sparse_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sparse_parser.add_argument(
        "--code-revision",
        default=None,
        help="explicit lineage revision when Git discovery is unavailable",
    )
    cache_dense_parser = retrieval_commands.add_parser(
        "cache-minilm",
        help="explicitly cache the pinned MiniLM snapshot before an offline dense build",
    )
    cache_dense_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    cache_dense_parser.add_argument(
        "--allow-network",
        action="store_true",
        help="explicitly permit the pinned Hugging Face snapshot download",
    )
    dense_parser = retrieval_commands.add_parser(
        "build-dense",
        help="build/resume normalized MiniLM vectors and an exact FAISS CPU index",
    )
    dense_parser.add_argument("--manifest", type=Path, default=DEFAULT_ESCI_MANIFEST)
    dense_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    dense_parser.add_argument(
        "--code-revision",
        default=None,
        help="explicit lineage revision when Git discovery is unavailable",
    )
    hybrid_parser = retrieval_commands.add_parser(
        "evaluate-hybrid",
        help="persist fixed-cohort BM25/dense/RRF candidates, metrics, CIs, and slices",
    )
    hybrid_parser.add_argument("--manifest", type=Path, default=DEFAULT_ESCI_MANIFEST)
    hybrid_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    hybrid_parser.add_argument(
        "--profile",
        choices=("development", "portfolio"),
        default=None,
        help="evaluation profile; defaults to evaluation.default_profile",
    )
    hybrid_parser.add_argument(
        "--code-revision",
        default=None,
        help="explicit lineage revision when Git discovery is unavailable",
    )
    return parser


def _failed_check_count(error: RawDataValidationError) -> int:
    report = error.report
    return sum(not check.passed for file in report.files for check in file.checks) + sum(
        not check.passed for check in report.dataset_checks
    )


def _run_download_esci(arguments: argparse.Namespace) -> int:
    config = load_config([arguments.config])
    release = load_release_manifest(arguments.manifest)
    code_revision = arguments.code_revision
    if code_revision is None:
        repository_root = _find_repository_root(config.source_paths[0].parent)
        code_revision = detect_code_revision(repository_root)

    policy = DownloadPolicy(
        connect_timeout_s=arguments.connect_timeout,
        read_timeout_s=arguments.read_timeout,
        max_attempts=arguments.attempts,
    )
    download_validate_esci(
        release,
        config,
        code_revision=code_revision,
        raw_root=arguments.raw_dir,
        policy=policy,
        progress=print,
    )
    return 0


def _resolve_code_revision(arguments: argparse.Namespace, config_path: Path) -> str:
    if arguments.code_revision is not None:
        return str(arguments.code_revision)
    repository_root = _find_repository_root(config_path.resolve(strict=False).parent)
    return detect_code_revision(repository_root)


def _run_build_esci_profiles(arguments: argparse.Namespace) -> int:
    config = load_config([arguments.config])
    release = load_release_manifest(arguments.manifest)
    result = build_esci_profiles(
        release,
        config,
        code_revision=_resolve_code_revision(arguments, arguments.config),
        raw_root=arguments.raw_dir,
    )
    development, portfolio = result.manifest.profiles
    print(
        f"Task-1 US: {result.manifest.task1_us_query_ids} query IDs, "
        f"{result.manifest.task1_us_judgments} judgments"
    )
    print(
        f"quarantined: {result.manifest.quarantined_train_query_ids} train query IDs "
        "colliding with official test"
    )
    print(
        f"development: {development.selected_normalized_query_groups} normalized-query groups; "
        f"portfolio: {portfolio.selected_normalized_query_groups} normalized-query groups"
    )
    status = "reused" if result.reused else "published"
    print(f"{status} profile artifact: {result.artifact.manifest.artifact_id}")
    return 0


def _run_build_esci_foundation(arguments: argparse.Namespace) -> int:
    config = load_config([arguments.config])
    release = load_release_manifest(arguments.manifest)
    result = build_esci_foundation(
        release,
        config,
        code_revision=_resolve_code_revision(arguments, arguments.config),
        raw_root=arguments.raw_dir,
    )
    development, portfolio = result.manifest.pools
    resource = result.manifest.resource_estimate
    print(f"canonical portfolio: {portfolio.query_ids} queries, {portfolio.judgments} judgments")
    print(
        f"fixed catalog: {result.manifest.catalog_products} products "
        f"({result.manifest.catalog_excluded_no_text} excluded without text)"
    )
    print(
        f"development pool: {development.query_ids} queries; resource gate: "
        f"{resource.projected_runtime_bytes}/{resource.rss_limit_bytes} bytes, proceed"
    )
    status = "reused" if result.reused else "published"
    print(f"{status} data foundation: {result.artifact.manifest.artifact_id}")
    return 0


def _run_build_bm25(arguments: argparse.Namespace) -> int:
    config = load_config([arguments.config])
    release = load_release_manifest(arguments.manifest)
    result = build_sparse_index(
        release,
        config,
        code_revision=_resolve_code_revision(arguments, arguments.config),
    )
    metadata = result.metadata
    print(
        f"BM25 index: {metadata.document_count} documents, "
        f"{metadata.vocabulary_size} terms, {metadata.posting_count} postings"
    )
    print(
        f"resource: {metadata.resource.index_payload_bytes} artifact bytes, "
        f"peak RSS {metadata.resource.peak_rss_bytes}/{metadata.resource.rss_limit_bytes} bytes"
    )
    status = "reused" if result.reused else "published"
    print(f"{status} sparse index: {result.artifact.manifest.artifact_id}")
    return 0


def _run_cache_minilm(arguments: argparse.Namespace) -> int:
    config = load_config([arguments.config])
    path = cache_dense_model(config, allow_network=bool(arguments.allow_network))
    dense = config.config.retrieval.dense
    print(f"cached model: {dense.model_id}@{dense.model_revision}")
    print(f"snapshot: {path}")
    return 0


def _run_build_dense(arguments: argparse.Namespace) -> int:
    config = load_config([arguments.config])
    release = load_release_manifest(arguments.manifest)
    result = build_dense_index(
        release,
        config,
        code_revision=_resolve_code_revision(arguments, arguments.config),
    )
    metadata = result.metadata
    print(
        f"dense index: {metadata.document_count} documents x "
        f"{metadata.embedding_dimension} float32 dimensions"
    )
    print(
        f"resource: {metadata.resource.artifact_payload_bytes} payload bytes, "
        f"peak RSS {metadata.resource.peak_rss_bytes}/{metadata.resource.rss_limit_bytes} bytes"
    )
    print(
        f"warm dense latency: p50 {metadata.latency.p50_ms:.3f} ms, "
        f"p95 {metadata.latency.p95_ms:.3f} ms over {metadata.latency.sample_queries} queries"
    )
    status = "reused" if result.reused else "published"
    print(f"{status} dense index: {result.artifact.manifest.artifact_id}")
    return 0


def _run_evaluate_hybrid(arguments: argparse.Namespace) -> int:
    config = load_config([arguments.config])
    release = load_release_manifest(arguments.manifest)
    result = build_retrieval_evaluation(
        release,
        config,
        code_revision=_resolve_code_revision(arguments, arguments.config),
        profile=arguments.profile,
    )
    manifest = result.manifest
    stages = ", ".join(
        f"{stage.stage}={stage.candidate_rows} candidates/{stage.empty_queries} empty"
        for stage in manifest.stages
    )
    print(f"retrieval cohort: {manifest.profile}, {manifest.query_count} queries")
    print(f"stages: {stages}")
    print(
        f"report: {manifest.aggregate_metric_rows} aggregate rows, "
        f"{manifest.comparison_metric_rows} paired comparison rows"
    )
    print(
        f"combined resource: peak RSS {manifest.resource.peak_rss_bytes}/"
        f"{manifest.resource.rss_limit_bytes} bytes"
    )
    status = "reused" if result.reused else "published"
    print(f"{status} retrieval evaluation: {result.artifact.manifest.artifact_id}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run one explicit CLI command and return a bounded process exit code."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "data" and arguments.data_command == "download-esci":
            return _run_download_esci(arguments)
        if arguments.command == "data" and arguments.data_command == "build-esci-profiles":
            return _run_build_esci_profiles(arguments)
        if arguments.command == "data" and arguments.data_command == "build-esci-foundation":
            return _run_build_esci_foundation(arguments)
        if arguments.command == "retrieval" and arguments.retrieval_command == "build-bm25":
            return _run_build_bm25(arguments)
        if arguments.command == "retrieval" and arguments.retrieval_command == "cache-minilm":
            return _run_cache_minilm(arguments)
        if arguments.command == "retrieval" and arguments.retrieval_command == "build-dense":
            return _run_build_dense(arguments)
        if arguments.command == "retrieval" and arguments.retrieval_command == "evaluate-hybrid":
            return _run_evaluate_hybrid(arguments)
    except RawDataValidationError as exc:
        print(
            f"error: raw ESCI validation failed ({_failed_check_count(exc)} checks failed)",
            file=sys.stderr,
        )
        return 1
    except (
        ArtifactError,
        ConfigError,
        DataFoundationError,
        DenseRetrievalError,
        ProfileBuildError,
        RawDataError,
        RetrievalEvaluationError,
        SparseRetrievalError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
