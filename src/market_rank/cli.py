"""Thin command-line interface for explicit MarketRank lifecycle operations."""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from market_rank.config import load_config
from market_rank.data.download import DownloadPolicy, download_validate_esci
from market_rank.data.esci_raw import RawDataError, RawDataValidationError, load_release_manifest
from market_rank.data.foundation import build_esci_foundation
from market_rank.data.profiles import build_esci_profiles
from market_rank.demo.client import DemoApiClient
from market_rank.evaluation.retrieval import (
    build_retrieval_evaluation,
)
from market_rank.features.artifact import build_ranking_features
from market_rank.retrieval.dense import (
    build_dense_index,
    cache_dense_model,
)
from market_rank.retrieval.sparse import build_sparse_index

DEFAULT_ESCI_MANIFEST = Path("configs/data/esci-release-7916cdf6ab75.json")
DEFAULT_CONFIG = Path("configs/base.yaml")


class CodeRevisionError(RawDataError):
    """Raised when the CLI cannot record a repository revision."""


class DemoLaunchError(RuntimeError):
    """Raised when the local Streamlit process cannot be launched cleanly."""


class CommandExecutionError(RuntimeError):
    """Raised when a lazily imported lifecycle component rejects a command."""


def _prime_torch_runtime() -> None:
    """Load PyTorch before LightGBM on macOS processes that need both runtimes."""
    try:
        importlib.import_module("torch")
    except ImportError as exc:
        raise CommandExecutionError(f"PyTorch is unavailable: {exc}") from exc


def build_rankers(*args: Any, **kwargs: Any) -> Any:
    from market_rank.ranking.training import build_rankers as implementation

    return implementation(*args, **kwargs)


def build_ranking_evaluation(*args: Any, **kwargs: Any) -> Any:
    from market_rank.evaluation.ranking import build_ranking_evaluation as implementation

    return implementation(*args, **kwargs)


def build_serving_bundle(*args: Any, **kwargs: Any) -> Any:
    from market_rank.serving.bundle import build_serving_bundle as implementation

    return implementation(*args, **kwargs)


def create_app(*args: Any, **kwargs: Any) -> Any:
    from market_rank.serving.api import create_app as implementation

    return implementation(*args, **kwargs)


def build_release_qualification(*args: Any, **kwargs: Any) -> Any:
    from market_rank.qualification import build_release_qualification as implementation

    return implementation(*args, **kwargs)


def verify_clean_reproduction(*args: Any, **kwargs: Any) -> Any:
    from market_rank.portfolio import verify_clean_reproduction as implementation

    return implementation(*args, **kwargs)


def build_portfolio_release(*args: Any, **kwargs: Any) -> Any:
    from market_rank.portfolio import build_portfolio_release as implementation

    return implementation(*args, **kwargs)


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
    feature_parser = command_parsers.add_parser(
        "features", help="versioned query-understanding and ranking-feature lifecycle commands"
    )
    feature_commands = feature_parser.add_subparsers(dest="feature_command", required=True)
    build_features_parser = feature_commands.add_parser(
        "build-ranking",
        help="persist parser state and bounded closed/candidate ltr_core_v1 matrices",
    )
    build_features_parser.add_argument("--manifest", type=Path, default=DEFAULT_ESCI_MANIFEST)
    build_features_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    build_features_parser.add_argument(
        "--profile",
        choices=("development", "portfolio"),
        default=None,
        help="feature profile; defaults to evaluation.default_profile",
    )
    build_features_parser.add_argument(
        "--code-revision",
        default=None,
        help="explicit lineage revision when Git discovery is unavailable",
    )
    ranking_parser = command_parsers.add_parser(
        "ranking", help="grouped supervised ranking-model lifecycle commands"
    )
    ranking_commands = ranking_parser.add_subparsers(dest="ranking_command", required=True)
    train_rankers_parser = ranking_commands.add_parser(
        "train",
        help="train identical-population pointwise LightGBM and LambdaMART models",
    )
    train_rankers_parser.add_argument("--manifest", type=Path, default=DEFAULT_ESCI_MANIFEST)
    train_rankers_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train_rankers_parser.add_argument(
        "--profile",
        choices=("development", "portfolio"),
        default=None,
        help="training profile; defaults to evaluation.default_profile",
    )
    train_rankers_parser.add_argument(
        "--code-revision",
        default=None,
        help="explicit lineage revision when Git discovery is unavailable",
    )
    evaluate_rankers_parser = ranking_commands.add_parser(
        "evaluate",
        help="evaluate validation ranking protocols and promote active relevance",
    )
    evaluate_rankers_parser.add_argument("--manifest", type=Path, default=DEFAULT_ESCI_MANIFEST)
    evaluate_rankers_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    evaluate_rankers_parser.add_argument(
        "--profile",
        choices=("development", "portfolio"),
        default=None,
        help="evaluation profile; defaults to evaluation.default_profile",
    )
    evaluate_rankers_parser.add_argument(
        "--code-revision",
        default=None,
        help="explicit lineage revision when Git discovery is unavailable",
    )
    serving_parser = command_parsers.add_parser(
        "serving", help="explicit relevance-bundle promotion and local API commands"
    )
    serving_commands = serving_parser.add_subparsers(dest="serving_command", required=True)
    promote_parser = serving_commands.add_parser(
        "promote", help="promote compatible Goldfish 006-012 artifacts into one serving bundle"
    )
    promote_parser.add_argument("--manifest", type=Path, default=DEFAULT_ESCI_MANIFEST)
    promote_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    promote_parser.add_argument(
        "--profile",
        choices=("development", "portfolio"),
        default=None,
        help="bundle profile; defaults to evaluation.default_profile",
    )
    promote_parser.add_argument(
        "--code-revision",
        default=None,
        help="explicit lineage revision when Git discovery is unavailable",
    )
    run_parser = serving_commands.add_parser(
        "run", help="run the local API from one explicit immutable bundle ID"
    )
    run_parser.add_argument("--bundle-id", required=True)
    run_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    demo_parser = command_parsers.add_parser(
        "demo", help="API-backed local Streamlit portfolio-demo commands"
    )
    demo_commands = demo_parser.add_subparsers(dest="demo_command", required=True)
    check_demo_parser = demo_commands.add_parser(
        "check", help="verify that the configured local MarketRank API is ready"
    )
    check_demo_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_demo_parser = demo_commands.add_parser("run", help="launch the thin local Streamlit client")
    run_demo_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    qualification_parser = command_parsers.add_parser(
        "qualification", help="fail-closed local release-qualification commands"
    )
    qualification_commands = qualification_parser.add_subparsers(
        dest="qualification_command", required=True
    )
    run_qualification_parser = qualification_commands.add_parser(
        "run", help="qualify one explicit serving bundle on the reference host"
    )
    run_qualification_parser.add_argument("--bundle-id", required=True)
    run_qualification_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_qualification_parser.add_argument(
        "--background-conditions",
        required=True,
        help="short operator record of power/background benchmark conditions",
    )
    run_qualification_parser.add_argument(
        "--code-revision",
        default=None,
        help="explicit lineage revision when Git discovery is unavailable",
    )
    portfolio_parser = command_parsers.add_parser(
        "portfolio", help="frozen project-test evaluation and final release-package commands"
    )
    portfolio_commands = portfolio_parser.add_subparsers(dest="portfolio_command", required=True)
    verify_portfolio_parser = portfolio_commands.add_parser(
        "verify-reproduction", help="run clean lock, static, and test gates"
    )
    verify_portfolio_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    verify_portfolio_parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/clean-reproduction.json"),
    )
    verify_portfolio_parser.add_argument("--code-revision", default=None)
    finalize_portfolio_parser = portfolio_commands.add_parser(
        "finalize", help="evaluate frozen project test and publish the final core package"
    )
    finalize_portfolio_parser.add_argument("--ranking-evaluation-id", required=True)
    finalize_portfolio_parser.add_argument("--serving-bundle-id", required=True)
    finalize_portfolio_parser.add_argument("--qualification-id", required=True)
    finalize_portfolio_parser.add_argument("--reproduction-evidence", type=Path, required=True)
    finalize_portfolio_parser.add_argument("--screenshots-dir", type=Path, required=True)
    finalize_portfolio_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    finalize_portfolio_parser.add_argument("--code-revision", default=None)
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


def _run_build_ranking_features(arguments: argparse.Namespace) -> int:
    config = load_config([arguments.config])
    release = load_release_manifest(arguments.manifest)
    result = build_ranking_features(
        release,
        config,
        code_revision=_resolve_code_revision(arguments, arguments.config),
        profile=arguments.profile,
    )
    manifest = result.manifest
    print(
        f"feature cohort: {manifest.profile}, {manifest.query_count} queries, "
        f"{manifest.feature_count} ordered features"
    )
    print(
        f"matrices: {manifest.closed_rows} labeled closed rows, "
        f"{manifest.candidate_rows} label-free candidate rows"
    )
    print(
        "closed exclusions: "
        f"{manifest.closed_excluded_outside_catalog} judged rows outside the fixed catalog"
    )
    print(
        f"resource: peak RSS {manifest.resource.peak_rss_bytes}/"
        f"{manifest.resource.rss_limit_bytes} bytes"
    )
    status = "reused" if result.reused else "published"
    print(f"{status} ranking features: {result.artifact.manifest.artifact_id}")
    return 0


def _run_train_rankers(arguments: argparse.Namespace) -> int:
    config = load_config([arguments.config])
    release = load_release_manifest(arguments.manifest)
    result = build_rankers(
        release,
        config,
        code_revision=_resolve_code_revision(arguments, arguments.config),
        profile=arguments.profile,
    )
    manifest = result.manifest
    population = manifest.population
    print(
        f"training population: {manifest.profile}, "
        f"{population.train.eligible_query_groups}/{population.train.eligible_rows} "
        "train groups/rows, "
        f"{population.validation.eligible_query_groups}/"
        f"{population.validation.eligible_rows} validation groups/rows"
    )
    for model in manifest.models:
        metrics = ", ".join(
            f"NDCG@{cutoff}={value:.4f}" for cutoff, value in model.validation_best_ndcg
        )
        print(f"{model.model_id}: iteration {model.best_iteration}; validation {metrics}")
    print(
        f"resource: peak RSS {manifest.resource.peak_rss_bytes}/"
        f"{manifest.resource.rss_limit_bytes} bytes"
    )
    status = "reused" if result.reused else "published"
    print(f"{status} ranking models: {result.artifact.manifest.artifact_id}")
    return 0


def _run_evaluate_rankers(arguments: argparse.Namespace) -> int:
    config = load_config([arguments.config])
    release = load_release_manifest(arguments.manifest)
    result = build_ranking_evaluation(
        release,
        config,
        code_revision=_resolve_code_revision(arguments, arguments.config),
        profile=arguments.profile,
    )
    manifest = result.manifest
    active = manifest.active_relevance
    print(
        f"ranking evaluation: {manifest.profile}, {manifest.validation_queries} validation "
        f"queries, {manifest.closed_rows} closed rows, {manifest.candidate_rows} candidate rows"
    )
    for candidate in active.candidates:
        metrics = ", ".join(
            f"NDCG@{cutoff}={value:.4f}" for cutoff, value in candidate.ndcg_by_cutoff
        )
        print(f"{candidate.stage}: {metrics}; eligible={str(candidate.eligible).lower()}")
    print(f"active relevance: {active.selected_stage}; test evaluated=false")
    print(
        f"resource: peak RSS {manifest.resource.peak_rss_bytes}/"
        f"{manifest.resource.rss_limit_bytes} bytes"
    )
    status = "reused" if result.reused else "published"
    print(f"{status} ranking evaluation: {result.artifact.manifest.artifact_id}")
    return 0


def _run_promote_serving(arguments: argparse.Namespace) -> int:
    config = load_config([arguments.config])
    release = load_release_manifest(arguments.manifest)
    result = build_serving_bundle(
        release,
        config,
        code_revision=_resolve_code_revision(arguments, arguments.config),
        profile=arguments.profile,
    )
    print(
        f"serving bundle: {result.manifest.product_count} products, "
        f"active stage {result.manifest.active_relevance.selected_stage}"
    )
    print(
        f"resource: peak RSS {result.manifest.resource.peak_rss_bytes}/"
        f"{result.manifest.resource.rss_limit_bytes} bytes"
    )
    status_text = "reused" if result.reused else "published"
    print(f"{status_text} serving bundle: {result.artifact.manifest.artifact_id}")
    return 0


def _run_serving_api(arguments: argparse.Namespace) -> int:
    import uvicorn

    _prime_torch_runtime()
    config = load_config([arguments.config])
    app = create_app(config, str(arguments.bundle_id))
    uvicorn.run(
        app,
        host=config.config.serving.bind_host,
        port=config.config.serving.port,
        workers=1,
    )
    return 0


def _run_demo_check(arguments: argparse.Namespace) -> int:
    config = load_config([arguments.config])
    demo = config.config.demo
    with DemoApiClient(demo.api_base_url, timeout_seconds=demo.request_timeout_seconds) as client:
        readiness = client.ready()
    state = "degraded" if readiness.degraded else "ready"
    print(f"demo API: {state}; active stage {readiness.active_stage}; bundle {readiness.bundle_id}")
    return 0


def _run_demo(arguments: argparse.Namespace) -> int:
    config = load_config([arguments.config])
    demo = config.config.demo
    app_path = Path(__file__).with_name("demo") / "app.py"
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.address",
                demo.bind_host,
                "--server.port",
                str(demo.port),
                "--browser.gatherUsageStats",
                "false",
                "--",
                "--config",
                str(arguments.config),
            ],
            check=False,
        )
    except OSError as exc:
        raise DemoLaunchError(f"cannot launch Streamlit: {exc}") from exc
    if completed.returncode != 0:
        raise DemoLaunchError(f"Streamlit exited with status {completed.returncode}")
    return 0


def _run_qualification(arguments: argparse.Namespace) -> int:
    _prime_torch_runtime()
    config = load_config([arguments.config])
    result = build_release_qualification(
        config,
        str(arguments.bundle_id),
        code_revision=_resolve_code_revision(arguments, arguments.config),
        background_conditions=str(arguments.background_conditions),
    )
    report = result.report
    mode_summary = ", ".join(
        f"{item.mode} p95={item.latency.p95_ms:.3f}ms" for item in report.mode_latencies
    )
    print(f"release qualification: pass; {mode_summary}")
    print(
        f"startup {report.startup_ms:.3f}ms; peak RSS "
        f"{report.peak_rss_bytes}/{report.rss_limit_bytes} bytes"
    )
    status = "reused" if result.reused else "published"
    print(f"{status} release qualification: {result.artifact.manifest.artifact_id}")
    return 0


def _run_portfolio_verify(arguments: argparse.Namespace) -> int:
    config = load_config([arguments.config])
    repository_root = _find_repository_root(arguments.config.resolve(strict=False).parent)
    evidence = verify_clean_reproduction(
        config,
        code_revision=_resolve_code_revision(arguments, arguments.config),
        repository_root=repository_root,
        output_path=arguments.output,
    )
    print(f"clean reproduction: {evidence.test_count} tests; evidence {arguments.output}")
    return 0


def _run_portfolio_finalize(arguments: argparse.Namespace) -> int:
    _prime_torch_runtime()
    config = load_config([arguments.config])
    result = build_portfolio_release(
        config,
        ranking_evaluation_id=str(arguments.ranking_evaluation_id),
        serving_bundle_id=str(arguments.serving_bundle_id),
        qualification_id=str(arguments.qualification_id),
        reproduction_evidence_path=arguments.reproduction_evidence,
        screenshots_dir=arguments.screenshots_dir,
        code_revision=_resolve_code_revision(arguments, arguments.config),
    )
    status = "reused" if result.reused else "published"
    print(
        f"portfolio release: {result.manifest.test_query_count} project-test queries; "
        f"active stage {result.manifest.active_stage}"
    )
    print(f"{status} portfolio release: {result.artifact.manifest.artifact_id}")
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
        if arguments.command == "features" and arguments.feature_command == "build-ranking":
            return _run_build_ranking_features(arguments)
        if arguments.command == "ranking" and arguments.ranking_command == "train":
            return _run_train_rankers(arguments)
        if arguments.command == "ranking" and arguments.ranking_command == "evaluate":
            return _run_evaluate_rankers(arguments)
        if arguments.command == "serving" and arguments.serving_command == "promote":
            return _run_promote_serving(arguments)
        if arguments.command == "serving" and arguments.serving_command == "run":
            return _run_serving_api(arguments)
        if arguments.command == "demo" and arguments.demo_command == "check":
            return _run_demo_check(arguments)
        if arguments.command == "demo" and arguments.demo_command == "run":
            return _run_demo(arguments)
        if arguments.command == "qualification" and arguments.qualification_command == "run":
            return _run_qualification(arguments)
        if (
            arguments.command == "portfolio"
            and arguments.portfolio_command == "verify-reproduction"
        ):
            return _run_portfolio_verify(arguments)
        if arguments.command == "portfolio" and arguments.portfolio_command == "finalize":
            return _run_portfolio_finalize(arguments)
    except RawDataValidationError as exc:
        print(
            f"error: raw ESCI validation failed ({_failed_check_count(exc)} checks failed)",
            file=sys.stderr,
        )
        return 1
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
