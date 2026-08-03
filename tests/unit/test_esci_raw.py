"""Tests for the pinned ESCI raw manifest and structural validator."""

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import polars as pl
import pytest

from market_rank.artifacts import ArtifactStore
from market_rank.data.esci_raw import (
    OFFICIAL_PAPER,
    OFFICIAL_REPOSITORY,
    EsciReleaseManifest,
    RawDataError,
    RawDataValidationError,
    RawFileSource,
    RawManifestError,
    RawValidationReport,
    ResolvedReleaseManifest,
    load_release_manifest,
    publish_raw_validation,
    validate_raw_dataset,
)

REPOSITORY_ROOT = Path(__file__).parents[2]
PINNED_RELEASE = REPOSITORY_ROOT / "configs" / "data" / "esci-release-7916cdf6ab75.json"
REVISION = "7916cdf6ab75a462e77f20ab40428a10923998d5"
RETRIEVED_UTC = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
CONFIG_SHA256 = "a" * 64


def _examples() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "example_id": [1, 2, 3],
            "query": ["mouse", "mouse", "shoes"],
            "query_id": [10, 10, 11],
            "product_id": ["p1", "p2", "p3"],
            "product_locale": ["us", "us", "us"],
            "esci_label": ["E", "S", "I"],
            "small_version": [1, 1, 1],
            "large_version": [1, 1, 1],
            "split": ["train", "train", "test"],
        },
        schema={
            "example_id": pl.Int64,
            "query": pl.String,
            "query_id": pl.Int64,
            "product_id": pl.String,
            "product_locale": pl.String,
            "esci_label": pl.String,
            "small_version": pl.Int8,
            "large_version": pl.Int8,
            "split": pl.String,
        },
    )


def _products() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "product_id": ["p1", "p2", "p3"],
            "product_title": ["Mouse", "Mouse pad", "Shoes"],
            "product_description": [None, "Pad", "Running shoes"],
            "product_bullet_point": ["Wireless", None, "Red"],
            "product_brand": ["A", "B", None],
            "product_color": ["Black", None, "Red"],
            "product_locale": ["us", "us", "us"],
        },
        schema={
            "product_id": pl.String,
            "product_title": pl.String,
            "product_description": pl.String,
            "product_bullet_point": pl.String,
            "product_brand": pl.String,
            "product_color": pl.String,
            "product_locale": pl.String,
        },
    )


def _sources() -> pl.DataFrame:
    return pl.DataFrame(
        {"query_id": [10, 11], "source": ["other", "negations"]},
        schema={"query_id": pl.Int64, "source": pl.String},
    )


def _write_raw_files(
    raw_root: Path,
    *,
    examples: pl.DataFrame | None = None,
    products: pl.DataFrame | None = None,
    sources: pl.DataFrame | None = None,
) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    (examples if examples is not None else _examples()).write_parquet(
        raw_root / "shopping_queries_dataset_examples.parquet"
    )
    (products if products is not None else _products()).write_parquet(
        raw_root / "shopping_queries_dataset_products.parquet"
    )
    (sources if sources is not None else _sources()).write_csv(
        raw_root / "shopping_queries_dataset_sources.csv"
    )


def _file_digest(path: Path) -> tuple[int, str]:
    content = path.read_bytes()
    return len(content), sha256(content).hexdigest()


def _fixture_release(raw_root: Path, manifest_path: Path) -> ResolvedReleaseManifest:
    sources: list[RawFileSource] = []
    for role, filename, file_format in (
        ("examples", "shopping_queries_dataset_examples.parquet", "parquet"),
        ("products", "shopping_queries_dataset_products.parquet", "parquet"),
        ("sources", "shopping_queries_dataset_sources.csv", "csv"),
    ):
        size_bytes, file_sha256 = _file_digest(raw_root / filename)
        sources.append(
            RawFileSource.model_validate(
                {
                    "role": role,
                    "filename": filename,
                    "format": file_format,
                    "source_url": (
                        f"{OFFICIAL_REPOSITORY}/raw/{REVISION}/shopping_queries_dataset/{filename}"
                    ),
                    "size_bytes": size_bytes,
                    "sha256": file_sha256,
                }
            )
        )

    manifest = EsciReleaseManifest.model_validate(
        {
            "dataset_version": f"esci-{REVISION[:12]}",
            "source_repository": OFFICIAL_REPOSITORY,
            "source_revision": REVISION,
            "source_commit_utc": datetime(2024, 10, 7, 15, 52, 6, tzinfo=UTC),
            "license_url": f"{OFFICIAL_REPOSITORY}/blob/{REVISION}/LICENSE",
            "paper_url": OFFICIAL_PAPER,
            "files": tuple(sources),
        }
    )
    manifest_path.write_text(
        json.dumps(
            manifest.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return load_release_manifest(manifest_path)


def _validate_fixture(
    tmp_path: Path,
    *,
    examples: pl.DataFrame | None = None,
    products: pl.DataFrame | None = None,
    sources: pl.DataFrame | None = None,
) -> tuple[ResolvedReleaseManifest, RawValidationReport]:
    raw_root = tmp_path / "raw"
    _write_raw_files(raw_root, examples=examples, products=products, sources=sources)
    release = _fixture_release(raw_root, tmp_path / "release.json")
    report = validate_raw_dataset(release, raw_root, retrieved_utc=RETRIEVED_UTC)
    return release, report


def test_checked_in_release_pins_official_file_identity() -> None:
    release = load_release_manifest(PINNED_RELEASE)

    assert release.manifest.source_revision == REVISION
    assert release.manifest.dataset_version == "esci-7916cdf6ab75"
    assert tuple(source.role for source in release.manifest.files) == (
        "examples",
        "products",
        "sources",
    )
    assert release.manifest.files[0].sha256 == (
        "4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a"
    )
    assert release.manifest.files[1].size_bytes == 1_108_857_465
    assert PINNED_RELEASE.read_text(encoding="utf-8").strip() == release.canonical_json


def test_valid_raw_dataset_can_be_atomically_published(tmp_path: Path) -> None:
    release, report = _validate_fixture(tmp_path)

    assert report.valid
    assert tuple(file.row_count for file in report.files) == (3, 3, 2)
    assert all(check.passed for check in report.dataset_checks)

    store = ArtifactStore(tmp_path / "artifacts")
    artifact = publish_raw_validation(
        release,
        report,
        store,
        config_sha256=CONFIG_SHA256,
        code_revision="abc123",
    )
    assert artifact.manifest.artifact_type == "raw-validation"
    assert artifact.manifest.dataset_version == release.manifest.dataset_version
    assert {file.relative_path for file in artifact.manifest.files} == {
        "release-manifest.json",
        "validation-report.json",
    }
    assert store.load(artifact.manifest.artifact_id) == artifact


def test_checksum_change_fails_before_schema_scan_and_promotion(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _write_raw_files(raw_root)
    release = _fixture_release(raw_root, tmp_path / "release.json")
    with (raw_root / "shopping_queries_dataset_sources.csv").open("a", encoding="utf-8") as stream:
        stream.write("12,other\n")

    report = validate_raw_dataset(release, raw_root, retrieved_utc=RETRIEVED_UTC)

    assert not report.valid
    source_checks = {check.check_id: check for check in report.files[2].checks}
    assert not source_checks["sha256"].passed
    assert source_checks["schema_readable"].detail == "skipped because integrity failed"
    with pytest.raises(RawDataValidationError) as caught:
        publish_raw_validation(
            release,
            report,
            ArtifactStore(tmp_path / "artifacts"),
            config_sha256=CONFIG_SHA256,
            code_revision="abc123",
        )
    assert caught.value.report == report


def test_missing_source_file_returns_complete_invalid_report(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _write_raw_files(raw_root)
    release = _fixture_release(raw_root, tmp_path / "release.json")
    (raw_root / "shopping_queries_dataset_products.parquet").unlink()

    report = validate_raw_dataset(release, raw_root, retrieved_utc=RETRIEVED_UTC)

    assert not report.valid
    assert report.files[1].size_bytes is None
    assert report.dataset_checks[0].check_id == "cross_files_ready"


def test_exact_columns_are_required(tmp_path: Path) -> None:
    invalid_examples = _examples().with_columns(pl.lit("unexpected").alias("extra"))
    _, report = _validate_fixture(tmp_path, examples=invalid_examples)

    assert not report.valid
    checks = {check.check_id: check for check in report.files[0].checks}
    assert not checks["exact_columns"].passed


def test_semantic_types_are_required(tmp_path: Path) -> None:
    invalid_examples = _examples().with_columns(pl.col("query_id").cast(pl.String))
    _, report = _validate_fixture(tmp_path, examples=invalid_examples)

    assert not report.valid
    checks = {check.check_id: check for check in report.files[0].checks}
    assert not checks["semantic_types"].passed


def test_label_domain_is_enforced(tmp_path: Path) -> None:
    invalid_examples = _examples().with_columns(
        pl.when(pl.col("example_id") == 1)
        .then(pl.lit("X"))
        .otherwise(pl.col("esci_label"))
        .alias("esci_label")
    )
    _, report = _validate_fixture(tmp_path, examples=invalid_examples)

    assert not report.valid
    checks = {check.check_id: check for check in report.files[0].checks}
    assert not checks["domain:esci_label"].passed


def test_primary_key_duplicates_are_rejected(tmp_path: Path) -> None:
    invalid_examples = pl.concat([_examples(), _examples().head(1)])
    _, report = _validate_fixture(tmp_path, examples=invalid_examples)

    assert not report.valid
    checks = {check.check_id: check for check in report.files[0].checks}
    assert not checks["primary_key_unique"].passed


def test_query_text_must_be_consistent_per_query_id(tmp_path: Path) -> None:
    inconsistent = pl.DataFrame(
        {
            "example_id": [4],
            "query": ["different query"],
            "query_id": [10],
            "product_id": ["p3"],
            "product_locale": ["us"],
            "esci_label": ["C"],
            "small_version": [1],
            "large_version": [1],
            "split": ["train"],
        },
        schema=_examples().schema,
    )
    _, report = _validate_fixture(tmp_path, examples=pl.concat([_examples(), inconsistent]))

    assert not report.valid
    checks = {check.check_id: check for check in report.files[0].checks}
    assert not checks["query_consistency"].passed


def test_every_example_product_key_must_join(tmp_path: Path) -> None:
    missing_product = _products().filter(pl.col("product_id") != "p3")
    _, report = _validate_fixture(tmp_path, products=missing_product)

    assert not report.valid
    checks = {check.check_id: check for check in report.dataset_checks}
    assert not checks["examples_join_products"].passed


def test_every_example_query_must_have_a_source(tmp_path: Path) -> None:
    missing_source = _sources().filter(pl.col("query_id") != 11)
    _, report = _validate_fixture(tmp_path, sources=missing_source)

    assert not report.valid
    checks = {check.check_id: check for check in report.dataset_checks}
    assert not checks["examples_join_sources"].passed


def test_release_manifest_rejects_unknown_and_duplicate_keys(tmp_path: Path) -> None:
    original = PINNED_RELEASE.read_text(encoding="utf-8")
    document = json.loads(original)
    document["unknown"] = True
    unknown_path = tmp_path / "unknown.json"
    unknown_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RawManifestError, match="invalid ESCI release manifest"):
        load_release_manifest(unknown_path)

    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text('{"schema_version":1,' + original[1:], encoding="utf-8")
    with pytest.raises(RawManifestError, match="duplicate JSON key"):
        load_release_manifest(duplicate_path)


def test_raw_symbolic_link_is_rejected(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _write_raw_files(raw_root)
    release = _fixture_release(raw_root, tmp_path / "release.json")
    source_path = raw_root / "shopping_queries_dataset_sources.csv"
    external_path = tmp_path / "external.csv"
    source_path.replace(external_path)
    source_path.symlink_to(external_path)

    report = validate_raw_dataset(release, raw_root, retrieved_utc=RETRIEVED_UTC)

    assert not report.valid
    assert "symbolic links" in report.files[2].checks[0].detail


def test_retrieved_timestamp_must_be_utc(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _write_raw_files(raw_root)
    release = _fixture_release(raw_root, tmp_path / "release.json")

    with pytest.raises(RawDataError, match="timezone-aware UTC"):
        validate_raw_dataset(
            release,
            raw_root,
            retrieved_utc=datetime(2026, 8, 1, 12, 0),
        )
