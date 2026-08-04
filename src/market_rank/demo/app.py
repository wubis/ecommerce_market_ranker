"""Streamlit entry point for the thin MarketRank portfolio demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import streamlit as st

from market_rank.config import ConfigError, DemoConfig, load_config
from market_rank.demo.client import DemoApiClient, DemoClientError
from market_rank.demo.presentation import ComparisonReport, compare_responses
from market_rank.serving.contracts import (
    ReadinessResponse,
    SearchMode,
    SearchRequest,
    SearchResponse,
)

DEFAULT_CONFIG = Path("configs/base.yaml")
MODE_LABELS: dict[SearchMode, str] = {
    "active": "Active champion",
    "bm25": "BM25",
    "dense": "Dense MiniLM",
    "hybrid": "Hybrid RRF",
    "pointwise": "Pointwise LightGBM",
    "lambdamart": "LambdaMART",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments, _ = parser.parse_known_args()
    return arguments


def _limitations() -> None:
    with st.expander("Dataset limitations — always read before interpreting results"):
        st.markdown(
            """
            Amazon ESCI contains bounded judged product lists, not exhaustive judgments over a
            live catalog. A product without a judgment is **unknown**, not irrelevant. This demo
            therefore shows no relevance metric for arbitrary searches.

            ESCI does not provide authoritative live price, inventory, seller, shipping,
            fulfillment, review, margin, conversion, returns, sponsorship, product age, or
            canonical category fields. Results are from a fixed research catalog and must not be
            described as current Amazon search behavior. Brand and title-list statistics below
            are transparent presentation diagnostics, not business, fairness, or causal claims.
            """
        )


def _readiness(client: DemoApiClient) -> ReadinessResponse | None:
    try:
        readiness = client.ready()
    except DemoClientError as exc:
        st.warning(
            "The local MarketRank API is not ready. Start the explicit serving bundle first. "
            f"Reason: `{exc.reason_code}`."
        )
        return None
    if readiness.degraded:
        unavailable = ", ".join(
            item.component for item in readiness.components if item.state != "ready"
        )
        st.warning(f"API is ready in degraded mode. Unavailable components: {unavailable}.")
    else:
        st.success(
            f"API ready · active stage `{readiness.active_stage}` · "
            f"bundle `{(readiness.bundle_id or '')[-12:]}`"
        )
    return readiness


def _summary_rows(report: ComparisonReport) -> list[dict[str, object]]:
    return [
        {
            "Mode": MODE_LABELS[summary.requested_mode],
            "Resolved stage": summary.resolved_stage,
            "Total ms": round(summary.total_ms, 2),
            "Candidates": summary.candidate_count,
            "Results": summary.metrics.result_count,
            "Unique brands": summary.metrics.unique_brand_count,
            "Max brand share": round(summary.metrics.maximum_brand_concentration, 3),
            "Brand entropy (bits)": round(summary.metrics.brand_entropy_bits, 3),
            "Title-token ILD": round(summary.metrics.title_token_ild, 3),
            "Fallbacks": ", ".join(summary.fallback_reason_codes) or "none",
        }
        for summary in report.summaries
    ]


def _rank_rows(report: ComparisonReport) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for change in report.rank_changes:
        row: dict[str, object] = {
            "Product": change.product_id,
            "Title": change.title,
            "Brand": change.brand or "(missing)",
        }
        for position in change.positions:
            label = MODE_LABELS[position.mode]
            row[f"{label} rank"] = position.rank
            if position.mode != report.baseline_mode:
                row[f"{label} Δ"] = position.change_from_baseline
        rows.append(row)
    return rows


def _timing_rows(response: SearchResponse) -> list[dict[str, object]]:
    timings = response.timings
    return [
        {"Stage": stage.replace("_", " ").title(), "Milliseconds": round(value, 3)}
        for stage, value in timings.model_dump().items()
    ]


def _render_product_cards(response: SearchResponse, maximum: int) -> None:
    st.subheader(f"{MODE_LABELS[response.requested_mode]} product cards")
    if not response.results:
        st.info("No products were returned for this mode.")
        return
    for result in response.results[:maximum]:
        with st.container(border=True):
            heading, score = st.columns((4, 1))
            heading.markdown(f"#### {result.rank}. {result.title or 'Untitled product'}")
            heading.caption(
                f"Product `{result.product_id}` · brand `{result.brand or '(missing)'}` · "
                f"color `{result.color or '(missing)'}`"
            )
            score.metric(result.score_field, f"{result.score:.5f}")
            if result.description_snippet:
                st.write(result.description_snippet)
            elif result.bullets:
                st.write(result.bullets)
            else:
                st.caption("No description or bullet text is available in the fixed catalog.")
            provenance = result.provenance
            st.caption(
                f"BM25 rank {provenance.bm25_rank or '—'} · "
                f"dense rank {provenance.dense_rank or '—'} · "
                f"RRF rank {provenance.rrf_rank} · {provenance.source_count} source(s)"
            )
            if result.debug is not None:
                with st.expander("Bounded model feature values"):
                    st.dataframe(
                        [
                            {"Feature": name, "Value": value}
                            for name, value in result.debug.feature_values
                        ],
                        hide_index=True,
                        use_container_width=True,
                    )


def _render_comparison(responses: tuple[SearchResponse, ...], demo: DemoConfig) -> None:
    report = compare_responses(responses)
    if any(response.degraded for response in responses):
        st.warning("At least one displayed mode used a degraded or fallback path.")

    st.subheader("Mode comparison")
    st.caption(
        "Brand and title-token statistics describe only the returned list. They are not "
        "relevance, fairness, or semantic-quality metrics."
    )
    st.dataframe(_summary_rows(report), hide_index=True, use_container_width=True)

    st.subheader("Rank changes")
    st.caption(
        f"Baseline: {MODE_LABELS[report.baseline_mode]}. Positive Δ means a product moved "
        "toward rank 1; blank means the product was absent from that mode's displayed top-K."
    )
    st.dataframe(_rank_rows(report), hide_index=True, use_container_width=True)

    selected_mode = st.selectbox(
        "Inspect one result list",
        options=tuple(response.requested_mode for response in responses),
        format_func=lambda mode: MODE_LABELS[mode],
    )
    selected = next(response for response in responses if response.requested_mode == selected_mode)
    if selected.fallbacks:
        st.info(
            "Fallbacks: "
            + "; ".join(
                f"{event.component}: {event.reason_code} → {event.resolved_stage}"
                for event in selected.fallbacks
            )
        )
    _render_product_cards(selected, demo.max_product_cards)

    with st.expander("Latency breakdown and reproducibility identifiers"):
        st.dataframe(_timing_rows(selected), hide_index=True, use_container_width=True)
        st.code(
            "\n".join(
                (
                    f"bundle_id={selected.bundle_id}",
                    f"catalog_id={selected.catalog_id}",
                    f"config_sha256={selected.config_sha256}",
                    f"query_sha256={selected.query_sha256}",
                    f"promoted_stage={selected.promoted_stage}",
                    f"resolved_stage={selected.resolved_stage}",
                )
            ),
            language="text",
        )


def main() -> None:
    st.set_page_config(page_title="MarketRank Portfolio Demo", page_icon="🔎", layout="wide")
    st.title("MarketRank")
    st.caption("CPU-first e-commerce retrieval and learning-to-rank · fixed ESCI research catalog")
    _limitations()

    try:
        config = load_config([_arguments().config])
    except ConfigError as exc:
        st.error(f"Demo configuration is invalid: {exc}")
        return
    demo = config.config.demo
    try:
        client = DemoApiClient(
            demo.api_base_url,
            timeout_seconds=demo.request_timeout_seconds,
        )
    except DemoClientError as exc:
        st.error(f"Demo client configuration failed: {exc}")
        return
    with client:
        readiness = _readiness(client)
        if readiness is None:
            st.session_state.pop("market_rank_responses", None)

        with st.form("market_rank_search"):
            example = st.selectbox(
                "ESCI-compatible example",
                options=("Custom query", *demo.example_queries),
            )
            custom_query = st.text_input(
                "Search query",
                placeholder="e.g. wireless mouse",
                disabled=example != "Custom query",
            )
            modes = st.multiselect(
                "Ranking modes",
                options=tuple(MODE_LABELS),
                default=("active", "hybrid", "lambdamart"),
                max_selections=demo.max_comparison_modes,
                format_func=lambda mode: MODE_LABELS[mode],
            )
            top_k = st.slider(
                "Products per mode",
                min_value=1,
                max_value=config.config.serving.max_response_top_k,
                value=demo.default_top_k,
            )
            option_columns = st.columns(3)
            neural = option_columns[0].checkbox("Request neural reranking")
            diversify = option_columns[1].checkbox("Request diversification")
            explain = option_columns[2].checkbox("Show bounded feature values")
            submitted = st.form_submit_button(
                "Compare rankings",
                type="primary",
                disabled=readiness is None,
                use_container_width=True,
            )

        if neural or diversify:
            st.caption(
                "Optional stages are passed to the API. If they are not promoted, the response "
                "will show explicit `not_promoted` fallback reasons."
            )
        if submitted:
            query = custom_query if example == "Custom query" else example
            if not query.strip():
                st.error("Enter a nonempty query or select an example.")
            elif not modes:
                st.error("Select at least one ranking mode.")
            else:
                request = SearchRequest(
                    query=query,
                    top_k=top_k,
                    neural_rerank=neural,
                    diversify=diversify,
                )
                try:
                    responses = client.compare(
                        request,
                        modes,
                        max_modes=demo.max_comparison_modes,
                        explain=explain,
                    )
                except DemoClientError as exc:
                    st.error(f"Search failed safely: {exc}. Reason: `{exc.reason_code}`.")
                    st.session_state.pop("market_rank_responses", None)
                else:
                    st.session_state["market_rank_responses"] = tuple(
                        response.model_dump(mode="json") for response in responses
                    )

        stored = st.session_state.get("market_rank_responses")
        if isinstance(stored, tuple):
            try:
                responses = tuple(SearchResponse.model_validate(item) for item in stored)
                _render_comparison(responses, demo)
            except (ValueError, TypeError):
                st.session_state.pop("market_rank_responses", None)
                st.error("Stored demo results no longer match the response contract; run again.")


if __name__ == "__main__":
    main()
