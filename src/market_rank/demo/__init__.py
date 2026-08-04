"""Thin API-backed portfolio demo for MarketRank."""

from market_rank.demo.client import DemoApiClient
from market_rank.demo.presentation import compare_responses, compute_list_metrics

__all__ = ["DemoApiClient", "compare_responses", "compute_list_metrics"]
