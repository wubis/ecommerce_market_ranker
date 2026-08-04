"""Deterministic validation-only Goldfish 012 champion-policy tests."""

from market_rank.config import RankingEvaluationConfig
from market_rank.evaluation.ranking import EvaluationStage, _champion_candidates


def test_champion_requires_improvement_without_material_regression() -> None:
    config = RankingEvaluationConfig(material_regression_tolerance=0.005)
    means: dict[tuple[EvaluationStage, int], float] = {
        ("rrf", 10): 0.50,
        ("rrf", 20): 0.50,
        ("pointwise", 10): 0.52,
        ("pointwise", 20): 0.49,
        ("lambdamart", 10): 0.52,
        ("lambdamart", 20): 0.499,
    }

    selected, candidates = _champion_candidates(means, config)

    assert selected == "lambdamart"
    assert candidates[0].eligible
    assert not candidates[1].eligible
    assert candidates[2].eligible


def test_champion_retains_simpler_rrf_on_exact_ties() -> None:
    stages: tuple[EvaluationStage, ...] = ("rrf", "pointwise", "lambdamart")
    means: dict[tuple[EvaluationStage, int], float] = {
        (stage, cutoff): 0.5 for stage in stages for cutoff in (10, 20)
    }

    selected, candidates = _champion_candidates(means, RankingEvaluationConfig())

    assert selected == "rrf"
    assert tuple(candidate.eligible for candidate in candidates) == (True, False, False)
