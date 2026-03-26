from src.agent.orchestrator import LLMAgent
from src.agent.query_plan import QueryPlan


def _make_plan(**overrides) -> QueryPlan:
    payload = {
        "current_question": "compare a1_b9 and b1_b14 power usage",
        "source": "llm",
        "query_mode": "comparison",
        "inferred_data_type": "ep",
        "explicit_device_codes": ("a1_b9", "b1_b14"),
        "search_targets": ("a1_b9", "b1_b14"),
        "project_hints": ("north campus",),
        "time_start": "2024-01-01",
        "time_end": "2024-01-31",
        "has_sensor_intent": True,
        "has_detect_data_types_intent": False,
        "has_project_listing_intent": False,
        "has_project_stats_intent": False,
        "has_device_listing_intent": False,
        "has_comparison_intent": True,
        "has_pagination_intent": False,
        "has_time_reference": True,
        "has_ranked_point_intent": False,
        "ranking_order": None,
        "ranking_limit": 5,
        "ranking_granularity": "day",
        "has_trend_decision_intent": False,
        "aggregation": "bucket",
        "response_style": "direct_answer",
        "period_compare_targets": (),
        "confidence": 0.95,
        "raw_plan": {},
    }
    payload.update(overrides)
    return QueryPlan(**payload)



def test_build_action_intent_hints_uses_shared_compat_state(monkeypatch) -> None:
    agent = LLMAgent(llm=None)
    plan = _make_plan(query_mode="ranked_buckets", has_comparison_intent=False, explicit_device_codes=("a1_b9",), search_targets=("a1_b9",))

    monkeypatch.setattr(agent, "_get_query_plan", lambda _query: plan)

    hints = agent._build_action_intent_hints(
        "get_sensor_data",
        {
            "device_codes": ["a1_b9"],
            "data_type": "ep",
            "start_time": "2024-01-01",
            "end_time": "2024-01-31",
            "user_query": "a1_b9 top 5 days in 2024-01",
            "query_plan": plan.to_dict(),
        },
    )

    assert hints["target"] == "a1_b9"
    assert hints["data_type"] == "ep"
    assert hints["time_start"] == "2024-01-01"
    assert hints["time_end"] == "2024-01-31"
    assert hints["query_mode"] == "ranked_buckets"
    assert hints["ranking_limit"] == 5
    assert hints["ranking_granularity"] == "day"
    assert hints["aggregation"] == "bucket"
    assert hints["response_style"] == "direct_answer"



def test_extract_search_targets_uses_shared_query_state(monkeypatch) -> None:
    agent = LLMAgent(llm=None)
    plan = _make_plan()

    monkeypatch.setattr(agent, "_get_query_plan", lambda _query: plan)

    targets = agent._extract_search_targets([], "compare a1_b9 and b1_b14 power usage", comparison_mode=True)

    assert targets == ["a1_b9", "b1_b14"]
