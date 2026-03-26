from src.agent.action_override_policy import (
    ActionOverrideContext,
    apply_action_override_policy,
    decide_metadata_override,
    decide_sensor_override,
)
from src.agent.orchestrator import LLMAgent
from src.agent.query_plan import QueryPlan


def _make_plan(**overrides) -> QueryPlan:
    payload = {
        "current_question": "query",
        "source": "llm",
        "query_mode": "general",
        "inferred_data_type": "ep",
        "explicit_device_codes": (),
        "search_targets": (),
        "project_hints": (),
        "time_start": None,
        "time_end": None,
        "has_sensor_intent": False,
        "has_detect_data_types_intent": False,
        "has_project_listing_intent": False,
        "has_project_stats_intent": False,
        "has_device_listing_intent": False,
        "has_comparison_intent": False,
        "has_pagination_intent": False,
        "has_time_reference": False,
        "has_ranked_point_intent": False,
        "ranking_order": None,
        "ranking_limit": None,
        "ranking_granularity": None,
        "has_trend_decision_intent": False,
        "aggregation": None,
        "response_style": "direct_answer",
        "period_compare_targets": (),
        "confidence": 0.9,
        "raw_plan": {},
    }
    payload.update(overrides)
    return QueryPlan(**payload)



def test_policy_decides_project_list_override_from_shared_state() -> None:
    plan = _make_plan(query_mode="project_listing", has_project_listing_intent=True)
    context = ActionOverrideContext(
        query_state={"query_plan": plan.to_dict()},
        action="final_answer",
        action_input={},
        history_actions=(),
    )

    decision = decide_metadata_override(context)

    assert decision is not None
    assert decision.action == "list_projects"
    assert decision.reason == "override_from_final_answer_project_list"



def test_policy_decides_sensor_override_from_shared_state() -> None:
    plan = _make_plan(
        query_mode="comparison",
        has_sensor_intent=True,
        has_comparison_intent=True,
        explicit_device_codes=("a1_b9", "b1_b14"),
        search_targets=("a1_b9", "b1_b14"),
        inferred_data_type="u_line",
    )
    context = ActionOverrideContext(
        query_state={"query_plan": plan.to_dict()},
        action="final_answer",
        action_input={"user_query": "compare devices"},
        history_actions=(),
        preferred_device_codes=("a1_b9", "b1_b14"),
        preferred_tg_values=("TG232", "TG314"),
        preferred_source="session_alias",
        resolved_time_range={"start_time": "2024-01-01", "end_time": "2024-01-31"},
    )

    decision = decide_sensor_override(context)

    assert decision is not None
    assert decision.action == "get_sensor_data"
    assert decision.action_input["device_codes"] == ["a1_b9", "b1_b14"]
    assert decision.action_input["tg_values"] == ["TG232", "TG314"]
    assert decision.action_input["data_type"] == "u_line"
    assert decision.action_input["page_size"] == 0
    assert decision.reason == "override_from_final_answer_session_alias"



def test_orchestrator_applies_unified_override_policy(monkeypatch) -> None:
    agent = LLMAgent(llm=None)
    plan = _make_plan(
        current_question="a1_b9 today voltage",
        query_mode="sensor_query",
        has_sensor_intent=True,
        explicit_device_codes=("a1_b9",),
        search_targets=("a1_b9",),
        inferred_data_type="u_line",
    )

    monkeypatch.setattr(agent, "_get_query_plan", lambda _query: plan)
    monkeypatch.setattr(agent, "_get_cached_device_codes_from_query", lambda _query: [])
    monkeypatch.setattr(agent, "_resolve_preferred_device_scope", lambda _query, _plan: (["a1_b9"], [{"device": "a1_b9", "tg": "TG232"}], "explicit_resolved"))
    monkeypatch.setattr(agent, "_resolve_time_range_from_query", lambda _query, now=None: {"start_time": "2026-03-19", "end_time": "2026-03-19"})

    action, action_input, reason = agent._apply_action_override_policy(
        "final_answer",
        {"user_query": "query a1_b9 today voltage", "query_plan": plan.to_dict()},
        {"user_query": "query a1_b9 today voltage", "history": []},
    )

    assert action == "get_sensor_data"
    assert action_input["device_codes"] == ["a1_b9"]
    assert action_input["tg_values"] == ["TG232"]
    assert action_input["data_type"] == "u_line"
    assert action_input["start_time"] == "2026-03-19"
    assert action_input["end_time"] == "2026-03-19"
    assert reason == "override_from_final_answer_explicit_resolved"



def test_policy_apply_returns_original_action_without_override() -> None:
    plan = _make_plan(query_mode="general")
    decision = apply_action_override_policy(
        ActionOverrideContext(
            query_state={"query_plan": plan.to_dict()},
            action="direct_answer",
            action_input={"message": "ok"},
            history_actions=(),
        )
    )

    assert decision.action == "direct_answer"
    assert decision.action_input == {"message": "ok"}
    assert decision.reason is None
