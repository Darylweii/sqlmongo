import json

from src.agent.orchestrator import LLMAgent
from src.agent.query_planner import LLMQueryPlanner
from src.analysis.insight_engine import InsightEngine
from src.tools.sensor_tool import _build_focused_result


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        return type("Resp", (), {"content": json.dumps(self.payload, ensure_ascii=False)})()


def test_query_planner_prefers_local_plan_for_daily_top1_query() -> None:
    llm = FakeLLM(
        {
            "query_mode": "ranked_buckets",
            "inferred_data_type": "ep",
            "explicit_device_codes": ["a1_b9"],
            "search_targets": ["a1_b9"],
            "has_sensor_intent": True,
            "has_time_reference": True,
            "has_ranked_point_intent": True,
            "ranking_order": "desc",
            "ranking_limit": 1,
            "ranking_granularity": "day",
            "aggregation": "bucket",
            "response_style": "direct_answer",
            "confidence": 0.96,
        }
    )
    planner = LLMQueryPlanner(llm)

    plan = planner.plan("a1_b9 \u8bbe\u59072024\u5e741\u6708\u54ea\u5929\u7528\u7535\u91cf\u6700\u9ad8")

    assert llm.calls == 0
    assert plan.source == "fallback"
    assert plan.query_mode == "ranked_buckets"
    assert plan.inferred_data_type == "ep"
    assert plan.explicit_device_codes == ("a1_b9",)
    assert plan.ranking_order == "desc"
    assert plan.ranking_limit == 1
    assert plan.ranking_granularity == "day"
    assert plan.response_style == "direct_answer"


def test_query_planner_uses_llm_when_local_parse_missing_core_slots() -> None:
    llm = FakeLLM(
        {
            "query_mode": "ranked_buckets",
            "inferred_data_type": "ep",
            "search_targets": ["\u672a\u77e5\u8bbe\u5907"],
            "has_sensor_intent": True,
            "has_time_reference": True,
            "has_ranked_point_intent": True,
            "ranking_order": "asc",
            "ranking_limit": 1,
            "ranking_granularity": "day",
            "aggregation": "bucket",
            "response_style": "direct_answer",
            "confidence": 0.92,
        }
    )
    planner = LLMQueryPlanner(llm)

    plan = planner.plan("\u8bf7\u5e2e\u6211\u627e\u51fa\u672a\u77e5\u8bbe\u5907")

    assert llm.calls == 1
    assert plan.source == "llm"
    assert plan.query_mode == "ranked_buckets"
    assert plan.ranking_order == "asc"
    assert plan.ranking_granularity == "day"


def test_orchestrator_heuristic_action_uses_local_first_query_plan() -> None:
    llm = FakeLLM(
        {
            "query_mode": "ranked_buckets",
            "inferred_data_type": "ep",
            "explicit_device_codes": ["a1_b9"],
            "search_targets": ["a1_b9"],
            "has_sensor_intent": True,
            "has_ranked_point_intent": True,
            "ranking_order": "desc",
            "ranking_limit": 1,
            "ranking_granularity": "day",
            "response_style": "direct_answer",
            "confidence": 0.95,
        }
    )
    agent = LLMAgent(llm=llm)
    state = agent._init_state("a1_b9 \u8bbe\u59072024\u5e741\u6708\u54ea\u5929\u7528\u7535\u91cf\u6700\u9ad8")

    action = agent._build_heuristic_action(state, fast_path=True)

    assert llm.calls == 0
    assert action is not None
    assert action["action"] == "get_sensor_data"
    assert action["action_input"]["device_codes"] == ["a1_b9"]
    assert action["action_input"]["data_type"] == "ep"
    assert action["action_input"]["query_plan"]["query_mode"] == "ranked_buckets"
    assert action["action_input"]["query_plan"]["ranking_granularity"] == "day"


def test_focused_result_uses_query_plan_hints_for_bucket_ranking() -> None:
    records = [
        {"logTime": "2024-01-01 00:00:00", "val": 100, "device": "a1_b9", "tag": "ep"},
        {"logTime": "2024-01-01 23:00:00", "val": 150, "device": "a1_b9", "tag": "ep"},
        {"logTime": "2024-01-02 00:00:00", "val": 150, "device": "a1_b9", "tag": "ep"},
        {"logTime": "2024-01-02 23:00:00", "val": 240, "device": "a1_b9", "tag": "ep"},
    ]
    analysis = {"mode": "single", "metric": "\u7528\u7535\u91cf", "unit": "kWh"}
    result = _build_focused_result(
        records,
        analysis,
        "a1_b9 \u8bbe\u59072024\u5e741\u6708\u54ea\u5929\u7528\u7535\u91cf\u6700\u9ad8",
        "ep",
        query_plan={
            "query_mode": "ranked_buckets",
            "has_ranked_point_intent": True,
            "ranking_order": "desc",
            "ranking_limit": 1,
            "ranking_granularity": "day",
            "response_style": "direct_answer",
        },
    )

    assert result is not None
    assert result["mode"] == "ranked_buckets"
    assert result["rows"][0]["time"] == "2024-01-02"
    assert result["rows"][0]["value"] == 90.0


def test_focused_result_supports_anomaly_points() -> None:
    records = [
        {"logTime": "2024-01-01 00:00:00", "val": 229, "device": "a1_b9", "tag": "ua"},
        {"logTime": "2024-01-01 01:00:00", "val": 230, "device": "a1_b9", "tag": "ua"},
        {"logTime": "2024-01-01 02:00:00", "val": 231, "device": "a1_b9", "tag": "ua"},
        {"logTime": "2024-01-01 03:00:00", "val": 260, "device": "a1_b9", "tag": "ua"},
        {"logTime": "2024-01-01 04:00:00", "val": 230, "device": "a1_b9", "tag": "ua"},
        {"logTime": "2024-01-01 05:00:00", "val": 229, "device": "a1_b9", "tag": "ua"},
    ]
    analysis = {"mode": "single", "metric": "\u7535\u538b", "unit": "V"}
    result = _build_focused_result(
        records,
        analysis,
        "a1_b9 \u6709\u6ca1\u6709\u5f02\u5e38\u7535\u538b\u65f6\u95f4\u70b9",
        "u_line",
        query_plan={
            "query_mode": "anomaly_points",
            "has_anomaly_point_intent": True,
            "response_style": "direct_answer",
            "ranking_limit": 3,
        },
    )

    assert result is not None
    assert result["mode"] == "anomaly_points"
    assert result["anomaly_count"] >= 1
    assert result["rows"]
    assert result["rows"][0]["time"] == "2024-01-01 03:00"


def test_query_planner_prefers_local_plan_for_weekly_bucket_query() -> None:
    llm = FakeLLM({"query_mode": "general", "confidence": 0.1})
    planner = LLMQueryPlanner(llm)

    plan = planner.plan("a1_b9 \u8bbe\u59072024\u5e741\u6708\u6bcf\u5468\u7684\u603b\u7528\u7535\u91cf")

    assert llm.calls == 0
    assert plan.source == "fallback"
    assert plan.query_mode == "sensor_query"
    assert plan.inferred_data_type == "ep"
    assert plan.explicit_device_codes == ("a1_b9",)
    assert plan.ranking_granularity == "week"
    assert plan.aggregation == "bucket"


def test_focused_result_supports_weekly_bucket_summary() -> None:
    records = [
        {"logTime": "2024-01-01 00:00:00", "val": 100, "device": "a1_b9", "tag": "ep"},
        {"logTime": "2024-01-07 23:00:00", "val": 160, "device": "a1_b9", "tag": "ep"},
        {"logTime": "2024-01-08 00:00:00", "val": 160, "device": "a1_b9", "tag": "ep"},
        {"logTime": "2024-01-14 23:00:00", "val": 250, "device": "a1_b9", "tag": "ep"},
    ]
    analysis = {"mode": "single", "metric": "\u7528\u7535\u91cf", "unit": "kWh"}

    result = _build_focused_result(
        records,
        analysis,
        "\u7edf\u8ba1 a1_b9 \u8bbe\u59072024\u5e741\u6708\u6bcf\u5468\u7684\u603b\u7528\u7535\u91cf",
        "ep",
        query_plan={
            "query_mode": "sensor_query",
            "aggregation": "bucket",
            "ranking_granularity": "week",
            "response_style": "direct_answer",
        },
    )

    assert result is not None
    assert result["mode"] == "bucket_summary"
    assert len(result["rows"]) == 2
    assert result["rows"][0]["value"] == 60.0
    assert result["rows"][1]["value"] == 90.0



def test_insight_engine_marks_single_phase_analysis() -> None:
    analysis, _ = InsightEngine.build(
        records=[
            {"logTime": "2024-01-01 00:00:00", "val": 229, "device": "a2_b1", "tag": "ua"},
            {"logTime": "2024-01-01 01:00:00", "val": 230, "device": "a2_b1", "tag": "ua"},
        ],
        statistics=None,
        data_type="ua",
        device_codes=["a2_b1"],
    )

    assert analysis is not None
    assert analysis["analysis_scope_mode"] == "single_phase"
    assert analysis["analysis_scope_label"] == "\u6309\u5355\u76f8\u5206\u6790"
    assert "\u6309\u5355\u76f8\u5206\u6790" in analysis["headline"]



def test_insight_engine_marks_three_phase_joint_analysis() -> None:
    analysis, _ = InsightEngine.build(
        records=[
            {"logTime": "2024-01-01 00:00:00", "val": 229, "device": "a2_b1", "tag": "ua"},
            {"logTime": "2024-01-01 00:00:00", "val": 230, "device": "a2_b1", "tag": "ub"},
            {"logTime": "2024-01-01 00:00:00", "val": 231, "device": "a2_b1", "tag": "uc"},
        ],
        statistics=None,
        data_type="u_line",
        device_codes=["a2_b1"],
    )

    assert analysis is not None
    assert analysis["analysis_scope_mode"] == "three_phase_joint"
    assert analysis["analysis_scope_label"] == "\u6309\u4e09\u76f8\u8054\u5408\u5206\u6790"
    assert "\u6309\u4e09\u76f8\u8054\u5408\u5206\u6790" in analysis["headline"]



def test_focused_result_inherits_analysis_scope_label() -> None:
    records = [
        {"logTime": "2024-01-01 00:00:00", "val": 229, "device": "a2_b1", "tag": "ua"},
        {"logTime": "2024-01-01 01:00:00", "val": 231, "device": "a2_b1", "tag": "ub"},
        {"logTime": "2024-01-01 02:00:00", "val": 230, "device": "a2_b1", "tag": "uc"},
    ]
    analysis = {
        "mode": "single",
        "metric": "??",
        "unit": "V",
        "analysis_scope_label": "\u6309\u4e09\u76f8\u8054\u5408\u5206\u6790",
    }

    result = _build_focused_result(
        records,
        analysis,
        "a2_b1 ??????????",
        "u_line",
        query_plan={
            "query_mode": "ranked_timepoints",
            "has_ranked_point_intent": True,
            "ranking_order": "desc",
            "ranking_limit": 1,
            "response_style": "direct_answer",
        },
    )

    assert result is not None
    assert result["analysis_scope_label"] == "\u6309\u4e09\u76f8\u8054\u5408\u5206\u6790"
    assert result["headline"].startswith("\u6309\u4e09\u76f8\u8054\u5408\u5206\u6790：")
