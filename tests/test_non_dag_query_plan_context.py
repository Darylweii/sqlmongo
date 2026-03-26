import importlib
import sys
from types import SimpleNamespace


def _make_query_plan(query_mode: str, *, data_type: str = "ep", requested_tags=None) -> dict:
    raw_plan = {}
    if requested_tags:
        raw_plan["requested_tags"] = list(requested_tags)
    return {
        "source": "llm",
        "current_question": "a1_b9 query",
        "query_mode": query_mode,
        "inferred_data_type": data_type,
        "explicit_device_codes": ["a1_b9"],
        "search_targets": ["a1_b9"],
        "time_start": "2024-01-01",
        "time_end": "2024-01-31",
        "has_sensor_intent": query_mode == "sensor_query",
        "has_detect_data_types_intent": query_mode == "detect_data_types",
        "response_style": "structured_analysis",
        "confidence": 0.96,
        "raw_plan": raw_plan,
    }


class _DummyFetcher:
    def fetch_sync(self, **_kwargs):
        return SimpleNamespace(
            data=[{"logTime": "2024-01-01 00:00:00", "device": "a1_b9", "tag": "ua", "val": 228}],
            total_count=1,
            is_sampled=False,
            statistics={"count": 1},
            query_info={"query": {"device": {"$in": ["a1_b9"]}}},
            page=1,
            page_size=50,
            total_pages=1,
            has_more=False,
            failed_collections=[],
        )


class _DummyCompressor:
    def compress(self, data, _output_format):
        return data


def test_sensor_tool_enriches_query_info_with_query_plan_context(monkeypatch) -> None:
    orchestrator_module = importlib.import_module("src.agent.orchestrator")
    sensor_tool = sys.modules[orchestrator_module.fetch_sensor_data_with_components.__module__]

    monkeypatch.setattr(sensor_tool, "get_collection_prefix", lambda data_type: f"source_data_{data_type}")
    monkeypatch.setattr(sensor_tool, "get_target_collections", lambda start, end, prefix=None: [f"{prefix}_{start}_{end}"])
    monkeypatch.setattr(sensor_tool, "get_data_tags", lambda _prefix: ["ua", "ub", "uc"])
    monkeypatch.setattr(sensor_tool.InsightEngine, "build", staticmethod(lambda *args, **kwargs: ({"mode": "single"}, [])))

    result = sensor_tool.fetch_sensor_data_with_components(
        device_codes=["a1_b9"],
        start_time="2024-01-01",
        end_time="2024-01-31",
        data_fetcher=_DummyFetcher(),
        compressor=_DummyCompressor(),
        data_type="u_line",
        user_query="query a1_b9 voltage data for 2024-01",
        query_plan=_make_query_plan("sensor_query", data_type="u_line", requested_tags=["ua"]),
    )

    context = result["query_info"]["query_plan_context"]
    assert context["query_mode"] == "sensor_query"
    assert context["data_type"] == "u_line"
    assert context["requested_tags"] == ["ua"]
    assert context["time_start"] == "2024-01-01"
    assert context["time_end"] == "2024-01-31"
    assert context["targets"] == ["a1_b9"]
    assert context["is_comparison"] is False



def test_orchestrator_execute_action_injects_query_plan_context(monkeypatch) -> None:
    orchestrator_module = importlib.import_module("src.agent.orchestrator")
    agent = orchestrator_module.LLMAgent(llm=None)

    monkeypatch.setattr(
        agent,
        "_action_detect_data_types",
        lambda params: {
            "success": True,
            "device_codes": params.get("device_codes", []),
            "available_types": {"ep": 12, "u_line": 36},
            "summary": "ep(12), u_line(36)",
        },
    )

    result = agent._execute_action(
        "detect_data_types",
        {
            "device_codes": ["a1_b9"],
            "tg_values": ["TG232"],
            "user_query": "what data types does a1_b9 have",
            "query_plan": _make_query_plan("detect_data_types", data_type="ep"),
        },
    )

    context = result["query_info"]["query_plan_context"]
    assert context["query_mode"] == "detect_data_types"
    assert context["data_type"] == "ep"
    assert context["targets"] == ["a1_b9"]
    assert context["time_start"] == "2024-01-01"
    assert context["time_end"] == "2024-01-31"
