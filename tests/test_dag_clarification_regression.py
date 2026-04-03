from dataclasses import dataclass

from src.agent.dag_orchestrator import DAGOrchestrator
from src.agent.types import (
    NODE_ACTION_OVERRIDE_POLICY,
    NODE_INTENT_PARSER,
    NODE_METADATA_MAPPER,
    NODE_PARALLEL_FETCHER,
    NODE_SHARDING_ROUTER,
    NODE_SYNTHESIZER,
)


@dataclass
class _Device:
    device: str
    name: str
    project_id: str
    project_name: str
    project_code_name: str
    tg: str | None = None
    device_type: str = "meter"

    def to_dict(self):
        return {
            "device": self.device,
            "name": self.name,
            "project_id": self.project_id,
            "project_name": self.project_name,
            "project_code_name": self.project_code_name,
            "tg": self.tg,
            "device_type": self.device_type,
        }


class _FakeMetadataEngine:
    def search_devices(self, keyword):
        assert keyword == "a1_b9"
        return [
            _Device("a1_b9", "B2柜", "1", "中国能建集团数据机房监控项目", "ceec"),
            _Device("a1_b9", "601-612", "2", "平陆运河项目", "plyh"),
        ], False

    def list_projects(self):
        return []

    def list_all_devices(self):
        return []


class _FakeDataFetcher:
    def fetch_sync(self, **kwargs):
        raise AssertionError("clarification path should stop before database fetch")


class _FakeLLM:
    pass


def test_dag_clarification_stops_before_db_fetch() -> None:
    orchestrator = DAGOrchestrator(
        llm=_FakeLLM(),
        metadata_engine=_FakeMetadataEngine(),
        data_fetcher=_FakeDataFetcher(),
    )

    query = "\u67e5\u8be2 a1_b9 \u8bbe\u5907\u4eca\u5929\u7684\u7535\u538b\u6570\u636e"
    events = list(orchestrator.run_with_progress(query))

    assert [event["node_name"] for event in events if event["type"] == "step_done"] == [
        "intent_parser",
        "metadata_mapper",
    ]

    final_event = events[-1]
    assert final_event["type"] == "final_answer"
    assert final_event["clarification_required"] is True
    assert final_event["clarification_candidates"][0]["keyword"] == "a1_b9"
    assert len(final_event["clarification_candidates"][0]["candidates"]) >= 2


def test_dag_sensor_final_event_includes_chart_payload_for_chart_follow_up() -> None:
    orchestrator = DAGOrchestrator(
        llm=_FakeLLM(),
        metadata_engine=_FakeMetadataEngine(),
        data_fetcher=_FakeDataFetcher(),
    )

    def _intent_parser(state):
        history = list(state.get("history", []))
        return {
            **state,
            "query_plan": {
                "current_question": "查询 a1_b9 设备今天的电流数据 画趋势图",
                "query_mode": "sensor_query",
                "inferred_data_type": "i",
                "time_start": "2026-04-02",
                "time_end": "2026-04-02",
                "has_sensor_intent": True,
                "has_detect_data_types_intent": False,
                "has_project_listing_intent": False,
                "has_project_stats_intent": False,
                "has_device_listing_intent": False,
            },
            "history": history + [{"node": NODE_INTENT_PARSER, "result": "QueryPlan=sensor_query"}],
        }

    def _action_override_policy(state):
        history = list(state.get("history", []))
        return {
            **state,
            "history": history + [{"node": NODE_ACTION_OVERRIDE_POLICY, "result": "无需 override"}],
        }

    def _metadata_mapper(state):
        history = list(state.get("history", []))
        return {
            **state,
            "device_codes": ["a1_b9"],
            "device_names": {"a1_b9": "1#变加联络 AA5-1 稻盛和夫"},
            "tg_values": ["TG232"],
            "history": history + [{"node": NODE_METADATA_MAPPER, "result": "解析出 1 个设备"}],
        }

    def _sharding_router(state):
        history = list(state.get("history", []))
        return {
            **state,
            "collections": ["source_data_i_202604"],
            "data_tags": ["ia", "ib", "ic"],
            "history": history + [{"node": NODE_SHARDING_ROUTER, "result": "计算出 1 个数据分片"}],
        }

    def _parallel_fetcher(state):
        history = list(state.get("history", []))
        return {
            **state,
            "raw_data": [
                {"logTime": "2026-04-02 00:00:00", "device": "a1_b9", "tag": "ia", "val": 3.2},
                {"logTime": "2026-04-02 00:00:00", "device": "a1_b9", "tag": "ib", "val": 4.1},
                {"logTime": "2026-04-02 00:00:00", "device": "a1_b9", "tag": "ic", "val": 5.0},
                {"logTime": "2026-04-02 01:00:00", "device": "a1_b9", "tag": "ia", "val": 6.8},
                {"logTime": "2026-04-02 01:00:00", "device": "a1_b9", "tag": "ib", "val": 7.2},
                {"logTime": "2026-04-02 01:00:00", "device": "a1_b9", "tag": "ic", "val": 8.5},
            ],
            "total_count": 6,
            "statistics": {"avg": 5.8, "max": 8.5, "min": 3.2, "count": 6, "trend": "上升", "change_rate": 75.0, "cv": 28.4, "anomaly_count": 0, "anomaly_ratio": 0.0},
            "query_info": {"collections": ["source_data_i_202604"]},
            "history": history + [{"node": NODE_PARALLEL_FETCHER, "result": "获取 6 条数据记录"}],
        }

    def _synthesizer(state):
        history = list(state.get("history", []))
        return {
            **state,
            "final_response": "已根据当前查询生成趋势分析。",
            "show_table": True,
            "table_type": "sensor_data",
            "history": history + [{"node": NODE_SYNTHESIZER, "result": "生成响应完成: 16 字, 显示表格=True"}],
        }

    orchestrator._intent_parser = _intent_parser
    orchestrator._action_override_policy = _action_override_policy
    orchestrator._metadata_mapper = _metadata_mapper
    orchestrator._sharding_router = _sharding_router
    orchestrator._parallel_fetcher = _parallel_fetcher
    orchestrator._synthesizer = _synthesizer

    events = list(orchestrator.run_with_progress("查询 a1_b9 设备今天的电流数据 画趋势图"))
    final_event = next(event for event in events if event.get("type") == "final_answer")

    assert final_event["show_table"] is True
    assert final_event["table_type"] == "sensor_data"
    assert final_event["show_charts"] is True
    assert final_event["analysis"] is not None
    assert final_event["analysis"]["mode"] == "single"
    assert len(final_event["chart_specs"]) >= 1
    assert final_event["chart_specs"][0]["id"] == "trend-line"
    assert final_event["table_preview"] is not None
    assert final_event["table_preview"]["show_charts"] is True
    assert len(final_event["table_preview"]["chart_specs"]) >= 1
