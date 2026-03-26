import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _ensure_package(name: str):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    return module


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_agent_modules():
    prefixes = (
        "src.agent",
        "src.metadata",
        "src.router",
        "src.semantic_layer",
        "src.fetcher",
        "src.cache",
        "src.compressor",
        "src.tools",
        "langchain_core",
        "langgraph",
    )
    for name in [key for key in list(sys.modules) if key.startswith(prefixes)]:
        sys.modules.pop(name, None)

    for package_name in [
        "src",
        "src.agent",
        "src.agent.nodes",
        "src.agent.utils",
        "src.metadata",
        "src.router",
        "src.semantic_layer",
        "src.fetcher",
        "src.cache",
        "src.compressor",
        "src.tools",
        "langchain_core",
        "langchain_core.language_models",
        "langchain_core.messages",
        "langgraph",
        "langgraph.graph",
    ]:
        _ensure_package(package_name)

    language_models = sys.modules["langchain_core.language_models"]
    messages = sys.modules["langchain_core.messages"]
    graph_mod = sys.modules["langgraph.graph"]

    class BaseChatModel:
        pass

    class SystemMessage:
        def __init__(self, content):
            self.content = content

    class HumanMessage:
        def __init__(self, content):
            self.content = content

    class StateGraph:
        def __init__(self, _state_type):
            pass

        def add_node(self, *_args, **_kwargs):
            pass

        def set_entry_point(self, *_args, **_kwargs):
            pass

        def add_edge(self, *_args, **_kwargs):
            pass

        def add_conditional_edges(self, *_args, **_kwargs):
            pass

        def compile(self):
            class _Compiled:
                def stream(self, _initial_state):
                    return []

                def invoke(self, initial_state):
                    return initial_state

            return _Compiled()

    language_models.BaseChatModel = BaseChatModel
    messages.SystemMessage = SystemMessage
    messages.HumanMessage = HumanMessage
    graph_mod.StateGraph = StateGraph
    graph_mod.START = "__start__"
    graph_mod.END = "__end__"

    exceptions_module = types.ModuleType("src.exceptions")
    exceptions_module.DatabaseConnectionError = type("DatabaseConnectionError", (Exception,), {})
    exceptions_module.MetadataEngineError = type("MetadataEngineError", (Exception,), {})
    exceptions_module.CircuitBreakerError = type("CircuitBreakerError", (Exception,), {})
    exceptions_module.InvalidDateRangeError = type("InvalidDateRangeError", (Exception,), {})
    exceptions_module.DataFetcherError = type("DataFetcherError", (Exception,), {})
    sys.modules["src.exceptions"] = exceptions_module

    metadata_module = types.ModuleType("src.metadata.metadata_engine")

    class DeviceInfo:
        def __init__(self, device, name, device_type="", project_id="", project_name=None, project_code_name=None):
            self.device = device
            self.name = name
            self.device_type = device_type
            self.project_id = project_id
            self.project_name = project_name
            self.project_code_name = project_code_name

    metadata_module.DeviceInfo = DeviceInfo
    metadata_module.MetadataEngine = type("MetadataEngine", (), {})
    sys.modules["src.metadata.metadata_engine"] = metadata_module

    router_module = types.ModuleType("src.router.collection_router")
    router_module.get_collection_prefix = lambda data_type: f"prefix_{data_type}"
    router_module.get_data_tags = lambda prefix: [f"tag_for_{prefix}"]
    router_module.get_target_collections = lambda start_date, end_date, collection_prefix, max_collections: [
        f"{collection_prefix}_{start_date}_{end_date}_{max_collections}"
    ]
    sys.modules["src.router.collection_router"] = router_module

    device_search_module = types.ModuleType("src.semantic_layer.device_search")

    class DeviceSemanticSearch:
        def __init__(self, results=None):
            self.is_initialized = True
            self._results = results or []

        def initialize(self):
            return True

        def search_devices(self, query, top_k, min_score):
            return list(self._results)

    device_search_module.DeviceSemanticSearch = DeviceSemanticSearch
    sys.modules["src.semantic_layer.device_search"] = device_search_module

    smart_filter_module = types.ModuleType("src.agent.utils.smart_device_filter")

    class SmartDeviceFilter:
        @staticmethod
        def filter_comparison_devices(devices_with_metadata):
            return devices_with_metadata, {"strategy": "???", "details": ""}

    smart_filter_module.SmartDeviceFilter = SmartDeviceFilter
    sys.modules["src.agent.utils.smart_device_filter"] = smart_filter_module

    for module_name, attrs in {
        "src.fetcher.data_fetcher": {"DataFetcher": type("DataFetcher", (), {}), "SensorDataResult": type("SensorDataResult", (), {})},
        "src.cache.cache_manager": {"CacheManager": type("CacheManager", (), {})},
        "src.compressor.context_compressor": {"ContextCompressor": type("ContextCompressor", (), {})},
        "src.semantic_layer.config": {"SemanticLayerConfig": type("SemanticLayerConfig", (), {})},
        "src.semantic_layer.semantic_layer": {"SemanticLayer": type("SemanticLayer", (), {"is_initialized": False}), "create_semantic_layer": lambda **_kwargs: None},
    }.items():
        module = types.ModuleType(module_name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[module_name] = module


    sensor_tool_module = types.ModuleType("src.tools.sensor_tool")

    def _build_focused_result(raw_data, analysis, user_query, data_type, query_plan=None):
        query_plan = query_plan or {}
        mode = query_plan.get("query_mode") or "ranked_timepoints"
        metric = (analysis or {}).get("metric") or data_type or "数值"
        unit = (analysis or {}).get("unit") or ""
        if mode == "anomaly_points":
            return {
                "mode": "anomaly_points",
                "metric": metric,
                "unit": unit,
                "headline": "已识别到异常时间点",
                "basis_label": "按原始时序",
                "anomaly_count": 1,
                "sample_count": len(raw_data or []),
                "anomaly_ratio_pct": 25.0,
                "rows": [
                    {
                        "time": "2024-01-01 02:00:00",
                        "device": "a1_b9",
                        "tag": data_type,
                        "value": 160,
                        "severity": 58,
                    }
                ],
            }
        return {
            "mode": "ranked_timepoints",
            "metric": metric,
            "unit": unit,
            "rows": [
                {
                    "time": "2024-01-01 02:00:00",
                    "device": "a1_b9",
                    "tag": data_type,
                    "value": 160,
                }
            ],
            "order": "desc",
        }

    sensor_tool_module._build_focused_result = _build_focused_result
    sensor_tool_module.fetch_sensor_data_with_components = lambda **_kwargs: {
        "success": True,
        "data": [],
        "total_count": 0,
        "page": 1,
        "page_size": 50,
        "total_pages": 1,
        "has_more": False,
        "statistics": None,
        "analysis": None,
        "focused_table": None,
        "chart_specs": [],
        "show_charts": False,
    }
    sys.modules["src.tools.sensor_tool"] = sensor_tool_module

    orchestrator_module = types.ModuleType("src.agent.orchestrator")
    orchestrator_module.create_agent_with_streaming = lambda **_kwargs: None
    sys.modules["src.agent.orchestrator"] = orchestrator_module

    entity_resolver_module = types.ModuleType("src.entity_resolver")
    entity_resolver_module.ChromaEntityResolver = type("ChromaEntityResolver", (), {"__init__": lambda self, *a, **k: None})
    sys.modules["src.entity_resolver"] = entity_resolver_module

    _load_module("src.agent.query_entities", "src/agent/query_entities.py")
    _load_module("src.agent.query_time_range", "src/agent/query_time_range.py")
    _load_module("src.agent.query_plan", "src/agent/query_plan.py")
    _load_module("src.agent.query_plan_state", "src/agent/query_plan_state.py")
    _load_module("src.agent.action_override_policy", "src/agent/action_override_policy.py")
    _load_module("src.agent.query_planner", "src/agent/query_planner.py")
    _load_module("src.agent.types", "src/agent/types.py")
    _load_module("src.agent.focused_response", "src/agent/focused_response.py")
    metadata_mapper = _load_module("src.agent.nodes.metadata_mapper", "src/agent/nodes/metadata_mapper.py")
    semantic_metadata_mapper = _load_module("src.agent.nodes.semantic_metadata_mapper", "src/agent/nodes/semantic_metadata_mapper.py")
    sharding_router = _load_module("src.agent.nodes.sharding_router", "src/agent/nodes/sharding_router.py")
    parallel_fetcher = _load_module("src.agent.nodes.parallel_fetcher", "src/agent/nodes/parallel_fetcher.py")
    synthesizer = _load_module("src.agent.nodes.synthesizer", "src/agent/nodes/synthesizer.py")
    intent_parser = _load_module("src.agent.nodes.intent_parser", "src/agent/nodes/intent_parser.py")
    action_override_policy_node = _load_module("src.agent.nodes.action_override_policy_node", "src/agent/nodes/action_override_policy_node.py")
    dag_orchestrator = _load_module("src.agent.dag_orchestrator", "src/agent/dag_orchestrator.py")
    sys.modules["src.agent"].DAGOrchestrator = dag_orchestrator.DAGOrchestrator

    return {
        "DeviceInfo": DeviceInfo,
        "DeviceSemanticSearch": DeviceSemanticSearch,
        "MetadataMapperNode": metadata_mapper.MetadataMapperNode,
        "SemanticMetadataMapperNode": semantic_metadata_mapper.SemanticMetadataMapperNode,
        "ShardingRouterNode": sharding_router.ShardingRouterNode,
        "ParallelFetcherNode": parallel_fetcher.ParallelFetcherNode,
        "SynthesizerNode": synthesizer.SynthesizerNode,
        "IntentParserNode": intent_parser.IntentParserNode,
        "ActionOverridePolicyNode": action_override_policy_node.ActionOverridePolicyNode,
        "DAGOrchestrator": dag_orchestrator.DAGOrchestrator,
        "route_after_query_plan": dag_orchestrator.route_after_query_plan,
        "route_after_action_override": dag_orchestrator.route_after_action_override,
        "route_after_metadata": dag_orchestrator.route_after_metadata,
    }


class FakeLLM:
    def __init__(self, payload=None, should_raise=False):
        self.payload = payload or {}
        self.should_raise = should_raise

    def invoke(self, _messages):
        if self.should_raise:
            raise RuntimeError("llm down")
        content = self.payload if isinstance(self.payload, str) else json.dumps(self.payload, ensure_ascii=False)
        return type("Resp", (), {"content": content})()


class FakeMetadataEngine:
    def __init__(self, mapping, projects=None, project_stats=None, project_devices=None):
        self.mapping = mapping
        self.projects = list(projects or [])
        self.project_stats = list(project_stats or [])
        self.project_devices = dict(project_devices or {})
        self.calls = []

    def search_devices(self, target):
        self.calls.append(target)
        return list(self.mapping.get(target, [])), f"sql:{target}"

    def list_projects(self):
        return list(self.projects)

    def get_project_device_stats(self):
        return list(self.project_stats)

    def get_devices_by_project(self, project_id):
        return list(self.project_devices.get(str(project_id), []))


def test_intent_parser_projects_query_plan_into_intent() -> None:
    modules = _load_agent_modules()
    node = modules["IntentParserNode"](
        llm=FakeLLM({
            "query_mode": "sensor_query",
            "inferred_data_type": "u_line",
            "explicit_device_codes": ["a1_b9"],
            "search_targets": ["a1_b9"],
            "has_sensor_intent": True,
            "has_time_reference": True,
            "response_style": "structured_analysis",
            "confidence": 0.94,
        })
    )

    result = node({"user_query": "query a1_b9 2024-01-05 voltage data", "history": []})

    assert result["query_plan"]["query_mode"] == "sensor_query"
    assert result["query_plan"]["time_start"] == "2024-01-05"
    assert result["query_plan"]["time_end"] == "2024-01-05"
    assert result["intent"]["target"] == "a1_b9"
    assert result["intent"]["data_type"] == "u_line"
    assert result["intent"]["time_start"] == "2024-01-05"
    assert result["intent"]["time_end"] == "2024-01-05"


def test_metadata_mapper_prefers_query_plan_targets() -> None:
    modules = _load_agent_modules()
    device = modules["DeviceInfo"]("a1_b9", "Target One Device")
    node = modules["MetadataMapperNode"](FakeMetadataEngine({"target_one": [device]}))

    result = node({
        "query_plan": {
            "query_mode": "sensor_query",
            "search_targets": ["target_one"],
            "explicit_device_codes": [],
            "inferred_data_type": "ep",
            "has_sensor_intent": True,
            "confidence": 0.9,
        },
        "intent": {"target": "ignored_target"},
        "history": [],
    })

    assert result["device_codes"] == ["a1_b9"]
    assert result["device_names"] == {"a1_b9": "Target One Device"}


def test_metadata_mapper_returns_terminal_device_table_for_listing_intent() -> None:
    modules = _load_agent_modules()
    device = modules["DeviceInfo"]("a1_b9", "Target One Device")
    device.tg = "TG232"
    node = modules["MetadataMapperNode"](FakeMetadataEngine({"b9": [device]}))

    result = node(
        {
            "history": [],
            "query_plan": {
                "query_mode": "device_listing",
                "search_targets": ["b9"],
                "has_device_listing_intent": True,
            },
        }
    )

    assert result["show_table"] is True
    assert result["table_type"] == "devices"
    assert result["devices"][0]["device"] == "a1_b9"
    assert result["tg_values"] == ["TG232"]


def test_route_after_metadata_short_circuits_device_listing() -> None:
    modules = _load_agent_modules()
    next_step = modules["route_after_metadata"](
        {
            "query_plan": {
                "query_mode": "device_listing",
                "has_device_listing_intent": True,
            },
            "show_table": True,
            "table_type": "devices",
        }
    )

    assert next_step == "terminal"


def test_route_after_metadata_short_circuits_clarification() -> None:
    modules = _load_agent_modules()
    next_step = modules["route_after_metadata"](
        {
            "clarification_required": True,
            "clarification_candidates": [{"keyword": "a1_b9", "candidates": []}],
        }
    )

    assert next_step == "terminal"


def test_metadata_mapper_builds_comparison_groups_from_query_plan() -> None:
    modules = _load_agent_modules()
    DeviceInfo = modules["DeviceInfo"]
    engine = FakeMetadataEngine({
        "target_one": [DeviceInfo("a1_b9", "Target One Device")],
        "target_two": [DeviceInfo("b1_b14", "Target Two Device")],
    })
    node = modules["MetadataMapperNode"](engine)

    result = node({
        "query_plan": {
            "query_mode": "comparison",
            "search_targets": ["target_one", "target_two"],
            "explicit_device_codes": [],
            "inferred_data_type": "ep",
            "has_sensor_intent": True,
            "has_comparison_intent": True,
            "confidence": 0.95,
        },
        "history": [],
    })

    assert result["is_comparison"] is True
    assert result["comparison_targets"] == ["target_one", "target_two"]
    assert result["comparison_device_groups"] == {"target_one": ["a1_b9"], "target_two": ["b1_b14"]}


def test_metadata_mapper_requires_clarification_for_cross_project_exact_code_in_comparison() -> None:
    modules = _load_agent_modules()
    DeviceInfo = modules["DeviceInfo"]

    candidate_one = DeviceInfo("a1_b9", "B2柜", project_id="p1", project_name="中国能建集团数据机房监控项目", project_code_name="cneb-room")
    candidate_one.tg = "TG1"
    candidate_two = DeviceInfo("a1_b9", "601-612", project_id="p2", project_name="平陆运河项目", project_code_name="pluyh")
    candidate_two.tg = "TG2"
    candidate_three = DeviceInfo("b1_b14", "电子楼 AA3-1 电源进线", project_id="p3", project_name="智慧物联网能效平台", project_code_name="iot-energy")
    candidate_three.tg = "TG233"

    engine = FakeMetadataEngine({
        "a1_b9": [candidate_one, candidate_two],
        "b1_b14": [candidate_three],
    })
    node = modules["MetadataMapperNode"](engine)

    result = node({
        "query_plan": {
            "query_mode": "comparison",
            "search_targets": ["a1_b9", "b1_b14"],
            "explicit_device_codes": ["a1_b9", "b1_b14"],
            "inferred_data_type": "ep",
            "has_sensor_intent": True,
            "has_comparison_intent": True,
            "confidence": 0.95,
        },
        "history": [],
    })

    assert result["clarification_required"] is True
    assert result["clarification_candidates"][0]["keyword"] == "a1_b9"
    assert result["device_codes"] == ["b1_b14"]
    assert result["tg_values"] == ["TG233"]


def test_metadata_mapper_project_hint_resolves_cross_project_exact_code() -> None:
    modules = _load_agent_modules()
    DeviceInfo = modules["DeviceInfo"]

    candidate_one = DeviceInfo("a1_b9", "B2柜", project_id="p1", project_name="中国能建集团数据机房监控项目", project_code_name="cneb-room")
    candidate_one.tg = "TG1"
    candidate_two = DeviceInfo("a1_b9", "B2柜", project_id="p2", project_name="IT资产管理系统", project_code_name="itam")
    candidate_two.tg = "TG9"
    candidate_three = DeviceInfo("b1_b14", "电子楼 AA3-1 电源进线", project_id="p3", project_name="智慧物联网能效平台", project_code_name="iot-energy")
    candidate_three.tg = "TG233"

    engine = FakeMetadataEngine({
        "a1_b9": [candidate_one, candidate_two],
        "b1_b14": [candidate_three],
    })
    node = modules["MetadataMapperNode"](engine)

    result = node({
        "query_plan": {
            "query_mode": "comparison",
            "search_targets": ["a1_b9", "b1_b14"],
            "explicit_device_codes": ["a1_b9", "b1_b14"],
            "project_hints": ["中国能建集团数据机房监控项目"],
            "inferred_data_type": "ep",
            "has_sensor_intent": True,
            "has_comparison_intent": True,
            "confidence": 0.95,
        },
        "history": [],
    })

    assert result.get("clarification_required") is not True
    assert result["comparison_device_groups"] == {"a1_b9": ["a1_b9"], "b1_b14": ["b1_b14"]}
    assert result["tg_values"] == ["TG1", "TG233"]


def test_metadata_mapper_requires_clarification_for_fuzzy_multi_match_sensor_query() -> None:
    modules = _load_agent_modules()
    DeviceInfo = modules["DeviceInfo"]

    candidate_one = DeviceInfo("a10_b1", "1#变压器集中器 味多美", project_id="p1", project_name="智慧物联网能效平台")
    candidate_one.tg = "TG1"
    candidate_two = DeviceInfo("a10_b2", "1#变压器集中器 金顺星宾馆", project_id="p2", project_name="智慧物联网能效平台")
    candidate_two.tg = "TG2"
    engine = FakeMetadataEngine({"变压器": [candidate_one, candidate_two]})
    node = modules["MetadataMapperNode"](engine)

    result = node({
        "query_plan": {
            "query_mode": "sensor_query",
            "search_targets": ["变压器"],
            "explicit_device_codes": [],
            "has_sensor_intent": True,
        },
        "history": [],
    })

    assert result["clarification_required"] is True
    assert result["clarification_candidates"][0]["keyword"] == "变压器"


def test_semantic_metadata_mapper_requires_clarification_for_cross_project_exact_code() -> None:
    modules = _load_agent_modules()
    DeviceInfo = modules["DeviceInfo"]

    candidate_one = DeviceInfo("a1_b9", "B2柜", project_id="p1", project_name="中国能建集团数据机房监控项目", project_code_name="cneb-room")
    candidate_one.tg = "TG1"
    candidate_two = DeviceInfo("a1_b9", "601-612", project_id="p2", project_name="平陆运河项目", project_code_name="pluyh")
    candidate_two.tg = "TG2"

    engine = FakeMetadataEngine({"a1_b9": [candidate_one, candidate_two]})
    node = modules["SemanticMetadataMapperNode"](metadata_engine=engine)

    result = node({
        "query_plan": {
            "query_mode": "sensor_query",
            "search_targets": ["a1_b9"],
            "explicit_device_codes": ["a1_b9"],
            "has_sensor_intent": True,
        },
        "history": [],
    })

    assert result["clarification_required"] is True
    assert result["clarification_candidates"][0]["keyword"] == "a1_b9"
    candidate = result["clarification_candidates"][0]["candidates"][0]
    assert candidate["match_type"] == "exact_code_conflict"
    assert candidate["match_reason"] == "精确码命中"
    assert candidate["match_score"] is None
    assert any(item.get("match_type") == "aggregate_all_option" for item in result["clarification_candidates"][0]["candidates"])


def test_semantic_metadata_mapper_comparison_allows_explicit_aggregate_scope() -> None:
    modules = _load_agent_modules()
    DeviceInfo = modules["DeviceInfo"]

    candidate_one = DeviceInfo("a1_b9", "B2?", project_id="p1", project_name="??????????????", project_code_name="cneb-room")
    candidate_one.tg = "TG1"
    candidate_two = DeviceInfo("a1_b9", "601-612", project_id="p2", project_name="??????", project_code_name="pluyh")
    candidate_two.tg = "TG2"
    candidate_three = DeviceInfo("b1_b14", "??? AA3-1 ????", project_id="p3", project_name="?????????", project_code_name="iot-energy")
    candidate_three.tg = "TG233"

    engine = FakeMetadataEngine({"a1_b9": [candidate_one, candidate_two], "b1_b14": [candidate_three]})
    node = modules["SemanticMetadataMapperNode"](metadata_engine=engine)

    result = node({
        "query": "汇总所有 a1_b9 与 b1_b14 哪个用电更多？",
        "query_plan": {
            "current_question": "汇总所有 a1_b9 与 b1_b14 哪个用电更多？",
            "query_mode": "comparison",
            "search_targets": ["a1_b9", "b1_b14"],
            "explicit_device_codes": ["a1_b9", "b1_b14"],
            "has_sensor_intent": True,
            "has_comparison_intent": True,
        },
        "history": [],
    })

    assert result.get("clarification_required") is not True
    assert result["comparison_device_groups"]["a1_b9"] == ["a1_b9", "a1_b9"]
    assert result["comparison_device_groups"]["b1_b14"] == ["b1_b14"]


def test_semantic_metadata_mapper_keeps_real_score_for_fuzzy_candidates() -> None:
    modules = _load_agent_modules()
    DeviceSemanticSearch = modules["DeviceSemanticSearch"]
    node = modules["SemanticMetadataMapperNode"](
        metadata_engine=FakeMetadataEngine({}),
        device_search=DeviceSemanticSearch(results=[]),
    )

    candidates = node._select_clarification_candidates([
        {
            "device_id": "a10_b1",
            "device_name": "1#?????? ???",
            "project_id": "p1",
            "project_name": "?????????",
            "project_code_name": "iot-energy",
            "tg": "TG1",
            "score": 88.0,
            "match_reason": "?????????",
            "matched_fields": ["device_name"],
        }
    ])

    assert candidates[0]["match_type"] == "semantic"
    assert candidates[0]["match_score"] == 88.0
    assert candidates[0]["match_reason"] == "?????????"


def test_sharding_router_prefers_query_plan_data_type() -> None:
    modules = _load_agent_modules()
    node = modules["ShardingRouterNode"](max_collections=12)

    result = node({
        "query_plan": {
            "query_mode": "sensor_query",
            "inferred_data_type": "u_line",
            "time_start": "2024-01-01",
            "time_end": "2024-01-31",
            "has_sensor_intent": True,
            "confidence": 0.9,
        },
        "intent": {"time_start": "2023-12-01", "time_end": "2023-12-31", "data_type": "ep"},
        "history": [],
    })

    assert result["collections"] == ["prefix_u_line_2024-01-01_2024-01-31_12"]
    assert result["data_tags"] == ["tag_for_prefix_u_line"]


def test_synthesizer_prefers_query_plan_data_type_for_fallback() -> None:
    modules = _load_agent_modules()
    node = modules["SynthesizerNode"](FakeLLM(should_raise=True))

    result = node({
        "query_plan": {
            "query_mode": "sensor_query",
            "inferred_data_type": "u_line",
            "has_sensor_intent": True,
            "response_style": "direct_answer",
            "confidence": 0.9,
        },
        "intent": {"time_start": "2024-01-01", "time_end": "2024-01-31"},
        "device_names": {"a1_b9": "Target One Device"},
        "statistics": {"avg": 220.5, "max": 231.0, "min": 218.0},
        "total_count": 24,
        "history": [],
        "error": None,
        "error_node": None,
    })

    assert "电压" in result["final_response"]


def test_synthesizer_prefers_query_plan_comparison_targets() -> None:
    modules = _load_agent_modules()
    node = modules["SynthesizerNode"](FakeLLM(should_raise=True))

    result = node({
        "query_plan": {
            "query_mode": "comparison",
            "search_targets": ["target_one", "target_two"],
            "inferred_data_type": "ep",
            "has_sensor_intent": True,
            "has_comparison_intent": True,
            "response_style": "compare",
            "confidence": 0.9,
        },
        "comparison_statistics": {
            "target_one": {"avg": 100.0, "sum": 500.0, "count": 5},
            "target_two": {"avg": 80.0, "sum": 400.0, "count": 5},
        },
        "comparison_targets": None,
        "is_comparison": False,
        "history": [],
        "error": None,
        "error_node": None,
        "total_count": 10,
    })

    assert "target_one" in result["final_response"]
    assert "target_two" in result["final_response"]
    assert "对比" in result["final_response"]


class FakeSensorFetchResult:
    def __init__(self, *, total_count=0, statistics=None, query_info=None, failed_collections=None, is_sampled=False):
        self.data = []
        self.total_count = total_count
        self.statistics = statistics
        self.query_info = query_info
        self.failed_collections = failed_collections or []
        self.is_sampled = is_sampled


class FakeDataFetcher:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def fetch_sync(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def test_parallel_fetcher_prefers_query_plan_time_range_and_context() -> None:
    modules = _load_agent_modules()
    fetcher = FakeDataFetcher(
        FakeSensorFetchResult(
            total_count=24,
            statistics={"avg": 220.5, "max": 231.0, "min": 218.0},
            query_info={"type": "MongoDB"},
        )
    )
    node = modules["ParallelFetcherNode"](fetcher)

    result = node({
        "query_plan": {
            "query_mode": "sensor_query",
            "search_targets": ["a1_b9"],
            "inferred_data_type": "u_line",
            "time_start": "2024-01-01",
            "time_end": "2024-01-31",
            "response_style": "direct_answer",
            "aggregation": "raw",
            "has_sensor_intent": True,
            "confidence": 0.9,
        },
        "intent": None,
        "device_codes": ["a1_b9"],
        "collections": ["prefix_u_line_2024-01-01_2024-01-31_12"],
        "data_tags": ["tag_for_prefix_u_line"],
        "history": [],
    })

    assert len(fetcher.calls) == 1
    assert fetcher.calls[0]["start_time"].strftime("%Y-%m-%d") == "2024-01-01"
    assert fetcher.calls[0]["end_time"].strftime("%Y-%m-%d") == "2024-01-31"
    assert result["query_info"]["query_plan_context"]["query_mode"] == "sensor_query"
    assert result["query_info"]["query_plan_context"]["data_type"] == "u_line"
    assert result["query_info"]["query_plan_context"]["time_start"] == "2024-01-01"
    assert result["query_info"]["query_plan_context"]["time_end"] == "2024-01-31"


def test_action_override_policy_node_short_circuits_project_listing() -> None:
    modules = _load_agent_modules()
    node = modules["ActionOverridePolicyNode"](
        FakeMetadataEngine(
            {},
            projects=[{"id": "p1", "project_name": "Demo Project", "code_name": "demo"}],
        )
    )

    result = node({
        "query_plan": {
            "query_mode": "project_listing",
            "search_targets": [],
            "explicit_device_codes": [],
            "has_sensor_intent": False,
            "has_project_listing_intent": True,
            "confidence": 0.95,
        },
        "intent": {},
        "history": [],
    })

    assert result["override_action"] == "list_projects"
    assert result["override_terminal"] is True
    assert result["show_table"] is True
    assert result["table_type"] == "projects"
    assert result["query_info"]["projects"][0]["project_name"] == "Demo Project"
    assert modules["route_after_action_override"](result) == "terminal"


def test_action_override_policy_node_keeps_default_when_scope_not_pre_resolved() -> None:
    modules = _load_agent_modules()
    device_a = modules["DeviceInfo"]("a1_b9", "Target One Device")
    device_b = modules["DeviceInfo"]("b1_b14", "Target Two Device")
    node = modules["ActionOverridePolicyNode"](
        FakeMetadataEngine({"a1_b9": [device_a], "b1_b14": [device_b]})
    )

    result = node({
        "query_plan": {
            "query_mode": "comparison",
            "search_targets": ["a1_b9", "b1_b14"],
            "explicit_device_codes": ["a1_b9", "b1_b14"],
            "inferred_data_type": "ep",
            "time_start": "2024-01-01",
            "time_end": "2024-01-31",
            "has_sensor_intent": True,
            "has_comparison_intent": True,
            "confidence": 0.92,
        },
        "intent": {},
        "history": [],
    })

    assert result["override_action"] == "search_devices"
    assert result["override_terminal"] is False
    assert result.get("device_codes") is None
    assert modules["route_after_action_override"](result) == "continue"



def test_route_after_query_plan_sends_explicit_sensor_query_to_metadata_first() -> None:
    modules = _load_agent_modules()

    route = modules["route_after_query_plan"]
    next_step = route({
        "query_plan": {
            "query_mode": "comparison",
            "search_targets": ["a1_b9", "b1_b14"],
            "explicit_device_codes": ["a1_b9", "b1_b14"],
            "inferred_data_type": "ep",
            "has_sensor_intent": True,
            "has_comparison_intent": True,
            "confidence": 0.92,
        },
        "intent": {},
        "history": [],
        "error": None,
    })

    assert next_step == "metadata"



def test_route_after_query_plan_general_query_goes_to_synthesizer() -> None:
    modules = _load_agent_modules()

    route = modules["route_after_query_plan"]
    next_step = route({
        "query_plan": {
            "query_mode": "general",
            "search_targets": [],
            "explicit_device_codes": [],
            "has_sensor_intent": False,
            "confidence": 0.60,
        },
        "intent": {},
        "history": [],
        "error": None,
    })

    assert next_step == "synthesizer"



def test_sharding_router_history_result_uses_readable_text() -> None:
    modules = _load_agent_modules()
    node = modules["ShardingRouterNode"](max_collections=10)

    result = node(
        {
            "history": [],
            "query_plan": {
                "query_mode": "sensor_query",
                "inferred_data_type": "ep",
                "time_start": "2024-01-01",
                "time_end": "2024-01-31",
            },
        }
    )

    assert result["history"][-1]["result"].startswith("计算出")


def test_synthesizer_history_result_uses_readable_text() -> None:
    modules = _load_agent_modules()
    node = modules["SynthesizerNode"](FakeLLM())

    result = node(
        {
            "user_query": "a1_b9 设备2024年1月是否存在异常用电时间点",
            "history": [],
            "total_count": 4,
            "raw_data": [
                {"logTime": "2024-01-01 00:00:00", "val": 100, "device": "a1_b9", "tag": "ep"},
                {"logTime": "2024-01-01 01:00:00", "val": 101, "device": "a1_b9", "tag": "ep"},
                {"logTime": "2024-01-01 02:00:00", "val": 160, "device": "a1_b9", "tag": "ep"},
                {"logTime": "2024-01-01 03:00:00", "val": 102, "device": "a1_b9", "tag": "ep"},
            ],
            "query_plan": {
                "current_question": "a1_b9 设备2024年1月是否存在异常用电时间点",
                "query_mode": "anomaly_points",
                "inferred_data_type": "ep",
                "search_targets": ["a1_b9"],
                "has_sensor_intent": True,
                "has_anomaly_point_intent": True,
                "response_style": "direct_answer",
                "time_start": "2024-01-01",
                "time_end": "2024-01-01",
            },
        }
    )

    assert result["history"][-1]["result"].startswith("生成响应完成")


def test_metadata_mapper_single_explicit_code_reuses_session_alias_without_reclarification() -> None:
    modules = _load_agent_modules()
    DeviceInfo = modules["DeviceInfo"]
    engine = FakeMetadataEngine({
        "a1_b9": [
            DeviceInfo("a1_b9", "B2?", project_id="p1", project_name="??????????????", project_code_name="cneb-room"),
            DeviceInfo("a1_b9", "601-612", project_id="p2", project_name="??????", project_code_name="pluyh"),
        ]
    })
    node = modules["MetadataMapperNode"](metadata_engine=engine)

    result = node({
        "query": "a1_b9 ?? 2024?1?????????50?",
        "query_plan": {
            "query_mode": "sensor_query",
            "search_targets": ["a1_b9"],
            "explicit_device_codes": ["a1_b9"],
            "has_sensor_intent": True,
        },
        "alias_memory": {
            "a1_b9": {
                "device": "a1_b9",
                "name": "B2?",
                "project_id": "p1",
                "project_name": "??????????????",
                "project_code_name": "cneb-room",
                "tg": "TG232",
            }
        },
        "history": [],
    })

    assert result.get("clarification_required") is not True
    assert result["device_codes"] == ["a1_b9"]
    assert result["tg_values"] == ["TG232"]


def test_semantic_metadata_mapper_single_explicit_code_reuses_session_alias_without_reclarification() -> None:
    modules = _load_agent_modules()
    node = modules["SemanticMetadataMapperNode"](metadata_engine=FakeMetadataEngine({}))

    result = node({
        "query": "a1_b9 ?? 2024?1?????????50?",
        "query_plan": {
            "query_mode": "sensor_query",
            "search_targets": ["a1_b9"],
            "explicit_device_codes": ["a1_b9"],
            "has_sensor_intent": True,
        },
        "alias_memory": {
            "a1_b9": {
                "device": "a1_b9",
                "name": "B2?",
                "project_id": "p1",
                "project_name": "??????????????",
                "project_code_name": "cneb-room",
                "tg": "TG232",
            }
        },
        "history": [],
    })

    assert result.get("clarification_required") is not True
    assert result["device_codes"] == ["a1_b9"]
    assert result["tg_values"] == ["TG232"]


def test_metadata_mapper_comparison_reuses_alias_scope_when_all_explicit_targets_confirmed() -> None:
    modules = _load_agent_modules()
    DeviceInfo = modules["DeviceInfo"]
    engine = FakeMetadataEngine({
        "a1_b9": [
            DeviceInfo("a1_b9", "B2?", project_id="p1", project_name="??????????????", project_code_name="cneb-room"),
            DeviceInfo("a1_b9", "601-612", project_id="p2", project_name="??????", project_code_name="pluyh"),
        ],
        "b1_b14": [
            DeviceInfo("b1_b14", "??? AA3-1 ????", project_id="p3", project_name="?????????", project_code_name="iot-energy"),
        ],
    })
    node = modules["MetadataMapperNode"](metadata_engine=engine)

    result = node({
        "query": "a1_b9 ? b1_b14 ???????",
        "query_plan": {
            "current_question": "a1_b9 ? b1_b14 ???????",
            "query_mode": "comparison",
            "search_targets": ["a1_b9", "b1_b14"],
            "explicit_device_codes": ["a1_b9", "b1_b14"],
            "has_sensor_intent": True,
            "has_comparison_intent": True,
        },
        "alias_memory": {
            "a1_b9": {
                "device": "a1_b9",
                "name": "B2?",
                "project_id": "p1",
                "project_name": "??????????????",
                "project_code_name": "cneb-room",
                "tg": "TG232",
            },
            "b1_b14": {
                "device": "b1_b14",
                "name": "??? AA3-1 ????",
                "project_id": "p3",
                "project_name": "?????????",
                "project_code_name": "iot-energy",
                "tg": "TG233",
            },
        },
        "history": [],
    })

    assert result.get("clarification_required") is not True
    assert result["comparison_device_groups"]["a1_b9"] == ["a1_b9"]
    assert result["comparison_device_groups"]["b1_b14"] == ["b1_b14"]
    assert sorted(result["tg_values"]) == ["TG232", "TG233"]


def test_metadata_mapper_comparison_reuses_confirmed_alias_even_if_other_target_is_only_uniquely_resolved() -> None:
    modules = _load_agent_modules()
    DeviceInfo = modules["DeviceInfo"]
    a1_1 = DeviceInfo("a1_b9", "B2?", project_id="p1", project_name="??????????????", project_code_name="cneb-room")
    a1_1.tg = "TG232"
    a1_2 = DeviceInfo("a1_b9", "601-612", project_id="p2", project_name="??????", project_code_name="pluyh")
    a1_2.tg = "TG19"
    b1 = DeviceInfo("b1_b14", "??? AA3-1 ????", project_id="p3", project_name="?????????", project_code_name="iot-energy")
    b1.tg = "TG233"
    engine = FakeMetadataEngine({
        "a1_b9": [a1_1, a1_2],
        "b1_b14": [b1],
    })
    node = modules["MetadataMapperNode"](metadata_engine=engine)

    result = node({
        "query": "a1_b9 ? b1_b14 ???????",
        "query_plan": {
            "current_question": "a1_b9 ? b1_b14 ???????",
            "query_mode": "comparison",
            "search_targets": ["a1_b9", "b1_b14"],
            "explicit_device_codes": ["a1_b9", "b1_b14"],
            "has_sensor_intent": True,
            "has_comparison_intent": True,
        },
        "alias_memory": {
            "a1_b9": {
                "device": "a1_b9",
                "name": "B2?",
                "project_id": "p1",
                "project_name": "??????????????",
                "project_code_name": "cneb-room",
                "tg": "TG232",
            },
        },
        "history": [],
    })

    assert result.get("clarification_required") is not True
    assert result["comparison_device_groups"]["a1_b9"] == ["a1_b9"]
    assert result["comparison_device_groups"]["b1_b14"] == ["b1_b14"]
    assert result["comparison_scope_groups"]["a1_b9"][0]["tg"] == "TG232"
    assert result["comparison_scope_groups"]["b1_b14"][0]["tg"] == "TG233"
    assert sorted(result["tg_values"]) == ["TG232", "TG233"]
def test_semantic_metadata_mapper_comparison_reuses_alias_scope_when_all_explicit_targets_confirmed() -> None:
    modules = _load_agent_modules()
    node = modules["SemanticMetadataMapperNode"](metadata_engine=FakeMetadataEngine({}))

    result = node({
        "query": "a1_b9 ? b1_b14 ???????",
        "query_plan": {
            "current_question": "a1_b9 ? b1_b14 ???????",
            "query_mode": "comparison",
            "search_targets": ["a1_b9", "b1_b14"],
            "explicit_device_codes": ["a1_b9", "b1_b14"],
            "has_sensor_intent": True,
            "has_comparison_intent": True,
        },
        "alias_memory": {
            "a1_b9": {
                "device": "a1_b9",
                "name": "B2?",
                "project_id": "p1",
                "project_name": "??????????????",
                "project_code_name": "cneb-room",
                "tg": "TG232",
            },
            "b1_b14": {
                "device": "b1_b14",
                "name": "??? AA3-1 ????",
                "project_id": "p3",
                "project_name": "?????????",
                "project_code_name": "iot-energy",
                "tg": "TG233",
            },
        },
        "history": [],
    })

    assert result.get("clarification_required") is not True
    assert result["comparison_device_groups"]["a1_b9"] == ["a1_b9"]
    assert result["comparison_device_groups"]["b1_b14"] == ["b1_b14"]
    assert sorted(result["tg_values"]) == ["TG232", "TG233"]

def test_semantic_metadata_mapper_comparison_reuses_confirmed_alias_even_if_other_target_is_only_uniquely_resolved() -> None:
    modules = _load_agent_modules()
    DeviceInfo = modules["DeviceInfo"]
    a1_1 = DeviceInfo("a1_b9", "B2?", project_id="p1", project_name="??????????????", project_code_name="cneb-room")
    a1_1.tg = "TG232"
    a1_2 = DeviceInfo("a1_b9", "601-612", project_id="p2", project_name="??????", project_code_name="pluyh")
    a1_2.tg = "TG19"
    b1 = DeviceInfo("b1_b14", "??? AA3-1 ????", project_id="p3", project_name="?????????", project_code_name="iot-energy")
    b1.tg = "TG233"
    engine = FakeMetadataEngine({
        "a1_b9": [a1_1, a1_2],
        "b1_b14": [b1],
    })
    node = modules["SemanticMetadataMapperNode"](metadata_engine=engine)

    result = node({
        "query": "a1_b9 ? b1_b14 ???????",
        "query_plan": {
            "current_question": "a1_b9 ? b1_b14 ???????",
            "query_mode": "comparison",
            "search_targets": ["a1_b9", "b1_b14"],
            "explicit_device_codes": ["a1_b9", "b1_b14"],
            "has_sensor_intent": True,
            "has_comparison_intent": True,
        },
        "alias_memory": {
            "a1_b9": {
                "device": "a1_b9",
                "name": "B2?",
                "project_id": "p1",
                "project_name": "??????????????",
                "project_code_name": "cneb-room",
                "tg": "TG232",
            },
        },
        "history": [],
    })

    assert result.get("clarification_required") is not True
    assert result["comparison_device_groups"]["a1_b9"] == ["a1_b9"]
    assert result["comparison_device_groups"]["b1_b14"] == ["b1_b14"]
    assert result["comparison_scope_groups"]["a1_b9"][0]["tg"] == "TG232"
    assert result["comparison_scope_groups"]["b1_b14"][0]["tg"] == "TG233"
    assert sorted(result["tg_values"]) == ["TG232", "TG233"]


def test_metadata_mapper_lists_devices_by_project_hint() -> None:
    modules = _load_agent_modules()
    DeviceInfo = modules["DeviceInfo"]
    device = DeviceInfo("bj_b1", "????????", project_id="p1", project_name="??????")
    device.tg = "TG100"
    engine = FakeMetadataEngine(
        mapping={"?????????": []},
        projects=[{"id": "p1", "project_name": "??????", "project_code_name": "beijing-power"}],
        project_devices={"p1": [device]},
    )
    node = modules["MetadataMapperNode"](engine)

    result = node({
        "history": [],
        "query_plan": {
            "query_mode": "device_listing",
            "search_targets": ["?????????"],
            "project_hints": ["??????"],
            "has_device_listing_intent": True,
        },
    })

    assert result["show_table"] is True
    assert result["table_type"] == "devices"
    assert result["devices"][0]["device"] == "bj_b1"
    assert result["devices"][0]["project_name"] == "??????"


def test_dag_sharding_router_prefers_requested_tags_for_single_phase() -> None:
    modules = _load_agent_modules()
    node = modules["ShardingRouterNode"]()
    GraphState = sys.modules['src.agent.types'].GraphState
    state = GraphState(
        user_query='a2_b1?2024?1?1??ua???',
        query_plan={
            'current_question': 'a2_b1?2024?1?1??ua???',
            'inferred_data_type': 'ua',
            'time_start': '2024-01-01',
            'time_end': '2024-01-01',
            'raw_plan': {'requested_tags': ['ua']},
        },
        intent={'data_type': 'ua', 'time_start': '2024-01-01', 'time_end': '2024-01-01'},
        history=[],
    )

    next_state = node(state)

    assert next_state['data_tags'] == ['ua']



def test_query_time_range_resolves_relaxed_exact_day_from_confirmation_text() -> None:
    from datetime import datetime

    _load_agent_modules()
    query_time_range = sys.modules['src.agent.query_time_range']

    result = query_time_range.resolve_time_range_from_query(
        'a2_b1?2024?1?1??ua???',
        now=datetime(2026, 3, 26, 12, 0, 0),
    )

    assert result == {'start_time': '2024-01-01', 'end_time': '2024-01-01'}



def test_fallback_query_plan_keeps_exact_day_for_confirmation_text() -> None:
    _load_agent_modules()
    query_plan = sys.modules['src.agent.query_plan']

    plan = query_plan.fallback_query_plan('a2_b1?2024?1?1??ua???')

    assert plan.time_start == '2024-01-01'
    assert plan.time_end == '2024-01-01'
    assert plan.inferred_data_type == 'ua'
