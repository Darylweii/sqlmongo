import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.chat_memory_service import ChatMemoryService
from src.user_memory_store import UserMemoryStore


class _FakeDevice:
    def __init__(self, device: str, name: str, project_name: str, tg: str = "TG-AC"):
        self.device = device
        self.name = name
        self.project_id = "p-ac"
        self.project_name = project_name
        self.project_code_name = "ac_proj"
        self.tg = tg
        self.device_type = "meter"

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
    def __init__(self, *args, **kwargs):
        self._device = _FakeDevice("ac_001", "空调主机", "智慧物联网能效平台")

    def search_devices(self, keyword):
        normalized = str(keyword or "").strip().lower()
        if normalized == "ac_001":
            return [self._device], False
        return [], False

    def list_all_devices(self):
        return [self._device]

    def list_projects(self):
        return []

    def get_project_device_stats(self):
        return []


class _FakeSemanticRuleStore:
    def list_rules(self, keyword="", status="enabled"):
        normalized = str(keyword or "").strip()
        if normalized == "冷气":
            return [{
                "canonical_term": "空调",
                "alias_terms": ["冷气"],
                "target_value": "空调",
            }]
        return []


class _MemoryAwareFakeAgent:
    def __init__(self):
        self.messages = []

    def run_with_progress(self, message_with_history):
        text = str(message_with_history or "")
        self.messages.append(text)
        if "空调" in text:
            yield {
                "type": "final_answer",
                "response": "已按“空调”继续查询，并返回电流结果。",
                "show_table": True,
                "table_type": "sensor_data",
                "query_params": {
                    "device_codes": ["ac_001"],
                    "start_time": "2026-03-30",
                    "end_time": "2026-03-30",
                    "data_type": "i",
                    "page": 1,
                    "page_size": 50,
                },
                "analysis": {"mode": "single"},
                "chart_specs": [],
                "show_charts": False,
                "total_duration_ms": 8,
                "clarification_required": False,
                "clarification_candidates": None,
            }
            return
        yield {
            "type": "final_answer",
            "response": "未找到与“冷气”匹配的设备，请检查设备名称、代号或换一种说法。",
            "show_table": False,
            "table_type": "",
            "query_params": None,
            "analysis": None,
            "chart_specs": [],
            "show_charts": False,
            "total_duration_ms": 3,
            "clarification_required": False,
            "clarification_candidates": None,
        }


def _parse_sse_events(response_text: str):
    events = []
    for line in str(response_text or "").splitlines():
        if not line.startswith("data: "):
            continue
        events.append(json.loads(line[6:]))
    return events


def _install_web_app_stubs(monkeypatch) -> None:
    for name in [
        "web.app",
        "src.agent",
        "src.agent.orchestrator",
        "src.agent.query_entities",
        "src.config",
        "src.metadata.metadata_engine",
        "src.fetcher.data_fetcher",
        "src.analysis",
        "src.compressor.context_compressor",
        "src.entity_resolver",
        "src.tools.sensor_tool",
        "src.router.collection_router",
        "pymongo",
        "langchain_openai",
    ]:
        sys.modules.pop(name, None)

    config_module = types.ModuleType("src.config")
    config_module.load_config = lambda: SimpleNamespace(
        mysql=SimpleNamespace(connection_string="mysql://test"),
        mongodb=SimpleNamespace(uri="mongodb://test", database_name="testdb"),
        semantic_layer=SimpleNamespace(dashscope_api_key="", embedding_model="text-embedding-v4", embedding_dimensions=1024),
        agent=SimpleNamespace(orchestrator_type="dag"),
    )
    monkeypatch.setitem(sys.modules, "src.config", config_module)

    metadata_module = types.ModuleType("src.metadata.metadata_engine")
    metadata_module.MetadataEngine = _FakeMetadataEngine
    monkeypatch.setitem(sys.modules, "src.metadata.metadata_engine", metadata_module)

    fetcher_module = types.ModuleType("src.fetcher.data_fetcher")
    class _FakeDataFetcher:
        def __init__(self, *args, **kwargs):
            pass
    fetcher_module.DataFetcher = _FakeDataFetcher
    monkeypatch.setitem(sys.modules, "src.fetcher.data_fetcher", fetcher_module)

    analysis_module = types.ModuleType("src.analysis")
    class _FakeInsightEngine:
        def analyze(self, *args, **kwargs):
            return {}
    analysis_module.InsightEngine = _FakeInsightEngine
    monkeypatch.setitem(sys.modules, "src.analysis", analysis_module)

    compressor_module = types.ModuleType("src.compressor.context_compressor")
    class _FakeContextCompressor:
        def __init__(self, *args, **kwargs):
            pass
    compressor_module.ContextCompressor = _FakeContextCompressor
    monkeypatch.setitem(sys.modules, "src.compressor.context_compressor", compressor_module)

    agent_package = types.ModuleType("src.agent")
    agent_package.__path__ = []
    agent_package.DAGOrchestrator = type("DAGOrchestrator", (), {})
    monkeypatch.setitem(sys.modules, "src.agent", agent_package)

    agent_orchestrator_module = types.ModuleType("src.agent.orchestrator")
    agent_orchestrator_module.create_agent_with_streaming = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "src.agent.orchestrator", agent_orchestrator_module)

    query_entities_module = types.ModuleType("src.agent.query_entities")
    query_entities_module.allows_explicit_multi_scope_aggregation = lambda *args, **kwargs: False
    monkeypatch.setitem(sys.modules, "src.agent.query_entities", query_entities_module)

    resolver_module = types.ModuleType("src.entity_resolver")
    class _FakeChromaEntityResolver:
        def __init__(self, *args, **kwargs):
            pass
    resolver_module.ChromaEntityResolver = _FakeChromaEntityResolver
    monkeypatch.setitem(sys.modules, "src.entity_resolver", resolver_module)

    sensor_tool_module = types.ModuleType("src.tools.sensor_tool")
    sensor_tool_module.fetch_sensor_data_with_components = lambda *args, **kwargs: {"success": True, "data": [], "total_count": 0, "statistics": {}, "analysis": {}}
    monkeypatch.setitem(sys.modules, "src.tools.sensor_tool", sensor_tool_module)

    router_module = types.ModuleType("src.router.collection_router")
    router_module.get_collection_prefix = lambda *args, **kwargs: "source_data_i_"
    router_module.get_target_collections = lambda *args, **kwargs: []
    router_module.get_data_tags = lambda *args, **kwargs: ["i"]
    monkeypatch.setitem(sys.modules, "src.router.collection_router", router_module)

    pymongo_module = types.ModuleType("pymongo")
    class _FakeMongoClient:
        def __init__(self, *args, **kwargs):
            pass
    pymongo_module.MongoClient = _FakeMongoClient
    monkeypatch.setitem(sys.modules, "pymongo", pymongo_module)

    openai_module = types.ModuleType("langchain_openai")
    class _FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            pass
    openai_module.ChatOpenAI = _FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", openai_module)


def _load_memory_regression_app(monkeypatch, tmp_path: Path):
    _install_web_app_stubs(monkeypatch)
    module = importlib.import_module("web.app")
    module = importlib.reload(module)
    module.SESSION_STATE.clear()

    metadata_engine = _FakeMetadataEngine()
    semantic_rule_store = _FakeSemanticRuleStore()
    user_memory_store = UserMemoryStore(tmp_path / "app_memory.sqlite3")
    chat_memory_service = ChatMemoryService(
        memory_store=user_memory_store,
        semantic_rule_store=semantic_rule_store,
        normalize_alias_key=module._normalize_alias_key,
        lookup_device_by_code=module._lookup_device_by_code,
        build_device_snapshot=module._build_device_snapshot,
        logger=module.logger,
    )
    agent = _MemoryAwareFakeAgent()

    monkeypatch.setattr(module, "metadata_engine", metadata_engine)
    monkeypatch.setattr(module, "semantic_rule_store", semantic_rule_store)
    monkeypatch.setattr(module, "user_memory_store", user_memory_store)
    monkeypatch.setattr(module, "chat_memory_service", chat_memory_service)
    monkeypatch.setattr(module, "_create_chat_agent", lambda **_kwargs: agent)
    monkeypatch.setattr(module, "ANSWER_STREAM_CHUNK_DELAY_MS", 0)
    return module, agent, user_memory_store


def test_memory_suggestion_returns_recommended_alias(monkeypatch, tmp_path: Path) -> None:
    app_module, _agent, _store = _load_memory_regression_app(monkeypatch, tmp_path)

    with TestClient(app_module.app) as client:
        response = client.post(
            "/api/chat/stream",
            json={
                "message": "查一下冷气设备今天的电流",
                "history": [],
                "session_id": "mem-suggest-s1",
                "user_id": "user-memory-reg",
            },
        )

    events = _parse_sse_events(response.text)
    complete_event = events[-1]
    assert complete_event["memory_suggestion"]["alias_text"] == "冷气"
    assert complete_event["memory_suggestion"]["recommended_canonical"] == "空调"
    assert complete_event["memory_suggestion"]["actions"] == ["remember", "query_once", "dismiss"]
    assert complete_event["query_params"] is None


def test_memory_suggestion_remember_continues_query_and_persists(monkeypatch, tmp_path: Path) -> None:
    app_module, agent, store = _load_memory_regression_app(monkeypatch, tmp_path)
    question = "查一下冷气设备今天的电流"

    with TestClient(app_module.app) as client:
        first = client.post(
            "/api/chat/stream",
            json={
                "message": question,
                "history": [],
                "session_id": "mem-suggest-s1",
                "user_id": "user-memory-reg",
            },
        )
        first_complete = _parse_sse_events(first.text)[-1]
        assert first_complete["memory_suggestion"]["recommended_canonical"] == "空调"

        second = client.post(
            "/api/chat/stream",
            json={
                "message": question,
                "history": [],
                "session_id": "mem-suggest-s1",
                "user_id": "user-memory-reg",
                "alias_confirmation": {
                    "action": "create_memory",
                    "alias_text": "冷气",
                    "canonical_text": "空调",
                    "scope_type": "global",
                    "scope_value": "global",
                    "source": "memory_suggestion",
                    "continue_query": True,
                    "original_question": question,
                },
            },
        )
        second_complete = _parse_sse_events(second.text)[-1]

        third = client.post(
            "/api/chat/stream",
            json={
                "message": question,
                "history": [],
                "session_id": "mem-suggest-s2",
                "user_id": "user-memory-reg",
            },
        )
        third_complete = _parse_sse_events(third.text)[-1]

    assert second_complete["memory_action_result"]["action"] == "created"
    assert second_complete["memory_action_result"]["continue_query"] is True
    assert second_complete["query_params"]["device_codes"] == ["ac_001"]
    assert second_complete["query_params"]["data_type"] == "i"
    assert second_complete["memory_suggestion"] is None
    assert "空调" in agent.messages[1]
    assert "冷气" not in agent.messages[1]

    items = store.list_alias_memories(user_id="user-memory-reg")
    assert len(items) == 1
    assert items[0]["alias_text"] == "冷气"
    assert items[0]["canonical_text"] == "空调"

    assert third_complete["memory_action_result"]["action"] == "applied"
    assert third_complete["memory_suggestion"] is None
    assert third_complete["query_params"]["device_codes"] == ["ac_001"]
    assert "空调" in agent.messages[2]
    assert "冷气" not in agent.messages[2]
