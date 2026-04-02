import importlib


def test_orchestrator_final_query_params_keep_tg_values(monkeypatch) -> None:
    orchestrator_module = importlib.import_module("src.agent.orchestrator")

    def fake_fetch_sensor_data_with_components(**kwargs):
        return {
            "success": True,
            "data": [],
            "total_count": 0,
            "page": kwargs.get("page", 1),
            "page_size": kwargs.get("page_size", 50),
            "total_pages": 0,
            "has_more": False,
            "statistics": None,
            "analysis": None,
            "chart_specs": [],
            "show_charts": False,
            "query_info": {
                "query": {
                    "device": {"$in": kwargs.get("device_codes", [])},
                    "tg": {"$in": kwargs.get("tg_values", [])},
                    "logTime": {
                        "$gte": kwargs.get("start_time", ""),
                        "$lte": kwargs.get("end_time", ""),
                    },
                },
                "collections": ["source_data_u_line_202603"],
            },
        }

    monkeypatch.setattr(orchestrator_module, "fetch_sensor_data_with_components", fake_fetch_sensor_data_with_components)

    agent = orchestrator_module.StreamingAgentOrchestrator(
        llm=object(),
        llm_non_streaming=object(),
        data_fetcher=object(),
        alias_memory={
            "a1_b9": {
                "device": "a1_b9",
                "name": "B2柜",
                "project_id": "p1",
                "project_name": "中国能建集团数据机房监控项目",
                "tg": "TG232",
            }
        },
    )

    monkeypatch.setattr(
        agent,
        "_build_heuristic_action",
        lambda state, fast_path=False: {
            "thought": "use direct sensor query",
            "action": "get_sensor_data",
            "action_input": {
                "device_codes": ["a1_b9"],
                "data_type": "u_line",
                "start_time": "2026-03-18",
                "end_time": "2026-03-18",
                "user_query": "a1_b9 今天的线电压数据",
            },
        },
    )

    events = list(agent.run_with_progress("a1_b9 今天的线电压数据"))
    final_event = next(event for event in events if event.get("type") == "final_answer")

    assert final_event["show_table"] is True
    assert final_event["table_type"] == "sensor_data"
    assert final_event["query_params"]["device_codes"] == ["a1_b9"]
    assert final_event["query_params"]["tg_values"] == ["TG232"]
    assert final_event["query_params"]["data_type"] == "u_line"



def test_partial_confirmed_explicit_compare_requires_reselection_and_offers_aggregate(monkeypatch) -> None:
    orchestrator_module = importlib.import_module("src.agent.orchestrator")
    agent = orchestrator_module.StreamingAgentOrchestrator(
        llm=object(),
        llm_non_streaming=object(),
        data_fetcher=object(),
        alias_memory={
            "a1_b9": {
                "device": "a1_b9",
                "name": "B2柜",
                "project_id": "p1",
                "project_name": "中国能建集团数据机房监控项目",
                "tg": "TG232",
            }
        },
    )

    def fake_lookup(device_code, query_text=""):
        if device_code == "a1_b9":
            return ([{
                "device": "a1_b9",
                "name": "B2柜",
                "project_id": "p1",
                "project_name": "中国能建集团数据机房监控项目",
                "project_code_name": "cneb-room",
                "tg": "TG232",
                "match_score": 100.0,
                "matched_fields": ["device"],
                "match_reason": "exact_device_code",
            }, {
                "device": "a1_b9",
                "name": "601-612",
                "project_id": "p2",
                "project_name": "平陆运河项目",
                "project_code_name": "pluyh",
                "tg": "TG19",
                "match_score": 100.0,
                "matched_fields": ["device"],
                "match_reason": "exact_device_code",
            }], {"type": "exact_device_code", "device_code": "a1_b9", "candidate_count": 2})
        if device_code == "b1_b14":
            return ([{
                "device": "b1_b14",
                "name": "电子楼 AA3-1 电源进线",
                "project_id": "p3",
                "project_name": "智慧物联网能效平台",
                "project_code_name": "iot-energy",
                "tg": "TG314",
                "match_score": 100.0,
                "matched_fields": ["device"],
                "match_reason": "exact_device_code",
            }], {"type": "exact_device_code", "device_code": "b1_b14", "candidate_count": 1})
        return ([], None)

    monkeypatch.setattr(agent, "_lookup_exact_device_code_candidates", fake_lookup)
    monkeypatch.setattr(
        agent,
        "_build_heuristic_action",
        lambda state, fast_path=False: {
            "thought": "compare explicit devices",
            "action": "get_sensor_data",
            "action_input": {
                "device_codes": ["a1_b9", "b1_b14"],
                "data_type": "ep",
                "start_time": "2026-03-18",
                "end_time": "2026-03-18",
                "user_query": "a1_b9 和 b1_b14 哪个耗电更多",
            },
        },
    )

    events = list(agent.run_with_progress("a1_b9 和 b1_b14 哪个耗电更多"))
    final_event = next(event for event in events if event.get("type") == "final_answer")

    assert final_event.get("clarification_required") is True
    group = final_event["clarification_candidates"][0]
    assert group["keyword"] == "a1_b9"
    assert any(item.get("match_type") == "aggregate_all_option" for item in group["candidates"])
    assert any(item.get("match_reason") == "精确码命中" for item in group["candidates"] if item.get("match_type") == "exact_code_conflict")



def test_sensor_terminal_response_prefers_ranked_timepoint_answer() -> None:
    orchestrator_module = importlib.import_module("src.agent.orchestrator")
    agent = orchestrator_module.LLMAgent(llm=None)

    response = agent._build_sensor_terminal_response(
        {
            "success": True,
            "total_count": 744,
            "analysis": {"mode": "single"},
            "focused_result": {
                "mode": "ranked_timepoints",
                "order": "desc",
                "metric": "用电量",
                "unit": "kWh",
                "rows": [
                    {"rank": 1, "time": "2024-01-31 23:00:05", "device": "a1_b9", "tag": "ep", "value": 208484.0},
                    {"rank": 2, "time": "2024-01-31 22:00:04", "device": "a1_b9", "tag": "ep", "value": 208470.0},
                ],
            },
        },
        "fallback",
    )

    assert "【结果】" in response
    assert "前 2 个时间点" in response
    assert "2024-01-31 23:00:05 / a1_b9 / ep / 208484.00 kWh" in response
    assert "【关键指标】" not in response


def test_sensor_terminal_response_prefers_ranked_bucket_answer() -> None:
    orchestrator_module = importlib.import_module("src.agent.orchestrator")
    agent = orchestrator_module.LLMAgent(llm=None)

    response = agent._build_sensor_terminal_response(
        {
            "success": True,
            "total_count": 744,
            "analysis": {"mode": "single"},
            "focused_result": {
                "mode": "ranked_buckets",
                "order": "desc",
                "metric": "用电量",
                "unit": "kWh",
                "granularity": "day",
                "aggregation_note": "累计型指标按每个周期内末值减首值聚合。",
                "rows": [
                    {"rank": 1, "time": "2024-01-31", "value": 1024.0, "sample_count": 24},
                    {"rank": 2, "time": "2024-01-30", "value": 980.0, "sample_count": 24},
                ],
            },
        },
        "fallback",
    )

    assert "【结果】" in response
    assert "前 2 个周期" in response
    assert "2024-01-31 / 1024.00 kWh / 样本 24" in response
    assert "累计型指标按每个周期内末值减首值聚合。" in response
    assert "【关键指标】" not in response


def test_sensor_terminal_response_prefers_trend_decision_answer() -> None:
    orchestrator_module = importlib.import_module("src.agent.orchestrator")
    agent = orchestrator_module.LLMAgent(llm=None)

    response = agent._build_sensor_terminal_response(
        {
            "success": True,
            "total_count": 31,
            "analysis": {"mode": "single"},
            "focused_result": {
                "mode": "trend_decision",
                "metric": "用电量",
                "unit": "kWh",
                "direction": "down",
                "direction_label": "下降",
                "headline": "用电量整体呈下降趋势",
                "basis_label": "按天",
                "start_mean": 120.0,
                "end_mean": 90.0,
                "start_label": "2024-01-01",
                "end_label": "2024-01-31",
                "change_rate": -25.0,
                "aggregation_note": "累计型指标趋势判断按周期增量均值比较。",
            },
        },
        "fallback",
    )

    assert "【结果】" in response
    assert "用电量整体呈下降趋势" in response
    assert "【判断依据】" in response
    assert "- 判断粒度: 按天" in response
    assert "- 变化幅度: -25.00%" in response
    assert "累计型指标趋势判断按周期增量均值比较。" in response
    assert "【关键指标】" not in response


def test_sensor_terminal_response_prefers_anomaly_points_answer() -> None:
    orchestrator_module = importlib.import_module("src.agent.orchestrator")
    agent = orchestrator_module.LLMAgent(llm=None)

    response = agent._build_sensor_terminal_response(
        {
            "success": True,
            "total_count": 744,
            "analysis": {"mode": "single"},
            "focused_result": {
                "mode": "anomaly_points",
                "metric": "用电量",
                "unit": "kWh",
                "basis_label": "按天",
                "sample_count": 31,
                "anomaly_count": 2,
                "anomaly_ratio_pct": 6.45,
                "lower_bound": 80.0,
                "upper_bound": 120.0,
                "aggregation_note": "累计型指标按周期增量识别异常。",
                "rows": [
                    {"rank": 1, "time": "2024-01-15", "value": 158.0, "severity": 38.0},
                    {"rank": 2, "time": "2024-01-21", "value": 42.0, "severity": 38.0},
                ],
            },
        },
        "fallback",
    )

    assert "\u3010\u7ed3\u679c\u3011" in response
    assert "\u5f02\u5e38\u7528\u7535\u91cf\u65f6\u95f4\u70b9" in response
    assert "2024-01-15 / 158.00 kWh / \u504f\u79bb 38.00 kWh" in response
    assert "- \u5f02\u5e38\u5360\u6bd4: 6.45%" in response
    assert "\u7d2f\u8ba1\u578b\u6307\u6807\u6309\u5468\u671f\u589e\u91cf\u8bc6\u522b\u5f02\u5e38\u3002" in response
    assert "\u3010\u5173\u952e\u6307\u6807\u3011" not in response


def test_explicit_code_cross_project_collision_requires_clarification() -> None:
    orchestrator_module = importlib.import_module("src.agent.orchestrator")
    agent = orchestrator_module.LLMAgent(llm=None)

    class FakeItem:
        def __init__(self, payload):
            self.payload = payload

        @property
        def device(self):
            return self.payload["device"]

        def to_dict(self):
            return dict(self.payload)

    class FakeMetadataEngine:
        def list_all_devices(self):
            return [
                FakeItem({
                    "device": "a1_b9",
                    "name": "B2柜",
                    "project_id": "p1",
                    "project_name": "中国能建集团数据机房监控项目",
                    "project_code_name": "cneb-room",
                    "tg": "TG1",
                    "match_score": 0.0,
                    "matched_fields": ["device"],
                }),
                FakeItem({
                    "device": "a1_b9",
                    "name": "601-612",
                    "project_id": "p2",
                    "project_name": "平陆运河项目",
                    "project_code_name": "pluyh",
                    "tg": "TG2",
                    "match_score": 0.0,
                    "matched_fields": ["device"],
                }),
                FakeItem({
                    "device": "b1_b14",
                    "name": "电子楼 AA3-1 电源进线",
                    "project_id": "p3",
                    "project_name": "智慧物联网能效平台",
                    "project_code_name": "iot-energy",
                    "tg": "TG233",
                    "match_score": 0.0,
                    "matched_fields": ["device"],
                }),
            ]

    agent.metadata_engine = FakeMetadataEngine()
    plan = agent._get_query_plan("计算 a1_b9 和 b1_b14 在2024年1月的日均用电量对比")

    result = agent._resolve_devices_from_query_plan("计算 a1_b9 和 b1_b14 在2024年1月的日均用电量对比", plan)

    assert result["success"] is False
    assert result["needs_clarification"] is True
    assert result["clarification_candidates"][0]["keyword"] == "a1_b9"
    assert len(result["resolved_devices"]) == 1
    assert result["resolved_devices"][0]["device"] == "b1_b14"


def test_explicit_code_comparison_allows_multi_scope_only_with_explicit_aggregate() -> None:
    orchestrator_module = importlib.import_module("src.agent.orchestrator")
    agent = orchestrator_module.LLMAgent(llm=None)

    class FakeItem:
        def __init__(self, payload):
            self.payload = payload

        @property
        def device(self):
            return self.payload["device"]

        def to_dict(self):
            return dict(self.payload)

    class FakeMetadataEngine:
        def list_all_devices(self):
            return [
                FakeItem({
                    "device": "a1_b9",
                    "name": "B2柜",
                    "project_id": "p1",
                    "project_name": "中国能建集团数据机房监控项目",
                    "project_code_name": "cneb-room",
                    "tg": "TG1",
                    "match_score": 0.0,
                    "matched_fields": ["device"],
                }),
                FakeItem({
                    "device": "a1_b9",
                    "name": "601-612",
                    "project_id": "p2",
                    "project_name": "平陆运河项目",
                    "project_code_name": "pluyh",
                    "tg": "TG2",
                    "match_score": 0.0,
                    "matched_fields": ["device"],
                }),
                FakeItem({
                    "device": "b1_b14",
                    "name": "电子楼 AA3-1 电源进线",
                    "project_id": "p3",
                    "project_name": "智慧物联网能效平台",
                    "project_code_name": "iot-energy",
                    "tg": "TG233",
                    "match_score": 0.0,
                    "matched_fields": ["device"],
                }),
            ]

    agent.metadata_engine = FakeMetadataEngine()
    plan = agent._get_query_plan("汇总所有 a1_b9 与 b1_b14 哪个用电更多？")

    result = agent._resolve_devices_from_query_plan("汇总所有 a1_b9 与 b1_b14 哪个用电更多？", plan)

    assert result["success"] is True
    assert [item["device"] for item in result["resolved_devices"]].count("a1_b9") == 2
    assert any(item.get("project_id") == "p1" for item in result["resolved_devices"])
    assert any(item.get("project_id") == "p2" for item in result["resolved_devices"])
    assert any(item.get("device") == "b1_b14" for item in result["resolved_devices"])


def test_explicit_code_with_project_hint_auto_resolves_target_project() -> None:
    orchestrator_module = importlib.import_module("src.agent.orchestrator")
    agent = orchestrator_module.LLMAgent(llm=None)

    class FakeItem:
        def __init__(self, payload):
            self.payload = payload

        @property
        def device(self):
            return self.payload["device"]

        def to_dict(self):
            return dict(self.payload)

    class FakeMetadataEngine:
        def list_all_devices(self):
            return [
                FakeItem({
                    "device": "a1_b9",
                    "name": "B2柜",
                    "project_id": "p1",
                    "project_name": "中国能建集团数据机房监控项目",
                    "project_code_name": "cneb-room",
                    "tg": "TG1",
                    "match_score": 0.0,
                    "matched_fields": ["device"],
                }),
                FakeItem({
                    "device": "a1_b9",
                    "name": "B2柜",
                    "project_id": "p2",
                    "project_name": "IT资产管理系统",
                    "project_code_name": "itam",
                    "tg": "TG9",
                    "match_score": 0.0,
                    "matched_fields": ["device"],
                }),
                FakeItem({
                    "device": "b1_b14",
                    "name": "电子楼 AA3-1 电源进线",
                    "project_id": "p3",
                    "project_name": "智慧物联网能效平台",
                    "project_code_name": "iot-energy",
                    "tg": "TG233",
                    "match_score": 0.0,
                    "matched_fields": ["device"],
                }),
            ]

    agent.metadata_engine = FakeMetadataEngine()
    query = "中国能建集团数据机房监控项目a1_b9和 b1_b14 在2024年1月的日均用电量对比"
    plan = agent._get_query_plan(query)

    result = agent._resolve_devices_from_query_plan(query, plan)

    assert result["success"] is True
    assert result["device_codes"] == ["a1_b9", "b1_b14"]
    assert result["tg_values"] == ["TG1", "TG233"]


def test_fuzzy_multi_match_sensor_query_requires_clarification() -> None:
    orchestrator_module = importlib.import_module("src.agent.orchestrator")
    agent = orchestrator_module.LLMAgent(llm=None)

    agent._search_device_candidates = lambda keyword, user_query="": (
        [
            {"device": "a10_b1", "name": "1#变压器集中器 味多美", "project_id": "p1", "project_name": "智慧物联网能效平台", "match_score": 88.0},
            {"device": "a10_b2", "name": "1#变压器集中器 金顺星宾馆", "project_id": "p2", "project_name": "智慧物联网能效平台", "match_score": 88.0},
        ],
        {"target": keyword},
    )

    result = agent._action_search_devices({"keywords": ["变压器"], "user_query": "变压器今天的用电量", "comparison_mode": False})

    assert result["clarification_required"] is True
    assert result["clarification_candidates"][0]["keyword"] == "变压器"


def test_orchestrator_sensor_final_event_includes_table_preview(monkeypatch) -> None:
    orchestrator_module = importlib.import_module("src.agent.orchestrator")

    def fake_fetch_sensor_data_with_components(**kwargs):
        assert kwargs.get("output_format") == "json"
        return {
            "success": True,
            "data": '[{"logTime": "2024-01-01 00:00:00", "device": "a2_b1", "tag": "ua", "val": 230.5}]',
            "total_count": 864,
            "page": 1,
            "page_size": 50,
            "total_pages": 18,
            "has_more": True,
            "statistics": {"avg": 230.5},
            "analysis": {"mode": "single", "metric": "ua"},
            "chart_specs": [],
            "show_charts": False,
            "focused_table": {
                "headers": ["指标", "结果"],
                "rows": [{"指标": "A相电压 ua 平均值", "结果": "230.50 V"}],
                "total_count": 1,
                "has_more": False,
                "page_size": 1,
                "view_label": "问题直答",
            },
            "query_info": {"query": {}, "collections": ["source_data_ua_202401"]},
        }

    monkeypatch.setattr(orchestrator_module, "fetch_sensor_data_with_components", fake_fetch_sensor_data_with_components)

    agent = orchestrator_module.StreamingAgentOrchestrator(
        llm=object(),
        llm_non_streaming=object(),
        data_fetcher=object(),
        alias_memory={
            "a2_b1": {
                "device": "a2_b1",
                "name": "2#变压器 AA8-1 配电室照明",
                "project_id": "p1",
                "project_name": "智慧物联网能效平台",
                "tg": "TG8",
            }
        },
    )

    monkeypatch.setattr(
        agent,
        "_build_heuristic_action",
        lambda state, fast_path=False: {
            "thought": "use direct sensor query",
            "action": "get_sensor_data",
            "action_input": {
                "device_codes": ["a2_b1"],
                "data_type": "ua",
                "start_time": "2024-01-01",
                "end_time": "2024-01-01",
                "user_query": "a2_b1在2024年1月1日的ua是多少",
            },
        },
    )

    events = list(agent.run_with_progress("a2_b1在2024年1月1日的ua是多少"))
    final_event = next(event for event in events if event.get("type") == "final_answer")

    assert final_event["show_table"] is True
    assert final_event["table_type"] == "sensor_data"
    assert final_event["table_preview"] is not None
    assert final_event["table_preview"]["data"][0]["device"] == "a2_b1"
    assert final_event["table_preview"]["data"][0]["tag"] == "ua"
    assert final_event["table_preview"]["focused_table"]["view_label"] == "问题直答"
