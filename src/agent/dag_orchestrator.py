"""
DAG Orchestrator - 基于 LangGraph 的 DAG 状态机编排器

核心设计理念："AI 做大脑，代码做手脚"
- LLM 负责意图理解和结果呈现
- 确定性 Python 代码负责数据路由和获取

架构：
1. Intent Parser - LLM 意图解析
2. Metadata Mapper - Python/SQLAlchemy 设备查询
3. Sharding Router - Python 集合名称计算
4. Parallel Fetcher - asyncio/Motor 并行数据获取
5. Synthesizer - LLM 总结呈现
"""

from typing import Dict, Optional, Any, Generator, TYPE_CHECKING
import logging
import time

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from src.metadata.metadata_engine import MetadataEngine
from src.fetcher.data_fetcher import DataFetcher
from src.cache.cache_manager import CacheManager
from src.compressor.context_compressor import ContextCompressor
from src.semantic_layer.config import SemanticLayerConfig
from src.semantic_layer.semantic_layer import SemanticLayer, create_semantic_layer
from src.agent.types import (
    GraphState,
    ProgressEvent,
    NODE_INTENT_PARSER,
    NODE_ACTION_OVERRIDE_POLICY,
    NODE_METADATA_MAPPER,
    NODE_SHARDING_ROUTER,
    NODE_PARALLEL_FETCHER,
    NODE_SYNTHESIZER,
)
from src.agent.query_entities import extract_current_question_text
from src.agent.query_plan_state import (
    get_data_type_from_state,
    get_explicit_device_codes_from_state,
    get_query_mode_from_state,
    get_time_range_from_state,
    has_detect_data_types_intent_from_state,
    has_device_listing_intent_from_state,
    has_project_listing_intent_from_state,
    has_project_stats_intent_from_state,
    has_sensor_query_intent_from_state,
)


logger = logging.getLogger(__name__)


# Re-export types for backward compatibility
__all__ = [
    "GraphState",
    "ProgressEvent",
    "NODE_INTENT_PARSER",
    "NODE_ACTION_OVERRIDE_POLICY",
    "NODE_METADATA_MAPPER",
    "NODE_SHARDING_ROUTER",
    "NODE_PARALLEL_FETCHER",
    "NODE_SYNTHESIZER",
    "should_continue",
    "route_after_query_plan",
    "route_after_action_override",
    "route_after_metadata",
    "DAGOrchestrator",
    "SemanticLayerConfig",
]


def should_continue(state: GraphState) -> str:
    """
    条件边函数：决定下一个节点
    
    如果有错误，直接跳转到 Synthesizer 生成友好提示
    否则继续正常流程
    
    Args:
        state: 当前图状态
    
    Returns:
        "continue" - 继续正常流程到下一个节点
        NODE_SYNTHESIZER - 跳转到 Synthesizer 处理错误
    
    需求引用：
    - 需求 1.4: 支持条件边，允许在特定条件下跳过某些节点或提前终止
    - 需求 1.5: 任意节点执行失败时，将错误信息传递给 Synthesizer
    """
    if state.get("error"):
        logger.debug(f"检测到错误，跳转到 Synthesizer: {state.get('error')}")
        return NODE_SYNTHESIZER
    return "continue"


def route_after_action_override(state: GraphState) -> str:
    """Route after the shared action override policy has been applied."""
    if state.get("error"):
        return "error"
    if state.get("override_terminal"):
        return "terminal"
    if state.get("override_action") == "get_sensor_data":
        return "sensor"
    return "continue"


def route_after_metadata(state: GraphState) -> str:
    """Route after metadata resolution.

    设备列表查询在 metadata 节点就已经拿到了完整结果，不应继续进入分片与数据库查询。
    """
    if state.get("error"):
        return "error"
    if state.get("clarification_required"):
        return "terminal"
    if has_device_listing_intent_from_state(state) and state.get("table_type") == "devices":
        return "terminal"
    return "continue"


def _build_table_preview_from_state(state: GraphState) -> Optional[Dict[str, Any]]:
    raw_data = state.get("raw_data") or []
    if not isinstance(raw_data, list) or not raw_data:
        return None

    preview_limit = 50
    rows = []
    for item in raw_data[:preview_limit]:
        if isinstance(item, dict):
            rows.append({
                "time": item.get("logTime") or item.get("time", ""),
                "device": item.get("device", "-"),
                "tag": item.get("tag", "-"),
                "value": item.get("val") if item.get("val") is not None else item.get("value", ""),
            })
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            rows.append({
                "time": item[0],
                "device": "-",
                "tag": "-",
                "value": item[1],
            })

    if not rows:
        return None

    focused_table = None
    focused_result = state.get("focused_result")
    if isinstance(focused_result, dict) and isinstance(focused_result.get("table"), dict):
        focused_table = focused_result.get("table")

    total_count = int(state.get("total_count") or len(raw_data) or len(rows))
    return {
        "success": True,
        "data": rows,
        "total_count": total_count,
        "page": 1,
        "page_size": min(preview_limit, len(rows)),
        "total_pages": max(1, (total_count + 49) // 50),
        "has_more": total_count > len(rows),
        "statistics": state.get("statistics"),
        "focused_table": focused_table,
    }

def _build_frontend_query_params_from_state(state: GraphState) -> Optional[Dict[str, Any]]:
    device_codes = [str(code).strip() for code in (state.get("device_codes") or []) if str(code).strip()]
    time_start, time_end = get_time_range_from_state(state)
    if not device_codes or not time_start or not time_end:
        return None

    tg_values = [str(value).strip() for value in (state.get("tg_values") or []) if str(value).strip()]
    query_plan = state.get("query_plan") if isinstance(state.get("query_plan"), dict) else None
    comparison_scope_groups = state.get("comparison_scope_groups") if isinstance(state.get("comparison_scope_groups"), dict) else None
    current_question = ""
    if isinstance(query_plan, dict):
        current_question = str(query_plan.get("current_question") or "").strip()
    if not current_question:
        current_question = extract_current_question_text(str(state.get("user_query") or ""))
    return {
        "device_codes": device_codes,
        "tg_values": tg_values,
        "comparison_scope_groups": comparison_scope_groups,
        "start_time": time_start,
        "end_time": time_end,
        "data_type": get_data_type_from_state(state, default="ep"),
        "page": 1,
        "page_size": 50,
        "user_query": current_question,
        "query_plan": query_plan,
    }


def route_after_query_plan(state: GraphState) -> str:
    """Choose the next DAG branch directly from QueryPlan-oriented state."""
    if state.get("error"):
        return "error"

    query_mode = get_query_mode_from_state(state)
    explicit_device_codes = get_explicit_device_codes_from_state(state)

    if (
        has_project_listing_intent_from_state(state)
        or has_project_stats_intent_from_state(state)
        or has_detect_data_types_intent_from_state(state)
    ):
        return "action_override"

    if has_sensor_query_intent_from_state(state) or has_device_listing_intent_from_state(state):
        return "metadata"

    if query_mode == "general":
        return "synthesizer"

    return "metadata"


# Import nodes after should_continue to avoid circular imports
from src.agent.nodes.intent_parser import IntentParserNode
from src.agent.nodes.action_override_policy_node import ActionOverridePolicyNode
from src.agent.nodes.metadata_mapper import MetadataMapperNode
from src.agent.nodes.semantic_metadata_mapper import SemanticMetadataMapperNode
from src.agent.nodes.sharding_router import ShardingRouterNode
from src.agent.nodes.parallel_fetcher import ParallelFetcherNode
from src.agent.nodes.synthesizer import SynthesizerNode


class DAGOrchestrator:
    """
    基于 LangGraph 的 DAG 编排器
    
    采用 5 节点 DAG 架构：
    Intent Parser → Metadata Mapper → Sharding Router → Parallel Fetcher → Synthesizer
    
    特点：
    - 确定性路由：设备映射和集合计算由 Python 代码完成
    - 高性能：利用 asyncio 并行获取数据
    - 可观测性：流式输出执行进度
    - 可复用性：最大化复用现有组件
    - 多模型支持：不同节点可使用不同 LLM（如代码模型、自然语言模型）
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        metadata_engine: MetadataEngine,
        data_fetcher: DataFetcher,
        cache_manager: Optional[CacheManager] = None,
        compressor: Optional[ContextCompressor] = None,
        max_collections: int = 50,
        coder_llm: Optional[BaseChatModel] = None,
        semantic_layer_config: Optional[SemanticLayerConfig] = None,
        use_semantic_search: bool = False,
        alias_memory: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化 DAG 编排器
        
        Args:
            llm: 主 LLM 实例（用于意图解析和总结呈现）
            metadata_engine: 元数据引擎（MySQL 查询）
            data_fetcher: 数据获取器（MongoDB 查询）
            cache_manager: 缓存管理器（可选）
            compressor: 上下文压缩器（可选）
            max_collections: 最大允许的集合数量阈值（默认 50）
            coder_llm: 代码模型 LLM 实例（可选，用于 SQL/代码生成）
            semantic_layer_config: 语义层配置（可选，为 None 时不使用语义层）
            use_semantic_search: 是否使用语义搜索进行设备匹配（默认 False）
        
        需求引用:
            - 需求 5.1: 在初始化时创建语义层实例（如果启用）
            - 需求 5.5: 不影响现有 DAG 节点的执行顺序和数据流
        """
        self.llm = llm
        self.coder_llm = coder_llm or llm  # 如果没有指定代码模型，使用主模型
        self.metadata_engine = metadata_engine
        self.data_fetcher = data_fetcher
        self.cache_manager = cache_manager
        self.compressor = compressor
        self.max_collections = max_collections
        self.use_semantic_search = use_semantic_search
        self.alias_memory = dict(alias_memory or {})
        
        # 初始化语义层（如果启用）
        # Requirements 5.1: 在初始化时创建语义层实例
        self._semantic_layer: Optional[SemanticLayer] = None
        if semantic_layer_config is not None:
            self._semantic_layer = create_semantic_layer(
                config=semantic_layer_config,
                auto_initialize=True,
            )
            if self._semantic_layer:
                logger.info("语义层已启用并初始化成功")
            else:
                logger.warning("语义层配置已提供但初始化失败，将使用传统模式")
        
        # 初始化节点实例
        # Intent Parser 使用主模型，并传递语义层实例
        # Requirements 5.5: 不影响现有执行顺序，仅增强 IntentParser 能力
        self._intent_parser = IntentParserNode(
            llm=llm,
            semantic_layer=self._semantic_layer,
        )
        self._synthesizer = SynthesizerNode(llm)
        self._action_override_policy = ActionOverridePolicyNode(metadata_engine)
        
        # MetadataMapper: 根据配置选择语义搜索或传统模式
        if use_semantic_search:
            logger.info("使用语义搜索进行设备匹配")
            self._metadata_mapper = SemanticMetadataMapperNode(
                metadata_engine=metadata_engine,
            )
        else:
            # 传统模式：使用 LIKE 查询
            use_llm_sql = coder_llm is not None
            self._metadata_mapper = MetadataMapperNode(
                metadata_engine, 
                coder_llm=coder_llm,
                use_llm_sql=use_llm_sql
            )
        
        # 纯 Python 节点（不需要 LLM）
        self._sharding_router = ShardingRouterNode(max_collections)
        self._parallel_fetcher = ParallelFetcherNode(data_fetcher)
        
        self.graph = self._build_graph()
    
    @property
    def semantic_layer_enabled(self) -> bool:
        """
        检查语义层是否已启用并初始化成功
        
        Returns:
            True 如果语义层已启用且初始化成功，否则 False
        """
        return self._semantic_layer is not None and self._semantic_layer.is_initialized
    
    @property
    def semantic_layer(self) -> Optional[SemanticLayer]:
        """
        获取语义层实例（可用于诊断或高级用途）
        
        Returns:
            SemanticLayer 实例，如果未启用则返回 None
        """
        return self._semantic_layer
    
    def close(self) -> None:
        """
        关闭编排器并释放资源
        
        清理语义层资源，停止后台线程。
        应在销毁 DAGOrchestrator 实例前调用。
        """
        if self._semantic_layer:
            logger.info("关闭语义层资源")
            self._semantic_layer.close()
            self._semantic_layer = None
    
    def __enter__(self) -> "DAGOrchestrator":
        """支持上下文管理器协议"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """支持上下文管理器协议，自动清理资源"""
        self.close()

    
    def _build_graph(self) -> StateGraph:
        """
        构建 LangGraph 状态图
        
        定义 5 个核心节点和条件边：
        1. intent_parser - 意图解析（LLM）
        2. metadata_mapper - 元数据映射（Python）
        3. sharding_router - 分片路由（Python）
        4. parallel_fetcher - 并行获取（asyncio）
        5. synthesizer - 总结呈现（LLM）
        
        条件边：任意节点出错时跳转到 synthesizer
        """
        # 创建状态图
        workflow = StateGraph(GraphState)
        
        # 添加节点（使用占位函数，后续任务会实现具体节点类）
        workflow.add_node(NODE_INTENT_PARSER, self._intent_parser_node)
        workflow.add_node(NODE_ACTION_OVERRIDE_POLICY, self._action_override_policy_node)
        workflow.add_node(NODE_METADATA_MAPPER, self._metadata_mapper_node)
        workflow.add_node(NODE_SHARDING_ROUTER, self._sharding_router_node)
        workflow.add_node(NODE_PARALLEL_FETCHER, self._parallel_fetcher_node)
        workflow.add_node(NODE_SYNTHESIZER, self._synthesizer_node)
        
        workflow.add_edge(START, NODE_INTENT_PARSER)

        # QueryPlan ?????? QueryPlan?????????????????????
        workflow.add_conditional_edges(
            NODE_INTENT_PARSER,
            route_after_query_plan,
            {
                "action_override": NODE_ACTION_OVERRIDE_POLICY,
                "metadata": NODE_METADATA_MAPPER,
                "synthesizer": NODE_SYNTHESIZER,
                "error": NODE_SYNTHESIZER,
            }
        )
        
        # ?????????Action Override -> Metadata Mapper / Sharding Router / END / Synthesizer
        workflow.add_conditional_edges(
            NODE_ACTION_OVERRIDE_POLICY,
            route_after_action_override,
            {
                "continue": NODE_METADATA_MAPPER,
                "sensor": NODE_SHARDING_ROUTER,
                "terminal": END,
                "error": NODE_SYNTHESIZER,
            }
        )
        
        # ?????????Metadata Mapper -> Sharding Router ??Synthesizer????????
        workflow.add_conditional_edges(
            NODE_METADATA_MAPPER,
            route_after_metadata,
            {
                "continue": NODE_SHARDING_ROUTER,
                "terminal": END,
                "error": NODE_SYNTHESIZER,
            }
        )
        
        # 添加条件边：Sharding Router -> Parallel Fetcher 或 Synthesizer（出错时）
        workflow.add_conditional_edges(
            NODE_SHARDING_ROUTER,
            should_continue,
            {
                "continue": NODE_PARALLEL_FETCHER,
                NODE_SYNTHESIZER: NODE_SYNTHESIZER
            }
        )
        
        # 添加条件边：Parallel Fetcher -> Synthesizer
        workflow.add_conditional_edges(
            NODE_PARALLEL_FETCHER,
            should_continue,
            {
                "continue": NODE_SYNTHESIZER,
                NODE_SYNTHESIZER: NODE_SYNTHESIZER
            }
        )
        
        # Synthesizer 是终点
        workflow.add_edge(NODE_SYNTHESIZER, END)
        
        return workflow
    
    def _run_timed_node(self, node_name: str, node_callable, state: GraphState) -> GraphState:
        """Execute a DAG node and attach duration metadata to its latest history entry."""
        start_time = time.perf_counter()
        output_state = node_callable(state)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        history = list(output_state.get("history", []))
        if history:
            last_entry = dict(history[-1])
            if last_entry.get("node") == node_name:
                last_entry["duration_ms"] = duration_ms
                history[-1] = last_entry
                output_state = {**output_state, "history": history}

        return output_state

    def _intent_parser_node(self, state: GraphState) -> GraphState:
        """
        意图解析节点 - 使用 IntentParserNode 实现
        
        调用 LLM 从用户查询中提取结构化意图
        """
        return self._run_timed_node(NODE_INTENT_PARSER, self._intent_parser, state)
    
    def _action_override_policy_node(self, state: GraphState) -> GraphState:
        """Apply the shared action override policy before metadata resolution."""
        return self._run_timed_node(NODE_ACTION_OVERRIDE_POLICY, self._action_override_policy, state)
    
    def _metadata_mapper_node(self, state: GraphState) -> GraphState:
        """
        元数据映射节点 - 使用 MetadataMapperNode 实现
        
        查询 MySQL 数据库获取设备代号
        """
        return self._run_timed_node(NODE_METADATA_MAPPER, self._metadata_mapper, state)
    
    def _sharding_router_node(self, state: GraphState) -> GraphState:
        """
        分片路由节点 - 使用 ShardingRouterNode 实现
        
        计算目标 MongoDB 集合名称
        """
        return self._run_timed_node(NODE_SHARDING_ROUTER, self._sharding_router, state)
    
    def _parallel_fetcher_node(self, state: GraphState) -> GraphState:
        """
        并行获取节点 - 使用 ParallelFetcherNode 实现
        
        并行查询 MongoDB 获取数据
        """
        return self._run_timed_node(NODE_PARALLEL_FETCHER, self._parallel_fetcher, state)
    
    def _synthesizer_node(self, state: GraphState) -> GraphState:
        """
        总结呈现节点 - 使用 SynthesizerNode 实现
        
        使用 LLM 生成人类可读的响应
        """
        return self._run_timed_node(NODE_SYNTHESIZER, self._synthesizer, state)
    
    def _init_state(self, user_query: str) -> GraphState:
        """初始化 GraphState"""
        current_question = extract_current_question_text(str(user_query or ""))
        return GraphState(
            user_query=current_question,
            alias_memory=dict(self.alias_memory),
            intent=None,
            query_plan=None,
            is_comparison=False,
            comparison_targets=None,
            device_codes=None,
            device_names=None,
            tg_values=None,
            resolved_devices=None,
            clarification_required=False,
            clarification_candidates=None,
            comparison_device_groups=None,
            comparison_scope_groups=None,
            collections=None,
            data_tags=None,
            raw_data=None,
            total_count=None,
            statistics=None,
            comparison_statistics=None,
            query_info=None,
            override_action=None,
            override_reason=None,
            override_terminal=False,
            final_response=None,
            show_table=False,
            table_type=None,
            error=None,
            error_node=None,
            history=[]
        )
    
    def run(self, user_query: str) -> str:
        """
        同步执行查询
        
        初始化 GraphState，编译并执行图，返回最终响应。
        
        Args:
            user_query: 用户查询字符串
        
        Returns:
            最终响应字符串
        
        需求引用：
        - 需求 7.5: 支持同步执行模式
        """
        logger.info(f"开始执行查询: {user_query[:50]}...")
        
        # 初始化状态
        initial_state = self._init_state(user_query)
        
        try:
            # 编译并执行图
            compiled_graph = self.graph.compile()
            final_state = compiled_graph.invoke(initial_state)
            
            # 记录执行历史
            history = final_state.get("history", [])
            logger.info(f"查询完成，执行了 {len(history)} 个节点")
            
            return final_state.get("final_response", "查询完成。")
            
        except Exception as e:
            logger.exception(f"DAG 执行异常: {e}")
            return f"处理您的请求时遇到问题：{str(e)}"
    
    def run_with_progress(self, user_query: str) -> Generator[Dict[str, Any], None, None]:
        """
        带进度的流式执行
        
        在每个节点执行前后发送进度事件，使用 Generator 返回进度更新。
        
        Args:
            user_query: 用户查询字符串
        
        Yields:
            进度事件字典，包含以下类型：
            - step_start: 节点开始执行
            - step_done: 节点执行完成
            - final_answer: 最终响应
        
        需求引用：
        - 需求 7.1: 支持流式输出
        - 需求 7.3: 节点开始执行时发送 step_start 事件
        - 需求 7.4: 节点执行完成时发送 step_done 事件
        """
        logger.info(f"开始流式执行查询: {user_query[:50]}...")
        run_start_time = time.perf_counter()
        
        # 初始化状态
        initial_state = self._init_state(user_query)
        
        # 编译图
        compiled_graph = self.graph.compile()
        
        # 节点名称到中文的映射
        node_names = {
            NODE_INTENT_PARSER: "解析 QueryPlan",
            NODE_ACTION_OVERRIDE_POLICY: "执行查询计划",
            NODE_METADATA_MAPPER: "解析设备范围",
            NODE_SHARDING_ROUTER: "计算数据分片",
            NODE_PARALLEL_FETCHER: "执行数据库查询",
            NODE_SYNTHESIZER: "生成响应",
        }

        current_state = initial_state
        
        try:
            # 使用 stream 方法获取每个节点的执行结果
            for output in compiled_graph.stream(initial_state):
                # output 是一个字典，key 是节点名称，value 是该节点的输出状态
                for node_name, node_output in output.items():
                    # 发送节点开始事件
                    yield {
                        "type": "step_start",
                        "node_name": node_name,
                        "step": node_names.get(node_name, node_name),
                        "timestamp_ms": round(time.time() * 1000, 2)
                    }
                    
                    # 更新当前状态
                    current_state = {**current_state, **node_output}
                    
                    # 获取节点执行结果摘要
                    history = current_state.get("history", [])
                    last_entry = history[-1] if history else {}
                    info = last_entry.get("result", "")
                    
                    # 发送节点完成事件
                    yield {
                        "type": "step_done",
                        "node_name": node_name,
                        "step": node_names.get(node_name, node_name),
                        "info": info,
                        "duration_ms": last_entry.get("duration_ms")
                    }
            
            # 发送最终响应
            query_info = current_state.get("query_info")
            frontend_query_params = _build_frontend_query_params_from_state(current_state)
            final_event = {
                "type": "final_answer",
                "response": current_state.get("final_response", "Query completed."),
                "show_table": current_state.get("show_table", False),
                "table_type": current_state.get("table_type"),
                "query_params": frontend_query_params,
                "query_plan": current_state.get("query_plan"),
                "clarification_required": bool(current_state.get("clarification_required", False)),
                "clarification_candidates": current_state.get("clarification_candidates"),
                "table_preview": _build_table_preview_from_state(current_state),
                "total_duration_ms": round((time.perf_counter() - run_start_time) * 1000, 2)
            }
            if isinstance(query_info, dict):
                if current_state.get("table_type") == "projects":
                    final_event["projects"] = query_info.get("projects")
                elif current_state.get("table_type") == "project_stats":
                    final_event["project_stats"] = query_info.get("stats")
                elif current_state.get("table_type") == "devices":
                    final_event["devices"] = query_info.get("devices")
            yield final_event
            
            logger.info(f"流式执行完成，执行了 {len(current_state.get('history', []))} 个节点")
            
        except Exception as e:
            logger.exception(f"流式执行异常: {e}")
            # 发送错误响应
            yield {
                "type": "final_answer",
                "response": f"处理您的请求时遇到问题：{str(e)}",
                "show_table": False,
                "table_type": None,
                "query_params": None,
                "query_plan": current_state.get("query_plan"),
                "total_duration_ms": round((time.perf_counter() - run_start_time) * 1000, 2)
            }
