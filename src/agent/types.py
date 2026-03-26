"""
DAG Orchestrator Types - 类型定义和常量

包含 GraphState 类型定义、ProgressEvent 数据类和节点常量。
这些定义被提取到单独的模块以避免循环导入。
"""

from typing import TypedDict, List, Dict, Optional, Any
from dataclasses import dataclass


class GraphState(TypedDict):
    """
    DAG 状态对象，在各节点间传递和累积数据
    
    字段说明：
    - user_query: 用户原始查询
    - intent: Intent Parser 解析的结构化意图
    - is_comparison: 是否为对比查询
    - comparison_targets: 对比查询的多个目标列表
    - device_codes: Metadata Mapper 查询到的设备代号列表
    - device_names: 设备代号到中文名称的映射
    - comparison_device_groups: 对比查询时，每个目标对应的设备组
    - collections: Sharding Router 计算的目标集合名称列表
    - data_tags: 数据类型标签列表
    - raw_data: Parallel Fetcher 获取的原始数据
    - statistics: 数据统计信息 (min, max, avg, count)
    - comparison_statistics: 对比查询时，每个目标的统计信息
    - query_info: 查询信息（MongoDB 查询详情）
    - final_response: Synthesizer 生成的最终响应
    - show_table: 是否显示表格
    - table_type: 表格类型
    - error: 错误信息
    - error_node: 发生错误的节点名称
    - history: 执行历史记录
    """
    # 输入
    user_query: str
    alias_memory: Optional[Dict[str, Any]]
    
    # Intent Parser 输出
    intent: Optional[Dict[str, Any]]
    query_plan: Optional[Dict[str, Any]]
    is_comparison: Optional[bool]  # 是否为对比查询
    comparison_targets: Optional[List[str]]  # 对比查询的多个目标
    
    # Metadata Mapper 输出
    device_codes: Optional[List[str]]
    device_names: Optional[Dict[str, str]]
    tg_values: Optional[List[str]]
    resolved_devices: Optional[List[Dict[str, Any]]]
    clarification_required: Optional[bool]
    clarification_candidates: Optional[List[Dict[str, Any]]]
    comparison_device_groups: Optional[Dict[str, List[str]]]  # ?? -> ??????
    comparison_scope_groups: Optional[Dict[str, List[Dict[str, Any]]]]
    
    collections: Optional[List[str]]
    data_tags: Optional[List[str]]
    
    # Parallel Fetcher 输出
    raw_data: Optional[List[Dict[str, Any]]]  # 保留字段但不传数据给 LLM
    total_count: Optional[int]  # 总数据量（用于前端分页）
    statistics: Optional[Dict[str, float]]
    comparison_statistics: Optional[Dict[str, Dict[str, float]]]  # 目标 -> 统计信息
    query_info: Optional[Dict[str, Any]]  # 包含 MongoDB 查询条件，前端可用于分页
    override_action: Optional[str]
    override_reason: Optional[str]
    override_terminal: Optional[bool]
    
    # Synthesizer 输出
    final_response: Optional[str]
    show_table: bool
    table_type: Optional[str]
    
    # 错误处理
    error: Optional[str]
    error_node: Optional[str]
    
    # 执行历史
    history: List[Dict[str, Any]]


@dataclass
class ProgressEvent:
    """流式输出的进度事件"""
    type: str  # step_start, step_done, final_answer
    step: Optional[str] = None
    info: Optional[str] = None
    response: Optional[str] = None
    show_table: bool = False
    table_type: Optional[str] = None
    query_params: Optional[Dict[str, Any]] = None
    projects: Optional[List[Dict[str, Any]]] = None
    project_stats: Optional[List[Dict[str, Any]]] = None
    duration_ms: Optional[float] = None
    timestamp_ms: Optional[float] = None
    total_duration_ms: Optional[float] = None


# 节点名称常量
NODE_INTENT_PARSER = "intent_parser"
NODE_ACTION_OVERRIDE_POLICY = "action_override_policy"
NODE_METADATA_MAPPER = "metadata_mapper"
NODE_SHARDING_ROUTER = "sharding_router"
NODE_PARALLEL_FETCHER = "parallel_fetcher"
NODE_SYNTHESIZER = "synthesizer"
