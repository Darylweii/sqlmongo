"""Agent Orchestrator module - LLM 驱动的智能代理"""

from typing import Optional, Literal, Union

from langchain_core.language_models import BaseChatModel

from src.agent.orchestrator import (
    LLMAgent,
    AgentOrchestrator,
    StreamingAgentOrchestrator,
    AgentState,
    create_agent,
    create_agent_with_streaming,
)
from src.agent.dag_orchestrator import DAGOrchestrator
from src.agent.query_plan import QueryPlan
from src.agent.query_planner import LLMQueryPlanner


# Type alias for orchestrator type
OrchestratorType = Literal["react", "dag"]


def create_orchestrator(
    llm: BaseChatModel,
    metadata_engine,
    data_fetcher,
    cache_manager=None,
    compressor=None,
    orchestrator_type: OrchestratorType = "dag",
    coder_llm: Optional[BaseChatModel] = None
) -> Union[LLMAgent, DAGOrchestrator]:
    """
    编排器工厂函数 - 根据配置创建 ReAct 或 DAG 编排器
    
    支持通过 orchestrator_type 参数选择编排器类型：
    - "react": 使用传统 ReAct 架构（LLM 迭代决策）
    - "dag": 使用 LangGraph DAG 架构（确定性流程控制）
    
    多模型支持：
    - llm: 主模型，用于意图解析和总结呈现（推荐使用大模型如 qwen3-32b）
    - coder_llm: 代码模型，用于 SQL/代码生成（推荐使用代码模型如 qwen2.5-coder）
    
    Args:
        llm: LangChain LLM 实例（主模型）
        metadata_engine: 元数据引擎（MySQL 查询）
        data_fetcher: 数据获取器（MongoDB 查询）
        cache_manager: 缓存管理器（可选）
        compressor: 上下文压缩器（可选）
        orchestrator_type: 编排器类型，"react" 或 "dag"（默认 "dag"）
        coder_llm: 代码模型 LLM 实例（可选，用于 SQL/代码生成）
    
    Returns:
        LLMAgent 或 DAGOrchestrator 实例
    
    需求引用：
    - 需求 8.3: 支持通过配置切换新旧编排器
    
    Example:
        >>> from src.agent import create_orchestrator
        >>> # 单模型模式
        >>> orchestrator = create_orchestrator(
        ...     llm=llm,
        ...     metadata_engine=metadata_engine,
        ...     data_fetcher=data_fetcher,
        ...     orchestrator_type="dag"
        ... )
        >>> 
        >>> # 多模型模式
        >>> orchestrator = create_orchestrator(
        ...     llm=main_llm,           # qwen3-32b 用于自然语言
        ...     coder_llm=coder_llm,    # qwen2.5-coder 用于代码生成
        ...     metadata_engine=metadata_engine,
        ...     data_fetcher=data_fetcher,
        ...     orchestrator_type="dag"
        ... )
        >>> response = orchestrator.run("查询电梯用电量")
    """
    if orchestrator_type == "dag":
        return DAGOrchestrator(
            llm=llm,
            metadata_engine=metadata_engine,
            data_fetcher=data_fetcher,
            cache_manager=cache_manager,
            compressor=compressor,
            coder_llm=coder_llm,
        )
    elif orchestrator_type == "react":
        return create_agent(
            llm=llm,
            metadata_engine=metadata_engine,
            data_fetcher=data_fetcher,
            cache_manager=cache_manager,
            compressor=compressor,
        )
    else:
        raise ValueError(f"不支持的编排器类型: {orchestrator_type}，请使用 'react' 或 'dag'")


__all__ = [
    "LLMAgent",
    "AgentOrchestrator",
    "StreamingAgentOrchestrator",
    "AgentState",
    "create_agent",
    "create_agent_with_streaming",
    "DAGOrchestrator",
    "QueryPlan",
    "LLMQueryPlanner",
    "create_orchestrator",
    "OrchestratorType",
]
