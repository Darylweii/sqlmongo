"""Agent package lazy exports."""

from __future__ import annotations

from typing import Any, Literal, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from src.agent.dag_orchestrator import DAGOrchestrator
    from src.agent.orchestrator import AgentOrchestrator, AgentState, LLMAgent, StreamingAgentOrchestrator
    from src.agent.query_planner import LLMQueryPlanner
    from src.agent.query_plan import QueryPlan
else:
    BaseChatModel = Any  # type: ignore
    DAGOrchestrator = Any  # type: ignore
    LLMAgent = Any  # type: ignore
    AgentOrchestrator = Any  # type: ignore
    StreamingAgentOrchestrator = Any  # type: ignore
    AgentState = Any  # type: ignore
    LLMQueryPlanner = Any  # type: ignore
    QueryPlan = Any  # type: ignore


OrchestratorType = Literal["react", "dag"]


def create_orchestrator(
    llm: BaseChatModel,
    metadata_engine,
    data_fetcher,
    cache_manager=None,
    compressor=None,
    orchestrator_type: OrchestratorType = "dag",
    coder_llm: Optional[BaseChatModel] = None,
) -> Union[LLMAgent, DAGOrchestrator]:
    if orchestrator_type == "dag":
        from src.agent.dag_orchestrator import DAGOrchestrator

        return DAGOrchestrator(
            llm=llm,
            metadata_engine=metadata_engine,
            data_fetcher=data_fetcher,
            cache_manager=cache_manager,
            compressor=compressor,
            coder_llm=coder_llm,
        )
    if orchestrator_type == "react":
        from src.agent.orchestrator import create_agent

        return create_agent(
            llm=llm,
            metadata_engine=metadata_engine,
            data_fetcher=data_fetcher,
            cache_manager=cache_manager,
            compressor=compressor,
        )
    raise ValueError(f"不支持的编排器类型: {orchestrator_type}")


def __getattr__(name: str):
    if name in {"LLMAgent", "AgentOrchestrator", "StreamingAgentOrchestrator", "AgentState", "create_agent", "create_agent_with_streaming"}:
        from src.agent import orchestrator as orchestrator_module

        return getattr(orchestrator_module, name)
    if name == "DAGOrchestrator":
        from src.agent.dag_orchestrator import DAGOrchestrator

        return DAGOrchestrator
    if name == "QueryPlan":
        from src.agent.query_plan import QueryPlan

        return QueryPlan
    if name == "LLMQueryPlanner":
        from src.agent.query_planner import LLMQueryPlanner

        return LLMQueryPlanner
    raise AttributeError(name)


__all__ = [
    "LLMAgent",
    "AgentOrchestrator",
    "StreamingAgentOrchestrator",
    "AgentState",
    "create_orchestrator",
    "DAGOrchestrator",
    "QueryPlan",
    "LLMQueryPlanner",
    "OrchestratorType",
]
