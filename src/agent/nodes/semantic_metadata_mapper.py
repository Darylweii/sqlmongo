"""Compatibility wrapper for the hybrid metadata mapper."""

from __future__ import annotations

from typing import Optional

from langchain_core.language_models import BaseChatModel

from src.agent.nodes.metadata_mapper import MetadataMapperNode
from src.metadata.metadata_engine import MetadataEngine
from src.semantic_layer.device_search import DeviceSemanticSearch


class SemanticMetadataMapperNode(MetadataMapperNode):
    """Backward-compatible semantic mapper backed by the hybrid resolver.

    This class keeps the historical import path stable while delegating all
    device resolution behavior to ``MetadataMapperNode``:
    session memory -> explicit code handling -> lexical retrieval -> semantic
    fallback -> merged ranking -> recommendation/clarification.
    """

    def __init__(
        self,
        metadata_engine: MetadataEngine,
        coder_llm: Optional[BaseChatModel] = None,
        use_llm_sql: bool = False,
        device_search: Optional[DeviceSemanticSearch] = None,
        min_score: float = 0.35,
        top_k: int = 12,
    ) -> None:
        super().__init__(
            metadata_engine=metadata_engine,
            coder_llm=coder_llm,
            use_llm_sql=use_llm_sql,
            device_search=device_search,
            enable_semantic_fallback=True,
        )
        self.resolver.semantic_min_score = min_score
        self.resolver.semantic_top_k = top_k


__all__ = ["SemanticMetadataMapperNode"]
