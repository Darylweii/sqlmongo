"""Shared types for entity resolution."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ResolvedEntityCandidate:
    """Normalized device candidate returned by the entity resolver."""

    device: str
    name: str = ""
    device_type: str = ""
    project_id: str = ""
    project_name: Optional[str] = None
    project_code_name: Optional[str] = None
    tg: Optional[str] = None
    match_score: float = 0.0
    matched_fields: List[str] = field(default_factory=list)
    match_reason: Optional[str] = None
    source: str = "entity_resolver"
    semantic_score: Optional[float] = None
    lexical_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "name": self.name,
            "device_type": self.device_type,
            "project_id": self.project_id,
            "project_name": self.project_name,
            "project_code_name": self.project_code_name,
            "tg": self.tg,
            "match_score": round(float(self.match_score or 0.0), 2),
            "matched_fields": list(self.matched_fields or []),
            "match_reason": self.match_reason,
            "source": self.source,
            "semantic_score": None if self.semantic_score is None else round(float(self.semantic_score), 4),
            "lexical_score": None if self.lexical_score is None else round(float(self.lexical_score), 4),
        }


@dataclass
class EntityResolutionResult:
    """Resolver output wrapper."""

    candidates: List[ResolvedEntityCandidate] = field(default_factory=list)
    query_info: Optional[Dict[str, Any]] = None

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [candidate.to_dict() for candidate in self.candidates]
