"""Entity resolver package."""

from src.entity_resolver.chroma_resolver import ChromaEntityResolver
from src.entity_resolver.types import EntityResolutionResult, ResolvedEntityCandidate

__all__ = [
    "ChromaEntityResolver",
    "EntityResolutionResult",
    "ResolvedEntityCandidate",
]
