from src.entity_resolver.chroma_resolver import ChromaEntityResolver
from src.entity_resolver.types import ResolvedEntityCandidate


class _DummyMetadataEngine:
    pass


def _candidate(*, source: str = "lexical", score: float = 96.0, fields=None) -> ResolvedEntityCandidate:
    return ResolvedEntityCandidate(
        device="a1_b9",
        name="B2?",
        device_type="meter",
        project_id="p1",
        project_name="??????????????",
        project_code_name="ceec-dc",
        tg="TG232",
        match_score=score,
        lexical_score=score if source != "semantic" else None,
        semantic_score=score if source == "semantic" else None,
        matched_fields=list(fields or ["name"]),
        match_reason=source,
        source=source,
    )


def test_entity_resolver_skips_semantic_for_explicit_device_code() -> None:
    resolver = ChromaEntityResolver(_DummyMetadataEngine(), embedding_api_key="test-key")
    resolver._ensure_catalog_ready = lambda *args, **kwargs: None
    resolver._semantic_available = True
    resolver._collection = object()
    resolver._catalog_devices = []
    resolver._search_explicit_device_codes = lambda query: [_candidate(source="exact", score=140.0, fields=["device"])]
    resolver._search_lexical = lambda query: ([_candidate(source="lexical", score=126.0, fields=["device"])], "SELECT 1")
    semantic_calls = []
    resolver._search_semantic = lambda query, top_k: semantic_calls.append((query, top_k)) or []

    result = resolver.search_device_candidates("a1_b9", top_k=5)

    assert semantic_calls == []
    assert result.query_info["semantic_used"] is False
    assert result.query_info["semantic_strategy"] == "skip_exact_device_code"


def test_entity_resolver_uses_semantic_only_when_lexical_is_weak() -> None:
    resolver = ChromaEntityResolver(_DummyMetadataEngine(), embedding_api_key="test-key")
    resolver._ensure_catalog_ready = lambda *args, **kwargs: None
    resolver._semantic_available = True
    resolver._collection = object()
    resolver._catalog_devices = []
    resolver._search_explicit_device_codes = lambda query: []
    resolver._search_lexical = lambda query: ([_candidate(source="lexical", score=18.0, fields=["device_type"])], "SELECT 1")
    semantic_calls = []
    resolver._search_semantic = lambda query, top_k: semantic_calls.append((query, top_k)) or [_candidate(source="semantic", score=82.0, fields=["semantic"])]

    result = resolver.search_device_candidates("???", top_k=5)

    assert semantic_calls == [("???", 8)]
    assert result.query_info["semantic_used"] is True
    assert result.query_info["semantic_strategy"] == "enabled"
