"""Chroma-backed entity resolver for device and project mentions."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import chromadb
except Exception:  # pragma: no cover - handled by graceful fallback
    chromadb = None

from src.metadata.metadata_engine import DeviceInfo, MetadataEngine
from src.semantic_layer.embedding import DashScopeEmbedding
from src.entity_resolver.types import EntityResolutionResult, ResolvedEntityCandidate


logger = logging.getLogger(__name__)

DEVICE_CODE_PATTERN = re.compile(r"[a-zA-Z]\d*_[a-zA-Z0-9_]+")


class ChromaEntityResolver:
    """Hybrid entity resolver using exact match + SQL lexical match + Chroma semantic search."""

    def __init__(
        self,
        metadata_engine: MetadataEngine,
        *,
        persist_directory: str = "data/chroma_entity_resolver",
        collection_name: str = "device_entities",
        embedding_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-v4",
        embedding_dimensions: int = 1024,
        embedding_base_url: Optional[str] = None,
        embedding_timeout: float = 20.0,
        semantic_top_k: int = 8,
        refresh_interval_seconds: int = 1800,
        enabled: bool = True,
    ):
        self.metadata_engine = metadata_engine
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_api_key = str(embedding_api_key or "").strip()
        self.embedding_model = embedding_model
        self.embedding_dimensions = int(embedding_dimensions or 1024)
        self.embedding_base_url = embedding_base_url
        self.embedding_timeout = float(embedding_timeout)
        self.embedding_batch_size = 10
        self.semantic_top_k = max(int(semantic_top_k or 8), 1)
        self.refresh_interval_seconds = max(int(refresh_interval_seconds or 1800), 60)
        self.enabled = bool(enabled)

        self._lock = threading.Lock()
        self._client = None
        self._collection = None
        self._embedding: Optional[DashScopeEmbedding] = None
        self._embedding_client_model: Optional[str] = None
        self._catalog_devices: List[DeviceInfo] = []
        self._catalog_by_device: Dict[str, List[DeviceInfo]] = {}
        self._last_refresh_at = 0.0
        self._semantic_available = False
        self._build_in_progress = False
        self._last_build_error: Optional[str] = None

    @property
    def manifest_path(self) -> Path:
        return self.persist_directory / "resolver_manifest.json"

    def is_ready(self) -> bool:
        return self.enabled and chromadb is not None and bool(self.embedding_api_key)

    def search_device_candidates(self, query: str, top_k: int = 10) -> EntityResolutionResult:
        query = str(query or "").strip()
        if not query:
            return EntityResolutionResult(candidates=[], query_info={"resolver": "chroma", "reason": "empty_query"})

        try:
            self._ensure_catalog_ready()
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("entity_resolver.catalog_init.failed error=%s", exc)

        exact_candidates = self._search_explicit_device_codes(query)
        lexical_candidates, lexical_sql = self._search_lexical(query)
        semantic_reason = self._decide_semantic_strategy(
            query,
            exact_candidates=exact_candidates,
            lexical_candidates=lexical_candidates,
            top_k=top_k,
        )
        semantic_candidates: List[ResolvedEntityCandidate] = []
        semantic_used = semantic_reason == "enabled"
        if semantic_used:
            semantic_candidates = self._search_semantic(query, top_k=max(top_k, self.semantic_top_k))

        merged = self._merge_candidates(exact_candidates, lexical_candidates, semantic_candidates)
        lexical_top_score = 0.0
        if lexical_candidates:
            lexical_top_score = float(lexical_candidates[0].lexical_score or lexical_candidates[0].match_score or 0.0)
        query_info = {
            "type": "entity_resolver",
            "resolver": "chroma",
            "vector_store": "chromadb" if self._semantic_available else "disabled",
            "semantic_enabled": self._semantic_available,
            "semantic_used": semantic_used,
            "semantic_strategy": semantic_reason,
            "semantic_building": self._build_in_progress,
            "semantic_build_error": self._last_build_error,
            "candidate_count": len(merged),
            "catalog_size": len(self._catalog_devices),
            "exact_candidate_count": len(exact_candidates),
            "lexical_candidate_count": len(lexical_candidates),
            "semantic_candidate_count": len(semantic_candidates),
            "lexical_top_score": round(lexical_top_score, 2),
            "lexical_sql": lexical_sql,
        }
        return EntityResolutionResult(candidates=merged[:top_k], query_info=query_info)

    def rebuild_index(self, force: bool = False) -> Dict[str, object]:
        self._ensure_catalog_ready(force=force)
        return {
            "resolver": "chroma",
            "semantic_enabled": self._semantic_available,
            "catalog_size": len(self._catalog_devices),
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory),
        }

    def _ensure_catalog_ready(self, force: bool = False) -> None:
        now = time.time()
        if not force and self._catalog_devices and now - self._last_refresh_at < self.refresh_interval_seconds:
            return

        with self._lock:
            now = time.time()
            if not force and self._catalog_devices and now - self._last_refresh_at < self.refresh_interval_seconds:
                return

            catalog = self.metadata_engine.list_all_devices()
            self._catalog_devices = catalog
            self._catalog_by_device = {}
            for device in catalog:
                device_code = self._normalize_code(getattr(device, "device", ""))
                if not device_code:
                    continue
                self._catalog_by_device.setdefault(device_code, []).append(device)
            self._last_refresh_at = now
            self._semantic_available = False

            if not self.enabled:
                logger.info("entity_resolver.disabled")
                return
            if chromadb is None:
                logger.warning("entity_resolver.chromadb_missing")
                return
            if not self.embedding_api_key:
                logger.warning("entity_resolver.embedding_key_missing")
                return

            signature = self._build_catalog_signature(catalog)
            manifest = self._load_manifest()
            count_matches = int(manifest.get("catalog_size") or 0) == len(catalog)
            signature_matches = manifest.get("catalog_signature") == signature

            collection = self._get_collection(reset=False)
            try:
                collection_count_matches = int(collection.count()) == len(catalog)
            except Exception:
                collection_count_matches = False

            needs_rebuild = force or not (count_matches and signature_matches and collection_count_matches)
            if needs_rebuild:
                if force:
                    collection = self._get_collection(reset=True)
                    self._rebuild_collection(collection, catalog)
                    self._save_manifest(signature=signature, catalog_size=len(catalog))
                    self._collection = collection
                    self._semantic_available = True
                    self._last_build_error = None
                else:
                    self._start_background_rebuild(catalog, signature)
                    return
            else:
                self._collection = collection
                self._semantic_available = True
                self._last_build_error = None

            logger.info(
                "entity_resolver.ready semantic=%s catalog_size=%s persist_dir=%s",
                self._semantic_available,
                len(catalog),
                self.persist_directory,
            )

    def _start_background_rebuild(self, catalog: Sequence[DeviceInfo], signature: str) -> None:
        if self._build_in_progress:
            return
        self._build_in_progress = True

        def _runner() -> None:
            try:
                collection = self._get_collection(reset=True)
                self._rebuild_collection(collection, catalog)
                self._save_manifest(signature=signature, catalog_size=len(catalog))
                self._collection = collection
                self._semantic_available = True
                self._last_build_error = None
                logger.info(
                    "entity_resolver.background_build_done catalog_size=%s persist_dir=%s",
                    len(catalog),
                    self.persist_directory,
                )
            except Exception as exc:  # pragma: no cover - background fallback
                self._semantic_available = False
                self._last_build_error = str(exc)
                logger.warning("entity_resolver.background_build_failed error=%s", exc)
            finally:
                self._build_in_progress = False

        thread = threading.Thread(target=_runner, name="chroma-entity-resolver-build", daemon=True)
        thread.start()
        logger.info(
            "entity_resolver.background_build_started catalog_size=%s persist_dir=%s",
            len(catalog),
            self.persist_directory,
        )

    def _get_collection(self, reset: bool = False):
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.persist_directory))
        if reset:
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._collection = None
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _rebuild_collection(self, collection, catalog: Sequence[DeviceInfo]) -> None:
        if not catalog:
            return
        ids = []
        metadatas = []
        documents = []
        for item in catalog:
            ids.append(str(item.device))
            metadatas.append(self._to_metadata(item))
            documents.append(self._to_document(item))

        embeddings = self._embed_texts(documents)
        batch_size = 64
        for index in range(0, len(ids), batch_size):
            end = index + batch_size
            collection.upsert(
                ids=ids[index:end],
                metadatas=metadatas[index:end],
                documents=documents[index:end],
                embeddings=embeddings[index:end],
            )

    def _search_explicit_device_codes(self, query: str) -> List[ResolvedEntityCandidate]:
        candidates: List[ResolvedEntityCandidate] = []
        seen = set()
        for match in DEVICE_CODE_PATTERN.findall(query):
            normalized_code = self._normalize_code(match)
            devices = self._catalog_by_device.get(normalized_code) or []
            if not devices:
                continue
            for device in devices:
                key = (str(device.device or ""), str(device.project_id or ""), str(device.name or ""))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    ResolvedEntityCandidate(
                        device=device.device,
                        name=device.name,
                        device_type=device.device_type,
                        project_id=device.project_id,
                        project_name=device.project_name,
                        project_code_name=device.project_code_name,
                        tg=device.tg,
                        match_score=140.0,
                        matched_fields=["device"],
                        match_reason="query_contains_device_code",
                        source="exact",
                        lexical_score=140.0,
                    )
                )
        return candidates

    def _search_lexical(self, query: str) -> Tuple[List[ResolvedEntityCandidate], str]:
        devices, sql = self.metadata_engine.search_devices(query)
        candidates = [
            ResolvedEntityCandidate(
                device=device.device,
                name=device.name,
                device_type=device.device_type,
                project_id=device.project_id,
                project_name=device.project_name,
                project_code_name=device.project_code_name,
                tg=device.tg,
                match_score=float(device.score or 0.0),
                matched_fields=list(device.matched_fields or []),
                match_reason=device.match_reason or "lexical_match",
                source="lexical",
                lexical_score=float(device.score or 0.0),
            )
            for device in devices
            if device.device
        ]
        return candidates, sql

    def _compact_query(self, query: str) -> str:
        return re.sub(r"\s+", "", self._normalize_text(query))

    def _is_short_keyword_query(self, query: str) -> bool:
        compact = self._compact_query(query)
        return 0 < len(compact) <= 8

    def _is_pure_contains_query(self, query: str) -> bool:
        compact = self._compact_query(query)
        if not compact:
            return False
        if DEVICE_CODE_PATTERN.fullmatch(compact):
            return True
        return not any(
            token in compact
            for token in ("多少", "什么", "哪些", "有没有", "怎么", "为何", "top", "排名", "对比")
        )

    def _has_strong_lexical_candidates(
        self,
        lexical_candidates: Sequence[ResolvedEntityCandidate],
        *,
        top_k: int,
    ) -> bool:
        if not lexical_candidates:
            return False
        top_score = float(lexical_candidates[0].lexical_score or lexical_candidates[0].match_score or 0.0)
        matched_fields = set(lexical_candidates[0].matched_fields or [])
        if top_score >= 96.0:
            return True
        if len(lexical_candidates) >= max(int(top_k or 1), 1) and top_score >= 88.0:
            return True
        return top_score >= 72.0 and bool(matched_fields & {"device", "name", "project_name", "project_code_name"})

    def _decide_semantic_strategy(
        self,
        query: str,
        *,
        exact_candidates: Sequence[ResolvedEntityCandidate],
        lexical_candidates: Sequence[ResolvedEntityCandidate],
        top_k: int,
    ) -> str:
        if not self._semantic_available or self._collection is None:
            return "semantic_unavailable"
        if exact_candidates:
            return "skip_exact_device_code"
        if self._is_short_keyword_query(query) and self._has_strong_lexical_candidates(lexical_candidates, top_k=top_k):
            return "skip_short_keyword_lexical_confident"
        if self._is_pure_contains_query(query) and self._has_strong_lexical_candidates(lexical_candidates, top_k=top_k):
            return "skip_contains_lexical_confident"
        return "enabled"

    def _search_semantic(self, query: str, top_k: int) -> List[ResolvedEntityCandidate]:
        if not self._semantic_available or self._collection is None:
            return []

        try:
            query_embedding = self._embed_text(query)
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=max(int(top_k or 1), 1),
                include=["metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning("entity_resolver.semantic_query.failed query=%s error=%s", query, exc)
            return []

        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        candidates: List[ResolvedEntityCandidate] = []
        for metadata, distance in zip(metadatas, distances):
            if not metadata:
                continue
            similarity = self._distance_to_similarity(distance)
            semantic_score = round(similarity * 100.0, 2)
            if semantic_score <= 0:
                continue
            matched_fields = ["semantic"]
            overlap_fields = self._estimate_overlap_fields(query, metadata)
            matched_fields.extend(field for field in overlap_fields if field not in matched_fields)
            candidates.append(
                ResolvedEntityCandidate(
                    device=str(metadata.get("device") or ""),
                    name=str(metadata.get("name") or ""),
                    device_type=str(metadata.get("device_type") or ""),
                    project_id=str(metadata.get("project_id") or ""),
                    project_name=metadata.get("project_name"),
                    project_code_name=metadata.get("project_code_name"),
                    tg=metadata.get("tg"),
                    match_score=semantic_score,
                    matched_fields=matched_fields,
                    match_reason=f"semantic_similarity={similarity:.3f}",
                    source="semantic",
                    semantic_score=semantic_score,
                )
            )
        return candidates

    def _merge_candidates(self, *candidate_groups: Sequence[ResolvedEntityCandidate]) -> List[ResolvedEntityCandidate]:
        merged: Dict[Tuple[str, str, str], ResolvedEntityCandidate] = {}
        for group in candidate_groups:
            for candidate in group:
                key = (candidate.device, candidate.project_id or "", candidate.name or "")
                existing = merged.get(key)
                if existing is None:
                    merged[key] = candidate
                    continue

                existing.match_score = max(float(existing.match_score or 0.0), float(candidate.match_score or 0.0))
                existing.lexical_score = self._coalesce_max(existing.lexical_score, candidate.lexical_score)
                existing.semantic_score = self._coalesce_max(existing.semantic_score, candidate.semantic_score)
                existing.source = self._merge_source(existing.source, candidate.source)
                existing.matched_fields = self._merge_unique(existing.matched_fields, candidate.matched_fields)
                existing.match_reason = self._merge_match_reason(existing.match_reason, candidate.match_reason)

        ranked = list(merged.values())
        for candidate in ranked:
            candidate.match_score = self._finalize_match_score(candidate)
        ranked.sort(
            key=lambda item: (
                -float(item.match_score or 0.0),
                -float(item.lexical_score or 0.0),
                -float(item.semantic_score or 0.0),
                str(item.device or ""),
                str(item.name or ""),
            )
        )
        return ranked

    def _finalize_match_score(self, candidate: ResolvedEntityCandidate) -> float:
        lexical = float(candidate.lexical_score or 0.0)
        semantic = float(candidate.semantic_score or 0.0)
        score = max(float(candidate.match_score or 0.0), lexical, semantic * 0.92)
        if candidate.source == "exact":
            score = max(score, 140.0)
        return round(score, 2)

    def _to_metadata(self, item: DeviceInfo) -> Dict[str, object]:
        return {
            "device": item.device,
            "name": item.name or "",
            "project_id": item.project_id or "",
            "project_name": item.project_name or "",
            "project_code_name": item.project_code_name or "",
            "device_type": item.device_type or "",
            "tg": item.tg or "",
        }

    def _to_document(self, item: DeviceInfo) -> str:
        return " | ".join(
            part
            for part in [
                f"device code: {item.device}",
                f"device name: {item.name or ''}",
                f"project name: {item.project_name or ''}",
                f"project code: {item.project_code_name or ''}",
                f"device type: {item.device_type or ''}",
                f"summary: {(item.project_name or '')} {(item.name or '')} {(item.device or '')}",
            ]
            if part.strip()
        )

    def _embed_text(self, text: str) -> List[float]:
        return self._run_with_embedding_fallback(lambda client: client.embed_text(text))

    def _embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        values = list(texts)
        if not values:
            return []

        vectors: List[List[float]] = []
        for index in range(0, len(values), self.embedding_batch_size):
            batch = values[index:index + self.embedding_batch_size]
            vectors.extend(self._run_with_embedding_fallback(lambda client, items=batch: client.embed_texts(items)))
        return vectors

    def _run_with_embedding_fallback(self, callback):
        models = [self.embedding_model]
        if self.embedding_model != "text-embedding-v4":
            models.append("text-embedding-v4")

        last_error = None
        for index, model_name in enumerate(models):
            try:
                client = self._get_embedding_client(model=model_name, refresh=index > 0)
                result = callback(client)
                if model_name != self.embedding_model:
                    logger.warning(
                        "entity_resolver.embedding_model_fallback from=%s to=%s",
                        self.embedding_model,
                        model_name,
                    )
                    self.embedding_model = model_name
                return result
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "entity_resolver.embedding_failed model=%s fallback=%s error=%s",
                    model_name,
                    index < len(models) - 1,
                    exc,
                )
        raise last_error

    def _get_embedding_client(self, model: Optional[str] = None, refresh: bool = False) -> DashScopeEmbedding:
        target_model = model or self.embedding_model
        if refresh or self._embedding is None or self._embedding_client_model != target_model:
            kwargs = {
                "api_key": self.embedding_api_key,
                "model": target_model,
                "dimensions": self.embedding_dimensions,
                "timeout": self.embedding_timeout,
            }
            if self.embedding_base_url:
                kwargs["base_url"] = self.embedding_base_url
            self._embedding = DashScopeEmbedding(**kwargs)
            self._embedding_client_model = target_model
        return self._embedding

    def _estimate_overlap_fields(self, query: str, metadata: Dict[str, object]) -> List[str]:
        normalized_query = self._normalize_text(query)
        fields = []
        for field_name in ["device", "name", "project_name", "project_code_name", "device_type"]:
            value = self._normalize_text(str(metadata.get(field_name) or ""))
            if not value:
                continue
            if value == normalized_query or value in normalized_query or normalized_query in value:
                fields.append(field_name)
        return fields

    def _build_catalog_signature(self, catalog: Sequence[DeviceInfo]) -> str:
        serialized = [
            "|".join(
                [
                    str(item.device or ""),
                    str(item.name or ""),
                    str(item.project_id or ""),
                    str(item.project_name or ""),
                    str(item.project_code_name or ""),
                    str(item.device_type or ""),
                ]
            )
            for item in sorted(catalog, key=lambda value: (value.device or "", value.project_id or "", value.name or ""))
        ]
        return hashlib.sha1("\n".join(serialized).encode("utf-8")).hexdigest()

    def _load_manifest(self) -> Dict[str, object]:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_manifest(self, *, signature: str, catalog_size: int) -> None:
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "catalog_signature": signature,
            "catalog_size": int(catalog_size),
            "updated_at": int(time.time()),
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
        }
        self.manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _distance_to_similarity(self, distance: object) -> float:
        try:
            numeric_distance = float(distance)
        except (TypeError, ValueError):
            return 0.0
        bounded = min(max(numeric_distance, 0.0), 2.0)
        return max(0.0, 1.0 - bounded / 2.0)

    def _normalize_code(self, value: str) -> str:
        return str(value or "").strip().lower()

    def _normalize_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip()).lower()

    def _merge_unique(self, left: Sequence[str], right: Sequence[str]) -> List[str]:
        values: List[str] = []
        seen = set()
        for item in list(left or []) + list(right or []):
            if not item or item in seen:
                continue
            seen.add(item)
            values.append(item)
        return values

    def _merge_match_reason(self, left: Optional[str], right: Optional[str]) -> Optional[str]:
        parts = []
        for item in [left, right]:
            normalized = str(item or "").strip()
            if normalized and normalized not in parts:
                parts.append(normalized)
        return " | ".join(parts) if parts else None

    def _merge_source(self, left: str, right: str) -> str:
        if left == right:
            return left
        values = [value for value in [left, right] if value]
        return "+".join(dict.fromkeys(values)) if values else "entity_resolver"

    def _coalesce_max(self, left: Optional[float], right: Optional[float]) -> Optional[float]:
        values = [value for value in [left, right] if value is not None]
        return max(values) if values else None
