"""
AI Data Router Agent - Web API
FastAPI 后端服务
"""
import ast
import asyncio
import os
import re
import json
import logging
from functools import lru_cache
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from src.config import load_config
from src.metadata.metadata_engine import MetadataEngine
from src.fetcher.data_fetcher import DataFetcher
from src.analysis import InsightEngine
from src.compressor.context_compressor import ContextCompressor
from src.agent import DAGOrchestrator
from src.agent.orchestrator import create_agent_with_streaming
from src.agent.query_entities import allows_explicit_multi_scope_aggregation
from src.entity_resolver import ChromaEntityResolver
from src.tools.sensor_tool import fetch_sensor_data_with_components
from src.router.collection_router import get_collection_prefix, get_target_collections, get_data_tags
from pymongo import MongoClient
import httpx
from langchain_openai import ChatOpenAI
from src.semantic_rules import SemanticRuleStore
from src.user_memory_store import UserMemoryStore
from src.chat_memory_service import ChatMemoryService
from src.memory_rewrite import parse_memory_rewrite_json
from src.version_info import APP_VERSION, build_version_payload

logger = logging.getLogger(__name__)

CHAT_HISTORY_LIMIT = 10
SESSION_ALIAS_LIMIT = 50
SESSION_STATE: Dict[str, Dict[str, Any]] = {}
ALIAS_CONFIRM_PATTERNS = [
    re.compile(r'(?:\u6211\u8bf4\u7684|\u6211\u6307\u7684|\u8bf4\u7684)(?P<alias>[^\n\r\uff0c\u3002,.;?]{1,32}?)(?:\u662f|\u6307\u7684\u662f)\s*(?P<device>[a-zA-Z]\d*_[a-zA-Z0-9_]+)', re.IGNORECASE),
    re.compile(r'(?P<alias>[^\n\r\uff0c\u3002,.;?]{1,32}?)(?:\u662f|\u6307\u7684\u662f)\s*(?P<device>[a-zA-Z]\d*_[a-zA-Z0-9_]+)', re.IGNORECASE),
]
DEVICE_CODE_PATTERN = re.compile(r"[a-zA-Z]\d*_[a-zA-Z0-9_]+")
EMPTY_RESULT_MESSAGE = "当前时间范围内未查询到符合条件的数据，请尝试放宽时间范围、检查设备代号或调整过滤条件。"


ANSWER_STREAM_CHUNK_SIZE = max(6, int(os.getenv("ANSWER_STREAM_CHUNK_SIZE", "12")))
ANSWER_STREAM_CHUNK_DELAY_MS = max(0, int(os.getenv("ANSWER_STREAM_CHUNK_DELAY_MS", "16")))
ANSWER_STREAM_PUNCTUATION = set("\uFF0C\u3002\uFF1F\uFF01\uFF1B\uFF1A,.!?;:\n")
RESOLVED_SCOPE_CARD_ITEM_LIMIT = max(1, int(os.getenv("RESOLVED_SCOPE_CARD_ITEM_LIMIT", "20")))


def _iter_answer_delta_chunks(text: str, chunk_size: int = ANSWER_STREAM_CHUNK_SIZE):
    value = str(text or "")
    if not value:
        return
    length = len(value)
    start = 0
    while start < length:
        end = min(length, start + chunk_size)
        if end < length:
            lookahead_end = min(length, end + 8)
            for index in range(end, lookahead_end):
                if value[index] in ANSWER_STREAM_PUNCTUATION:
                    end = index + 1
                    break
        chunk = value[start:end]
        if chunk:
            yield chunk
        start = end


def _short_text(text: str, limit: int = 120) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= limit:
        return value
    return value[: max(limit - 3, 0)] + "..."


def _sample_items(values, limit: int = 5):
    if not values:
        return []
    return list(values)[:limit]


def _get_or_create_session_id(session_id: Optional[str]) -> str:
    value = str(session_id or "").strip()
    return value or uuid4().hex


def _normalize_alias_key(alias: str) -> str:
    return re.sub(r"\s+", " ", str(alias or "").strip()).lower()


def _get_session_state(session_id: str) -> Dict[str, Any]:
    state = SESSION_STATE.setdefault(session_id, {"device_aliases": {}, "updated_at": None, "last_user_query": None})
    state.setdefault("device_aliases", {})
    state.setdefault("last_user_query", None)
    state["updated_at"] = datetime.now().isoformat()
    return state


def _clear_session_alias(session_id: str, alias: Optional[str]) -> bool:
    normalized_alias = _normalize_alias_key(alias or "")
    if not session_id or not normalized_alias:
        return False
    state = _get_session_state(session_id)
    alias_map = state.setdefault("device_aliases", {})
    removed = alias_map.pop(normalized_alias, None)
    return removed is not None


def _clear_session_scope(session_id: str) -> int:
    if not session_id:
        return 0
    state = _get_session_state(session_id)
    alias_map = state.setdefault("device_aliases", {})
    count = len(alias_map)
    alias_map.clear()
    return count


SCOPE_CLEAR_ALL_PATTERNS = [
    re.compile(r"^(?:请|麻烦)?\s*(?:清除|清空|重置)当前确认范围$", re.IGNORECASE),
    re.compile(r"^(?:请|麻烦)?\s*(?:清除|清空|重置)当前作用域$", re.IGNORECASE),
]
SCOPE_RESET_ALIAS_PATTERNS = [
    re.compile(r"^(?:请|麻烦)?\s*重新确认\s*(?P<alias>[A-Za-z]\d*_[A-Za-z0-9_]+)\s*$", re.IGNORECASE),
    re.compile(r"^(?:请|麻烦)?\s*重新选择\s*(?P<alias>[A-Za-z]\d*_[A-Za-z0-9_]+)\s*$", re.IGNORECASE),
    re.compile(r"^(?:请|麻烦)?\s*清除\s*(?P<alias>[A-Za-z]\d*_[A-Za-z0-9_]+)\s*的确认\s*$", re.IGNORECASE),
    re.compile(r"^(?:请|麻烦)?\s*(?P<alias>[A-Za-z]\d*_[A-Za-z0-9_]+)\s*(?:不是这个|重新确认|重新选择)\s*$", re.IGNORECASE),
]
SCOPE_SWITCH_PROJECT_PATTERNS = [
    re.compile(
        r"^(?:请|麻烦)?\s*把\s*(?P<alias>[A-Za-z]\d*_[A-Za-z0-9_]+)\s*改成\s*(?P<project>.+?)\s*那个\s*$",
        re.IGNORECASE,
    ),
]


def _extract_scope_control_command(message: str) -> Optional[dict]:
    content = str(message or "").strip()
    if not content:
        return None
    for pattern in SCOPE_CLEAR_ALL_PATTERNS:
        if pattern.search(content):
            return {"action": "clear_scope"}
    for pattern in SCOPE_SWITCH_PROJECT_PATTERNS:
        match = pattern.search(content)
        if match:
            alias = str(match.group("alias") or "").strip()
            project_hint = str(match.group("project") or "").strip()
            if alias and project_hint:
                return {"action": "switch_project", "alias": alias, "project_hint": project_hint}
    for pattern in SCOPE_RESET_ALIAS_PATTERNS:
        match = pattern.search(content)
        if match:
            alias = str(match.group("alias") or "").strip()
            if alias:
                return {"action": "reset_alias", "alias": alias}
    return None


def _score_project_hint(device_payload: dict, project_hint: str) -> int:
    normalized_hint = _normalize_alias_key(project_hint)
    if not normalized_hint:
        return 0
    project_name = _normalize_alias_key(device_payload.get("project_name"))
    project_code_name = _normalize_alias_key(device_payload.get("project_code_name"))
    if normalized_hint == project_name:
        return 100
    if normalized_hint == project_code_name:
        return 95
    if normalized_hint and normalized_hint in project_name:
        return 80
    if normalized_hint and normalized_hint in project_code_name:
        return 75
    return 0


def _resolve_alias_to_project_device(alias: str, project_hint: str) -> Optional[dict]:
    normalized_alias = str(alias or "").strip()
    normalized_hint = str(project_hint or "").strip()
    if not normalized_alias or not normalized_hint:
        return None
    try:
        devices, _ = metadata_engine.search_devices(normalized_alias)
    except Exception as exc:  # pragma: no cover
        logger.warning("scope.resolve_project_alias.failed alias=%s project=%s error=%s", normalized_alias, normalized_hint, exc)
        return None

    exact_rows = []
    normalized_code = _normalize_alias_key(normalized_alias)
    for device in devices or []:
        payload = device.to_dict() if hasattr(device, "to_dict") else {
            "device": getattr(device, "device", ""),
            "name": getattr(device, "name", ""),
            "project_id": getattr(device, "project_id", ""),
            "project_name": getattr(device, "project_name", None),
            "project_code_name": getattr(device, "project_code_name", None),
            "device_type": getattr(device, "device_type", None),
            "tg": getattr(device, "tg", None),
        }
        if _normalize_alias_key(payload.get("device")) == normalized_code:
            exact_rows.append(payload)

    if not exact_rows:
        return None

    scored = []
    for row in exact_rows:
        score = _score_project_hint(row, normalized_hint)
        if score > 0:
            scored.append((score, row))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    top_score = scored[0][0]
    top_rows = [row for score, row in scored if score == top_score]
    if len(top_rows) == 1:
        return top_rows[0]
    return None


def _infer_data_type_from_context(message: str, query_info: Optional[dict]) -> str:
    context = query_info or {}
    plan_context = context.get("query_plan_context") if isinstance(context, dict) else {}
    if not isinstance(plan_context, dict):
        plan_context = {}

    requested_tags = [str(tag).lower() for tag in (plan_context.get("requested_tags") or []) if str(tag).strip()]
    single_phase_tags = {"ua", "ub", "uc", "ia", "ib", "ic"}
    if len(requested_tags) == 1 and requested_tags[0] in single_phase_tags:
        return requested_tags[0]

    data_type = str(plan_context.get("data_type") or "").strip().lower()
    if data_type:
        return data_type

    normalized = str(message or "").lower()
    if any(keyword in normalized for keyword in ["ua", "ub", "uc", "电压"]):
        return "u_line"
    if any(keyword in normalized for keyword in ["ia", "ib", "ic", "电流"]):
        return "i"
    if any(keyword in normalized for keyword in ["功率", "p"]):
        return "p"
    return "ep"


def _extract_device_search_keywords_from_message(message: str) -> List[str]:
    normalized = str(message or "").strip()
    if not normalized:
        return []

    keywords: List[str] = []
    explicit_codes = re.findall(r"[A-Za-z]+\d+_[A-Za-z]+\d+", normalized)
    for code in explicit_codes:
        if code not in keywords:
            keywords.append(code)

    cleaned = normalized
    for token in ["搜索", "查询", "查一下", "查一查", "查找", "列出", "有哪些", "有啥", "什么", "设备", "列表", "信息"]:
        cleaned = cleaned.replace(token, " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned and cleaned not in keywords:
        keywords.append(cleaned)

    if "b9" in normalized.lower() and "b9" not in keywords:
        keywords.append("b9")

    return keywords[:3]


@lru_cache(maxsize=512)
def _has_multiple_exact_scope_candidates(keyword: str) -> bool:
    normalized_keyword = _normalize_alias_key(keyword)
    if not normalized_keyword:
        return False
    try:
        devices, _ = metadata_engine.search_devices(keyword)
    except Exception as exc:  # pragma: no cover
        logger.warning("scope.multiple_exact_candidates.failed keyword=%s error=%s", keyword, exc)
        return False

    exact_scopes = set()
    for device in devices or []:
        payload = device.to_dict() if hasattr(device, "to_dict") else {
            "device": getattr(device, "device", ""),
            "project_id": getattr(device, "project_id", ""),
            "project_name": getattr(device, "project_name", None),
            "project_code_name": getattr(device, "project_code_name", None),
            "tg": getattr(device, "tg", None),
        }
        if _normalize_alias_key(payload.get("device")) != normalized_keyword:
            continue
        exact_scopes.add(
            (
                str(payload.get("project_id") or "").strip(),
                str(payload.get("project_name") or payload.get("project_code_name") or "").strip(),
                str(payload.get("tg") or "").strip(),
            )
        )
        if len(exact_scopes) > 1:
            return True
    return False


def _scope_item_can_switch(raw: dict) -> bool:
    alias = str(raw.get("alias") or "").strip()
    device_code = str(raw.get("device") or "").strip()
    if alias and _normalize_alias_key(alias) != _normalize_alias_key(device_code):
        return True
    if not device_code:
        return False
    return _has_multiple_exact_scope_candidates(device_code)


def _resolve_device_from_catalog(device_code: str, project_id: Optional[str] = None, project_name: Optional[str] = None, project_code_name: Optional[str] = None) -> Optional[dict]:
    normalized_code = _normalize_alias_key(device_code)
    if not normalized_code:
        return None
    normalized_project_id = str(project_id or "").strip()
    normalized_project_name = _normalize_alias_key(project_name)
    normalized_project_code_name = _normalize_alias_key(project_code_name)
    try:
        devices = metadata_engine.list_all_devices()
    except Exception as exc:  # pragma: no cover
        logger.warning("scope.resolve_catalog_device.failed device=%s error=%s", device_code, exc)
        return None

    matches = []
    for device in devices or []:
        payload = device.to_dict() if hasattr(device, "to_dict") else {
            "device": getattr(device, "device", ""),
            "name": getattr(device, "name", ""),
            "project_id": getattr(device, "project_id", ""),
            "project_name": getattr(device, "project_name", None),
            "project_code_name": getattr(device, "project_code_name", None),
            "device_type": getattr(device, "device_type", None),
            "tg": getattr(device, "tg", None),
        }
        if _normalize_alias_key(payload.get("device")) != normalized_code:
            continue
        if normalized_project_id and str(payload.get("project_id") or "").strip() == normalized_project_id:
            matches.append(payload)
            continue
        if normalized_project_name and _normalize_alias_key(payload.get("project_name")) == normalized_project_name:
            matches.append(payload)
            continue
        if normalized_project_code_name and _normalize_alias_key(payload.get("project_code_name")) == normalized_project_code_name:
            matches.append(payload)
            continue
    if len(matches) == 1:
        return matches[0]
    return None


def _extract_alias_action(alias_confirmation: Optional[dict]) -> Optional[dict]:
    if not isinstance(alias_confirmation, dict):
        return None
    action = str(alias_confirmation.get("action") or "").strip().lower()
    if action not in {"reset_alias", "clear_scope", "switch_project"}:
        return None
    payload = {"action": action}
    if alias_confirmation.get("alias"):
        payload["alias"] = str(alias_confirmation.get("alias") or "").strip()
    if alias_confirmation.get("project_hint"):
        payload["project_hint"] = str(alias_confirmation.get("project_hint") or "").strip()
    if alias_confirmation.get("original_question"):
        payload["original_question"] = str(alias_confirmation.get("original_question") or "").strip()
    return payload


def _build_device_snapshot(device: Optional[dict], alias: Optional[str] = None, source: str = "confirmation") -> Optional[dict]:
    if not isinstance(device, dict):
        return None
    device_code = device.get("device")
    if not device_code:
        return None
    return {
        "device": device_code,
        "name": device.get("name"),
        "project_id": device.get("project_id"),
        "project_name": device.get("project_name"),
        "project_code_name": device.get("project_code_name"),
        "device_type": device.get("device_type"),
        "tg": device.get("tg"),
        "match_score": device.get("match_score", 1000.0),
        "matched_fields": device.get("matched_fields") or ["session_alias"],
        "match_reason": device.get("match_reason") or "同一会话历史确认",
        "alias": alias,
        "source": source,
        "updated_at": datetime.now().isoformat(),
    }


def _lookup_device_by_code(device_code: str) -> Optional[dict]:
    if not device_code:
        return None
    try:
        devices, _ = metadata_engine.search_devices(device_code)
        normalized_code = str(device_code).strip().lower()
        for device in devices:
            if str(device.device or "").strip().lower() == normalized_code:
                return device.to_dict()
        if devices:
            return devices[0].to_dict()
    except Exception as exc:  # pragma: no cover
        logger.warning("alias.lookup_device.failed device=%s error=%s", device_code, exc)
    return None


def _learn_session_alias(session_id: str, alias: str, device: dict, source: str = "confirmation") -> Optional[dict]:
    normalized_alias = _normalize_alias_key(alias)
    snapshot = _build_device_snapshot(device, alias=alias, source=source)
    if not session_id or not normalized_alias or snapshot is None:
        return None

    state = _get_session_state(session_id)
    alias_map = state.setdefault("device_aliases", {})
    alias_map[normalized_alias] = snapshot
    while len(alias_map) > SESSION_ALIAS_LIMIT:
        first_key = next(iter(alias_map))
        alias_map.pop(first_key, None)
    return snapshot


def _collect_alias_variants(alias: str, alias_confirmation: Optional[dict], device_payload: dict) -> List[str]:
    values: List[str] = []
    seen = set()

    def _add(value: Any) -> None:
        text = str(value or "").strip()
        normalized = _normalize_alias_key(text)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        values.append(text)

    _add(alias)
    if isinstance(alias_confirmation, dict):
        _add(alias_confirmation.get("keyword"))
        _add(alias_confirmation.get("target"))
        original_question = str(alias_confirmation.get("original_question") or "").strip()
        if original_question:
            for match in DEVICE_CODE_PATTERN.findall(original_question):
                if _normalize_alias_key(match) == _normalize_alias_key(device_payload.get("device")):
                    _add(match)
    _add(device_payload.get("device"))
    return values


def _apply_alias_confirmation(session_id: str, alias_confirmation: Optional[dict], *, source: str = "confirmation") -> Optional[dict]:
    if not isinstance(alias_confirmation, dict):
        return None
    alias = alias_confirmation.get("alias") or alias_confirmation.get("keyword") or alias_confirmation.get("target")
    alias = str(alias or "").strip()
    if not alias:
        return None

    device_payload = alias_confirmation.get("device_info") if isinstance(alias_confirmation.get("device_info"), dict) else None
    if device_payload is None:
        device_payload = {
            "device": alias_confirmation.get("device"),
            "name": alias_confirmation.get("name"),
            "project_id": alias_confirmation.get("project_id"),
            "project_name": alias_confirmation.get("project_name"),
            "project_code_name": alias_confirmation.get("project_code_name"),
            "device_type": alias_confirmation.get("device_type"),
            "tg": alias_confirmation.get("tg"),
            "match_score": alias_confirmation.get("match_score", 1000.0),
            "matched_fields": alias_confirmation.get("matched_fields") or ["session_alias"],
            "match_reason": alias_confirmation.get("match_reason") or "同一会话历史确认",
        }

    if not device_payload.get("device"):
        return None

    if (not device_payload.get("name")) or (not device_payload.get("tg")):
        resolved_device = None
        for project_hint in (
            device_payload.get("project_name"),
            device_payload.get("project_code_name"),
        ):
            if project_hint:
                resolved_device = _resolve_alias_to_project_device(device_payload.get("device"), str(project_hint))
                if resolved_device:
                    break
        if resolved_device is None or not resolved_device.get("tg"):
            catalog_device = _resolve_device_from_catalog(
                device_payload.get("device"),
                project_id=device_payload.get("project_id"),
                project_name=device_payload.get("project_name"),
                project_code_name=device_payload.get("project_code_name"),
            )
            if catalog_device:
                resolved_device = catalog_device
        if resolved_device is None:
            resolved_device = _lookup_device_by_code(device_payload.get("device"))
        if resolved_device:
            resolved_device.update({k: v for k, v in device_payload.items() if v not in (None, "", [], {})})
            device_payload = resolved_device

    learned_snapshot = None
    for alias_variant in _collect_alias_variants(alias, alias_confirmation, device_payload):
        snapshot = _learn_session_alias(session_id, alias_variant, device_payload, source=source)
        if learned_snapshot is None and snapshot is not None:
            learned_snapshot = snapshot
    return learned_snapshot


def _extract_alias_confirmation_from_message(message: str) -> Optional[dict]:
    content = str(message or "").strip()
    if not content:
        return None
    for pattern in ALIAS_CONFIRM_PATTERNS:
        match = pattern.search(content)
        if match:
            alias = str(match.group("alias") or "").strip()
            device = str(match.group("device") or "").strip()
            if alias and device:
                return {"alias": alias, "device": device}
    return None


def _create_chat_agent(alias_memory: Optional[Dict[str, Any]] = None):
    orchestrator_type = str(getattr(config.agent, "orchestrator_type", "react") or "react").strip().lower()
    if orchestrator_type == "dag":
        logger.info("chat.agent.create orchestrator=dag")
        return DAGOrchestrator(
            llm=llm,
            metadata_engine=metadata_engine,
            data_fetcher=data_fetcher,
            cache_manager=None,
            compressor=compressor,
            alias_memory=alias_memory,
        )

    logger.info("chat.agent.create orchestrator=react")
    return create_agent_with_streaming(
        llm=llm,
        llm_non_streaming=llm,
        metadata_engine=metadata_engine,
        data_fetcher=data_fetcher,
        cache_manager=None,
        compressor=compressor,
        alias_memory=alias_memory,
        entity_resolver=entity_resolver,
    )


def _prepare_chat_context(request: "ChatRequest") -> Dict[str, Any]:
    session_id = _get_or_create_session_id(request.session_id)
    history = request.history or []
    learned_aliases: List[dict] = []
    session_state = _get_session_state(session_id)

    scope_action = _extract_alias_action(request.alias_confirmation) or _extract_scope_control_command(request.message)
    if scope_action:
        original_question = str(scope_action.get("original_question") or session_state.get("last_user_query") or "").strip()
        effective_message = original_question or request.message

        if scope_action.get("action") == "clear_scope":
            _clear_session_scope(session_id)
        elif scope_action.get("action") == "reset_alias":
            _clear_session_alias(session_id, scope_action.get("alias"))
        elif scope_action.get("action") == "switch_project":
            alias = str(scope_action.get("alias") or "").strip()
            project_hint = str(scope_action.get("project_hint") or "").strip()
            _clear_session_alias(session_id, alias)
            resolved_device = _resolve_alias_to_project_device(alias, project_hint)
            if resolved_device:
                switched_alias = _apply_alias_confirmation(
                    session_id,
                    {
                        "alias": alias,
                        "keyword": alias,
                        "device_info": resolved_device,
                        "device": resolved_device.get("device"),
                        "name": resolved_device.get("name"),
                        "project_id": resolved_device.get("project_id"),
                        "project_name": resolved_device.get("project_name"),
                        "project_code_name": resolved_device.get("project_code_name"),
                        "device_type": resolved_device.get("device_type"),
                        "tg": resolved_device.get("tg"),
                        "original_question": original_question,
                    },
                    source="project_switch",
                )
                if switched_alias:
                    learned_aliases.append(switched_alias)
                effective_message = original_question or request.message
            elif original_question:
                effective_message = f"{project_hint} {original_question}".strip()

        session_state = _get_session_state(session_id)
        if effective_message and effective_message != request.message:
            session_state["last_user_query"] = effective_message
        return {
            "session_id": session_id,
            "history": history,
            "effective_message": effective_message,
            "alias_memory": session_state.get("device_aliases", {}),
            "learned_aliases": learned_aliases,
        }

    confirmed_alias = _apply_alias_confirmation(session_id, request.alias_confirmation, source="button_confirm")
    if confirmed_alias:
        learned_aliases.append(confirmed_alias)

    parsed_alias = _extract_alias_confirmation_from_message(request.message)
    if parsed_alias:
        learned_from_message = _apply_alias_confirmation(session_id, parsed_alias, source="message_confirm")
        if learned_from_message:
            learned_aliases.append(learned_from_message)

    effective_message = request.message
    if isinstance(request.alias_confirmation, dict):
        original_question = str(request.alias_confirmation.get("original_question") or "").strip()
        if original_question:
            effective_message = original_question

    if effective_message:
        session_state["last_user_query"] = effective_message
    session_state = _get_session_state(session_id)
    return {
        "session_id": session_id,
        "history": history,
        "effective_message": effective_message,
        "alias_memory": session_state.get("device_aliases", {}),
        "learned_aliases": learned_aliases,
    }


def _normalize_scope_value(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _finalize_resolved_scope_payload(
    *,
    title: str,
    items: List[dict],
    tg_count: int,
    aggregation_scope_codes: Optional[List[str]] = None,
) -> dict:
    normalized_items = list(items or [])
    normalized_aggregation_scope_codes = list(aggregation_scope_codes or [])
    device_count = len(normalized_items)
    suppressed = device_count > RESOLVED_SCOPE_CARD_ITEM_LIMIT
    return {
        "title": title,
        "items": [] if suppressed else normalized_items,
        "device_count": device_count,
        "tg_count": tg_count,
        "aggregation_scope_codes": normalized_aggregation_scope_codes,
        "scope_card_suppressed": suppressed,
        "scope_card_item_limit": RESOLVED_SCOPE_CARD_ITEM_LIMIT,
    }


def _build_resolved_scope(
    query_params: Optional[dict],
    alias_memory: Optional[Dict[str, Any]] = None,
    learned_aliases: Optional[List[dict]] = None,
) -> Optional[dict]:
    if not isinstance(query_params, dict):
        return None

    device_codes = [str(code).strip() for code in query_params.get("device_codes") or [] if str(code).strip()]
    tg_values = [str(value).strip() for value in query_params.get("tg_values") or [] if str(value).strip()]
    comparison_scope_groups = query_params.get("comparison_scope_groups") if isinstance(query_params.get("comparison_scope_groups"), dict) else None
    if not device_codes:
        return None

    if comparison_scope_groups:
        items: List[dict] = []
        seen = set()
        for scopes in comparison_scope_groups.values():
            if not isinstance(scopes, list):
                continue
            for raw in scopes:
                if not isinstance(raw, dict):
                    continue
                device_code = str(raw.get("device") or "").strip()
                if not device_code:
                    continue
                dedupe_key = (
                    _normalize_scope_value(device_code),
                    _normalize_scope_value(raw.get("project_id")),
                    _normalize_scope_value(raw.get("project_name")),
                    _normalize_scope_value(raw.get("tg")),
                )
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                items.append({
                    "device": device_code,
                    "name": raw.get("name"),
                    "project_id": raw.get("project_id"),
                    "project_name": raw.get("project_name"),
                    "project_code_name": raw.get("project_code_name"),
                    "tg": str(raw.get("tg") or "").strip() or None,
                    "alias": raw.get("alias"),
                    "source": raw.get("source") or "comparison_scope",
                    "can_switch": _scope_item_can_switch(raw),
                })
        if items:
            items.sort(key=lambda item: (
                str(item.get("device") or ""),
                str(item.get("project_name") or item.get("project_code_name") or ""),
                str(item.get("tg") or ""),
            ))
            return _finalize_resolved_scope_payload(
                title="\u5f53\u524d\u786e\u8ba4\u4f5c\u7528\u57df",
                items=items,
                tg_count=len({str(item.get("tg") or "").strip() for item in items if str(item.get("tg") or "").strip()}),
                aggregation_scope_codes=[],
            )

    device_lookup = {_normalize_scope_value(code) for code in device_codes}
    device_index = {_normalize_scope_value(code): code for code in device_codes}
    tg_lookup = {_normalize_scope_value(value) for value in tg_values}
    expected_tg_by_device: Dict[str, str] = {}
    if len(device_codes) == len(tg_values):
        for device_code, tg_value in zip(device_codes, tg_values):
            normalized_device = _normalize_scope_value(device_code)
            normalized_tg = _normalize_scope_value(tg_value)
            if normalized_device and normalized_tg:
                expected_tg_by_device[normalized_device] = normalized_tg

    def _iter_sources():
        for item in learned_aliases or []:
            if isinstance(item, dict):
                yield item
        if isinstance(alias_memory, dict):
            for item in alias_memory.values():
                if isinstance(item, dict):
                    yield item

    items: List[dict] = []
    seen = set()

    def _append_scope_item(raw: dict, source: str) -> None:
        device_code = str(raw.get("device") or "").strip()
        normalized_device = _normalize_scope_value(device_code)
        if not device_code or normalized_device not in device_lookup:
            return
        tg_value = str(raw.get("tg") or "").strip()
        normalized_tg = _normalize_scope_value(tg_value)
        expected_tg = expected_tg_by_device.get(normalized_device)
        if expected_tg:
            if normalized_tg != expected_tg:
                return
        elif tg_lookup and normalized_tg not in tg_lookup:
            return
        dedupe_key = (
            normalized_device,
            _normalize_scope_value(raw.get("project_id")),
            _normalize_scope_value(raw.get("project_name")),
            normalized_tg,
        )
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        items.append(
            {
                "device": device_code,
                "name": raw.get("name"),
                "project_id": raw.get("project_id"),
                "project_name": raw.get("project_name"),
                "project_code_name": raw.get("project_code_name"),
                "tg": tg_value or None,
                "alias": raw.get("alias"),
                "source": source,
                "can_switch": _scope_item_can_switch(raw),
            }
        )

    def _resolved_device_lookup() -> set:
        return {_normalize_scope_value(item.get("device")) for item in items if item.get("device")}

    for item in _iter_sources():
        _append_scope_item(item, str(item.get("source") or "session_alias"))

    missing_devices = [
        device_index[normalized_device]
        for normalized_device in device_lookup
        if normalized_device not in _resolved_device_lookup()
    ]
    if missing_devices:
        try:
            for device in metadata_engine.list_all_devices():
                device_dict = device.to_dict()
                _append_scope_item(device_dict, "metadata_catalog")
        except Exception as exc:  # pragma: no cover
            logger.warning("scope.resolve.metadata_catalog_failed error=%s", exc)

    resolved_after_catalog = _resolved_device_lookup()
    for device_code in device_codes:
        normalized_device = _normalize_scope_value(device_code)
        if normalized_device in resolved_after_catalog:
            continue
        fallback_tg = None
        expected_tg = expected_tg_by_device.get(normalized_device)
        if expected_tg:
            fallback_tg = next((tg for tg in tg_values if _normalize_scope_value(tg) == expected_tg), None)
        elif len(device_codes) == 1 and len(tg_values) == 1:
            fallback_tg = tg_values[0]
        items.append(
            {
                "device": device_code,
                "name": None,
                "project_id": None,
                "project_name": None,
                "project_code_name": None,
                "tg": fallback_tg,
                "alias": None,
                "source": "query_params_fallback",
                "can_switch": _has_multiple_exact_scope_candidates(device_code),
            }
        )

    if not items:
        return None

    items.sort(
        key=lambda item: (
            str(item.get("device") or ""),
            str(item.get("project_name") or item.get("project_code_name") or ""),
            str(item.get("tg") or ""),
        )
    )
    scope_user_query = str(query_params.get("user_query") or "").strip()
    aggregation_scope_codes: List[str] = []
    for device_code in sorted({str(item.get("device") or "").strip() for item in items if str(item.get("device") or "").strip()}):
        matched_items = [item for item in items if str(item.get("device") or "").strip() == device_code]
        project_scope_count = len({
            (
                str(item.get("project_id") or ""),
                str(item.get("project_name") or item.get("project_code_name") or ""),
                str(item.get("tg") or ""),
            )
            for item in matched_items
        })
        if project_scope_count > 1 and allows_explicit_multi_scope_aggregation(scope_user_query, device_code):
            aggregation_scope_codes.append(device_code)
    return _finalize_resolved_scope_payload(
        title="\u5f53\u524d\u786e\u8ba4\u4f5c\u7528\u57df",
        items=items,
        tg_count=len({str(item.get("tg") or "").strip() for item in items if str(item.get("tg") or "").strip()}),
        aggregation_scope_codes=aggregation_scope_codes,
    )


def _build_message_with_history(message: str, history: Optional[List[dict]]) -> str:
    if not history:
        return message

    history_text = "\n".join(
        f"{'用户' if item.get('role') == 'user' else 'AI'}: {item.get('content', '')}"
        for item in history[-CHAT_HISTORY_LIMIT:]
    )
    return f"对话历史:\n{history_text}\n\n当前问题: {message}"


def _log_structured(event: str, **fields) -> None:
    logger.info(
        "%s",
        json.dumps({"event": event, **fields}, ensure_ascii=False, default=str),
    )


# 初始化组件
def _build_error_payload(message: str, *, code: str = "INTERNAL_ERROR", request_id: Optional[str] = None, detail=None) -> dict:
    payload = {
        "success": False,
        "error": message or "服务处理请求时发生异常，请稍后重试。",
        "error_code": code,
    }
    if request_id:
        payload["request_id"] = request_id
    if detail is not None:
        payload["detail"] = detail
    return payload


def _build_success_payload(*, request_id: Optional[str] = None, **fields) -> dict:
    payload = {"success": True, **fields}
    if request_id:
        payload["request_id"] = request_id
    return payload


def _log_exception_event(event: str, *, error: Exception, request_id: Optional[str] = None, **fields) -> None:
    payload = {"event": event, **fields, "error": str(error)}
    if request_id:
        payload["request_id"] = request_id
    logger.exception("%s", json.dumps(payload, ensure_ascii=False, default=str))


def _build_empty_result_payload(
    *,
    request_id: Optional[str] = None,
    data=None,
    total_count: int = 0,
    page: int = 1,
    page_size: int = 0,
    total_pages: int = 1,
    has_more: bool = False,
    statistics=None,
    analysis=None,
    chart_specs=None,
    show_charts: bool = False,
    is_sampled: bool = False,
    is_aggregated: bool = False,
    aggregation_type=None,
) -> dict:
    payload = {
        "success": True,
        "data": list(data or []),
        "total_count": total_count,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "has_more": has_more,
        "statistics": statistics,
        "analysis": analysis,
        "chart_specs": chart_specs,
        "show_charts": show_charts,
        "is_sampled": is_sampled,
        "is_aggregated": is_aggregated,
        "aggregation_type": aggregation_type,
        "message": EMPTY_RESULT_MESSAGE,
        "empty_result": True,
        "error": None,
        "error_code": None,
    }
    if request_id:
        payload["request_id"] = request_id
    return payload



def _parse_query_datetime(value: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("\u7f3a\u5c11\u65f6\u95f4\u8303\u56f4")
    normalized = text.replace("/", "-")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    raise ValueError(f"\u65e0\u6548\u7684\u65f6\u95f4\u683c\u5f0f: {value}")



def _build_comparison_query_result(request: "DataQueryRequest") -> Dict[str, Any]:
    start_dt = _parse_query_datetime(request.start_time)
    end_dt = _parse_query_datetime(request.end_time)
    if start_dt > end_dt:
        raise ValueError("\u5f00\u59cb\u65f6\u95f4\u4e0d\u80fd\u665a\u4e8e\u7ed3\u675f\u65f6\u95f4")

    collection_prefix = get_collection_prefix(request.data_type)
    collections = get_target_collections(request.start_time, request.end_time, collection_prefix)
    data_tags = get_data_tags(collection_prefix)

    comparison_scope_groups = request.comparison_scope_groups or {}
    requested_codes = [str(code).strip() for code in (request.device_codes or []) if str(code or "").strip()]
    merged_records: List[Dict[str, Any]] = []
    device_names: Dict[str, str] = {}
    failed_collections: List[str] = []
    target_profiles: Dict[str, Dict[str, Any]] = {}

    def _normalize_records(rows: Any) -> List[dict]:
        return [row for row in (rows or []) if isinstance(row, dict)]

    fetch_targets: List[tuple[str, List[dict]]] = []
    for target, rows in comparison_scope_groups.items():
        fetch_targets.append((str(target or "").strip(), _normalize_records(rows)))

    covered_targets = {target for target, _ in fetch_targets if target}
    for device_code in requested_codes:
        if device_code not in covered_targets:
            fetch_targets.append((device_code, []))

    for target, scope_rows in fetch_targets:
        scoped_devices = list(dict.fromkeys(
            str(row.get("device") or "").strip()
            for row in scope_rows
            if str(row.get("device") or "").strip()
        ))
        if not scoped_devices and target:
            scoped_devices = [target]

        scoped_tgs = list(dict.fromkeys(
            str(row.get("tg") or "").strip()
            for row in scope_rows
            if str(row.get("tg") or "").strip()
        ))
        if not scoped_tgs:
            scoped_tgs = [str(value).strip() for value in (request.tg_values or []) if str(value or "").strip()] if not scope_rows else []

        display_name = target or (scoped_devices[0] if scoped_devices else "")
        for row in scope_rows:
            device_code = str(row.get("device") or "").strip()
            if not device_code:
                continue
            row_display_name = str(row.get("name") or "").strip() or device_code
            project_name = str(row.get("project_name") or "").strip()
            if project_name and project_name not in row_display_name:
                row_display_name = f"{row_display_name}（{project_name}）"
            device_names.setdefault(device_code, row_display_name)
            if device_code == target or not display_name:
                display_name = row_display_name

        target_profiles[target] = {
            "target": target,
            "display_name": display_name or target,
            "devices": list(scoped_devices),
            "tgs": list(scoped_tgs),
            "record_count": 0,
        }

        if not scoped_devices:
            continue

        sensor_result = data_fetcher.fetch_sync(
            collections=collections,
            devices=scoped_devices,
            tgs=scoped_tgs or None,
            start_time=start_dt,
            end_time=end_dt,
            tags=data_tags,
            page=1,
            page_size=0,
            value_filter=request.value_filter,
        )
        target_profiles[target]["record_count"] = int(sensor_result.total_count or 0)
        if sensor_result.data:
            merged_records.extend([item for item in sensor_result.data if isinstance(item, dict)])
        failed_collections.extend(list(sensor_result.failed_collections or []))

    deduped_records: List[Dict[str, Any]] = []
    seen_keys = set()
    for item in merged_records:
        dedupe_key = (
            str(item.get("device") or ""),
            str(item.get("tg") or ""),
            str(item.get("tag") or ""),
            str(item.get("logTime") or item.get("time") or item.get("dataTime") or ""),
            str(item.get("val") if item.get("val") is not None else item.get("value") if item.get("value") is not None else ""),
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        deduped_records.append(item)

    deduped_records.sort(
        key=lambda item: (
            str(item.get("logTime") or item.get("time") or item.get("dataTime") or ""),
            str(item.get("device") or ""),
            str(item.get("tag") or ""),
            str(item.get("tg") or ""),
            str(item.get("_id") or ""),
        )
    )

    analysis, chart_specs = InsightEngine.build(
        deduped_records,
        None,
        data_type=request.data_type,
        device_codes=request.device_codes,
        device_names=device_names,
        user_query=request.user_query,
    )

    if isinstance(analysis, dict) and len(target_profiles) > 1:
        analysis.setdefault("mode", "comparison")
        analysis.setdefault("devices", [])
        analysis.setdefault("rankings", {})
        existing_names = {
            str(item.get("name") or "").strip()
            for item in (analysis.get("devices") or [])
            if isinstance(item, dict)
        }
        missing_profiles = [
            profile
            for profile in target_profiles.values()
            if str(profile.get("display_name") or "").strip()
            and str(profile.get("display_name") or "").strip() not in existing_names
        ]
        for profile in missing_profiles:
            analysis["devices"].append(
                {
                    "name": profile.get("display_name") or profile.get("target"),
                    "avg": None,
                    "sum": None,
                    "max": None,
                    "min": None,
                    "cv": None,
                    "trend": "无数据",
                    "anomaly_ratio": None,
                }
            )

        for ranking_key in ("avg", "sum", "stability"):
            ranking_rows = analysis["rankings"].setdefault(ranking_key, [])
            ranked_names = {
                str(item.get("name") or "").strip()
                for item in ranking_rows
                if isinstance(item, dict)
            }
            for profile in target_profiles.values():
                display_name = str(profile.get("display_name") or profile.get("target") or "").strip()
                if not display_name or display_name in ranked_names:
                    continue
                ranking_rows.append({"name": display_name, "value": None})
                ranked_names.add(display_name)

        no_data_names = [
            str(profile.get("display_name") or profile.get("target") or "").strip()
            for profile in target_profiles.values()
            if int(profile.get("record_count") or 0) <= 0
        ]
        if no_data_names:
            insights = analysis.setdefault("insights", [])
            note = f"以下设备在当前时间范围内无数据：{'、'.join(no_data_names)}。"
            if note not in insights:
                insights.append(note)
            headline = str(analysis.get("headline") or "")
            if headline:
                analysis["headline"] = headline + f"；{len(no_data_names)} 个设备暂无数据"

    total_count = len(deduped_records)
    page = max(int(request.page or 1), 1)
    page_size = max(int(request.page_size or 0), 0)
    if page_size > 0:
        total_pages = max((total_count + page_size - 1) // page_size, 1)
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        page_records = deduped_records[start_index:end_index]
        has_more = end_index < total_count
    else:
        total_pages = 1
        page_records = deduped_records
        has_more = False

    return {
        "success": True,
        "data": page_records,
        "total_count": total_count,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "has_more": has_more,
        "statistics": None,
        "analysis": analysis,
        "focused_table": None,
        "chart_specs": chart_specs,
        "show_charts": False,
        "failed_collections": sorted(set(failed_collections)),
        "is_sampled": False,
        "is_aggregated": False,
        "aggregation_type": None,
        "error": None,
    }

def _is_empty_query_result(result: dict, table_data: list) -> bool:
    if not isinstance(result, dict) or not result.get("success"):
        return False

    total_count = result.get("total_count")
    if isinstance(total_count, int):
        return total_count == 0

    raw_data = result.get("data")
    if raw_data in (None, "", [], {}, ()):  # pragma: no cover - defensive
        return True
    if isinstance(raw_data, str) and raw_data.strip() in {"", "[]", "{}"}:
        return True
    return len(table_data) == 0


config = load_config()
metadata_engine = MetadataEngine(config.mysql.connection_string)
mongo_client = MongoClient(config.mongodb.uri)
data_fetcher = DataFetcher(
    mongo_client=mongo_client,
    database_name=config.mongodb.database_name,
    max_records=2000
)
compressor = ContextCompressor(max_tokens=4000)
entity_resolver = ChromaEntityResolver(
    metadata_engine=metadata_engine,
    persist_directory=os.getenv("ENTITY_RESOLVER_CHROMA_PATH", "data/chroma_entity_resolver"),
    embedding_api_key=(
        os.getenv("EMBEDDING_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or config.semantic_layer.dashscope_api_key
    ),
    embedding_model=(
        os.getenv("ENTITY_RESOLVER_EMBEDDING_MODEL")
        or os.getenv("EMBEDDING_MODEL")
        or config.semantic_layer.embedding_model
        or "text-embedding-v4"
    ),
    embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS") or config.semantic_layer.embedding_dimensions),
    embedding_base_url=os.getenv("EMBEDDING_BASE_URL"),
    embedding_timeout=float(os.getenv("ENTITY_RESOLVER_EMBEDDING_TIMEOUT", "20")),
    semantic_top_k=int(os.getenv("ENTITY_RESOLVER_TOP_K", "8")),
    refresh_interval_seconds=int(os.getenv("ENTITY_RESOLVER_REFRESH_SECONDS", "1800")),
    enabled=os.getenv("ENTITY_RESOLVER_ENABLED", "true").lower() == "true",
)

# 初始化 LLM
vllm_base = os.getenv("VLLM_API_BASE") or os.getenv("LLM_BASE_URL")
http_client = httpx.Client(trust_env=False, timeout=httpx.Timeout(300.0))  # 增加到 5 分钟
llm = ChatOpenAI(
    model=os.getenv("VLLM_MODEL") or os.getenv("LLM_MODEL", "/models/Qwen3-32B-AWQ"),
    openai_api_base=vllm_base,
    openai_api_key=os.getenv("VLLM_API_KEY") or os.getenv("LLM_API_KEY") or "not-needed",
    temperature=0.7,
    max_tokens=16384,
    http_client=http_client,
    request_timeout=300.0,  # 添加请求超时设置
)


def _coerce_llm_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text_value = item.get("text") or item.get("content") or ""
                if text_value:
                    parts.append(str(text_value))
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def _rewrite_memory_command_with_llm(message: str) -> Optional[Dict[str, Any]]:
    content = str(message or "").strip()
    if not content:
        return None
    prompt = (
        "You rewrite a Chinese memory command for an energy-data assistant.\n"
        "Return JSON only: {\"alias_text\": string, \"target_text\": string, \"confidence\": string, \"source\": \"llm\"}.\n"
        "Rules:\n"
        "1. Remove instruction wrappers and keep only the normalized alias -> canonical target mapping.\n"
        "2. alias_text is what the user says later. target_text is the canonical meaning.\n"
        "3. If one side is a device code like a1_b9 and the other side is a human-friendly name, use the human-friendly name as alias_text and the device code as target_text.\n"
        "4. If the message is not a memory command or you cannot determine a reliable mapping, return {}.\n"
        "Examples:\n"
        "Input: \u628a\u51b7\u6c14\u8bb0\u6210\u7a7a\u8c03\n"
        "Output: {\"alias_text\": \"\u51b7\u6c14\", \"target_text\": \"\u7a7a\u8c03\", \"confidence\": \"high\", \"source\": \"llm\"}\n"
        "Input: \u8bf7\u8bb0\u4f4f\u4ee5\u540ea1_b9\u4ee3\u8868\u4e00\u53f7\u8bbe\u5907\n"
        "Output: {\"alias_text\": \"\u4e00\u53f7\u8bbe\u5907\", \"target_text\": \"a1_b9\", \"confidence\": \"high\", \"source\": \"llm\"}\n"
        "Input: \u5e2e\u6211\u8bb0\u4e00\u4e0b\u7814\u53d1\u52a8\u529b\u8868\u5176\u5b9e\u5c31\u662f\u4e03\u5c42\u52a8\u529b\u8868\n"
        "Output: {\"alias_text\": \"\u7814\u53d1\u52a8\u529b\u8868\", \"target_text\": \"\u4e03\u5c42\u52a8\u529b\u8868\", \"confidence\": \"medium\", \"source\": \"llm\"}\n"
        f"Input: {content}"
    )
    try:
        response = llm.invoke(prompt)
        payload = parse_memory_rewrite_json(_coerce_llm_text_content(getattr(response, "content", response)))
    except Exception as exc:
        logger.warning("memory.rewrite.llm.failed message=%s error=%s", _short_text(content, 80), exc)
        return None
    if not isinstance(payload, dict):
        return None
    alias_text = str(payload.get("alias_text") or "").strip()
    target_text = str(payload.get("target_text") or "").strip()
    if not alias_text or not target_text:
        return None
    confidence = str(payload.get("confidence") or "medium").strip().lower() or "medium"
    return {
        "alias_text": alias_text,
        "target_text": target_text,
        "confidence": confidence,
        "source": str(payload.get("source") or "llm").strip() or "llm",
    }

semantic_rule_store = SemanticRuleStore()
user_memory_store = UserMemoryStore()
chat_memory_service = ChatMemoryService(
    memory_store=user_memory_store,
    semantic_rule_store=semantic_rule_store,
    normalize_alias_key=_normalize_alias_key,
    lookup_device_by_code=_lookup_device_by_code,
    build_device_snapshot=_build_device_snapshot,
    logger=logger,
    llm_memory_rewrite=_rewrite_memory_command_with_llm,
)

app = FastAPI(title="AI Data Router Agent", version=APP_VERSION)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = None  # 对话历史
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    alias_confirmation: Optional[dict] = None


class MemoryAliasCreateRequest(BaseModel):
    alias_text: str
    canonical_text: str
    scope_type: str = "global"
    scope_value: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source: Optional[str] = "user_setting"


class MemoryAliasUpdateRequest(BaseModel):
    alias_text: Optional[str] = None
    canonical_text: Optional[str] = None
    scope_type: Optional[str] = None
    scope_value: Optional[str] = None
    status: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class DataQueryRequest(BaseModel):
    device_codes: List[str]
    tg_values: Optional[List[str]] = None
    start_time: str
    end_time: str
    data_type: str = "ep"
    page: int = 1
    page_size: int = 50
    user_query: str = ""  # 原始用户问题
    query_plan: Optional[dict] = None  # QueryPlan 原始结构
    comparison_scope_groups: Optional[dict] = None
    value_filter: Optional[dict] = None  # 原始用户问题 {"gt": 100} / {"lt": 50}


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回聊天首页"""
    with open("web/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/semantic-mapping", response_class=HTMLResponse)
async def semantic_mapping_admin():
    """返回语义映射管理页"""
    with open("web/semantic_mapping_admin.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/changelog", response_class=HTMLResponse)
async def changelog_page():
    """返回更新日志页"""
    with open("web/changelog.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/memory/alias")
async def list_memory_aliases(user_id: str = "", session_id: str = "", keyword: str = "", canonical_text: str = "", scope_type: str = "", status: str = "enabled", limit: int = 100):
    request_id = uuid4().hex
    resolved_user_id = chat_memory_service.resolve_chat_user_id(user_id, session_id)
    items = [chat_memory_service.memory_view(item) for item in user_memory_store.list_alias_memories(user_id=resolved_user_id, keyword=keyword, canonical_text=canonical_text, scope_type=scope_type, status=status, limit=limit)]
    return _build_success_payload(request_id=request_id, user_id=resolved_user_id, items=items, total=len(items))


@app.post("/api/memory/alias")
async def create_memory_alias(request: MemoryAliasCreateRequest):
    request_id = uuid4().hex
    session_id = _get_or_create_session_id(request.session_id)
    user_id = chat_memory_service.resolve_chat_user_id(request.user_id, session_id)
    target = chat_memory_service.validate_memory_target(request.canonical_text)
    stored = user_memory_store.upsert_alias_memory(
        user_id=user_id,
        alias_text=request.alias_text,
        canonical_text=target.get("canonical_text"),
        target_type=target.get("target_type"),
        target_value=target.get("target_value"),
        scope_type=request.scope_type,
        scope_value=request.scope_value or ("global" if request.scope_type == "global" else ""),
        source=str(request.source or "user_setting"),
    )
    return _build_success_payload(request_id=request_id, user_id=user_id, item=chat_memory_service.memory_view(stored))


@app.put("/api/memory/alias/{memory_id}")
async def update_memory_alias(memory_id: int, request: MemoryAliasUpdateRequest):
    request_id = uuid4().hex
    session_id = _get_or_create_session_id(request.session_id)
    user_id = chat_memory_service.resolve_chat_user_id(request.user_id, session_id)
    updated = user_memory_store.update_alias_memory(
        memory_id=memory_id,
        user_id=user_id,
        alias_text=request.alias_text,
        canonical_text=request.canonical_text,
        scope_type=request.scope_type,
        scope_value=request.scope_value,
        status=request.status,
    )
    return _build_success_payload(request_id=request_id, user_id=user_id, item=chat_memory_service.memory_view(updated))


@app.delete("/api/memory/alias")
async def delete_memory_alias(memory_id: int, user_id: str = "", session_id: str = ""):
    request_id = uuid4().hex
    resolved_user_id = chat_memory_service.resolve_chat_user_id(user_id, session_id)
    deleted = user_memory_store.delete_alias_memory(memory_id=memory_id, user_id=resolved_user_id)
    return _build_success_payload(request_id=request_id, user_id=resolved_user_id, item=chat_memory_service.memory_view(deleted))


@app.get("/api/chat/history")
async def get_chat_history(session_id: str = "", user_id: str = "", limit: int = 50):
    request_id = uuid4().hex
    resolved_session_id = str(session_id or "").strip() or None
    resolved_user_id = str(user_id or "").strip() or None
    items = user_memory_store.list_chat_history(session_id=resolved_session_id, user_id=resolved_user_id, limit=limit)
    return _build_success_payload(request_id=request_id, session_id=resolved_session_id, user_id=resolved_user_id, items=items, total=len(items))


@app.get("/api/chat/sessions")
async def get_chat_sessions(user_id: str = "", session_id: str = "", limit: int = 50):
    request_id = uuid4().hex
    resolved_user_id = chat_memory_service.resolve_chat_user_id(user_id, session_id)
    items = user_memory_store.list_chat_sessions(user_id=resolved_user_id, limit=limit)
    return _build_success_payload(request_id=request_id, user_id=resolved_user_id, items=items, total=len(items))


@app.get("/api/semantic-rules")
async def list_semantic_rules(keyword: str = "", scope_type: str = "", entity_type: str = "", status: str = "", scope_value: str = ""):
    request_id = uuid4().hex
    rules = semantic_rule_store.list_rules(keyword=keyword, scope_type=scope_type, entity_type=entity_type, status=status, scope_value=scope_value)
    summary = {
        "total": len(rules),
        "enabled": sum(1 for rule in rules if str(rule.get("status") or "").lower() == "enabled"),
        "system": sum(1 for rule in rules if str(rule.get("scope_type") or "").lower() == "system"),
        "project": sum(1 for rule in rules if str(rule.get("scope_type") or "").lower() == "project"),
        "user": sum(1 for rule in rules if str(rule.get("scope_type") or "").lower() == "user"),
    }
    return _build_success_payload(request_id=request_id, rules=rules, summary=summary)


@app.get("/api/semantic-rules/history")
async def semantic_rules_history(limit: int = 20):
    request_id = uuid4().hex
    history = semantic_rule_store.history(limit=limit)
    return _build_success_payload(request_id=request_id, history=history)


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """AI chat streaming endpoint (SSE)"""

    async def generate():
        request_id = uuid4().hex
        context = _prepare_chat_context(request)
        session_id = context["session_id"]
        history = context["history"]
        effective_message = context["effective_message"]
        alias_memory = context["alias_memory"]
        learned_aliases = context["learned_aliases"]
        user_id = chat_memory_service.resolve_chat_user_id(request.user_id, session_id)
        intent_type = chat_memory_service.classify_chat_intent(request.message)
        memory_action_result = None
        memory_suggestion = None
        memory_effect_payload = None
        chat_memory_service.record_chat_message(
            session_id=session_id,
            user_id=user_id,
            role="user",
            message=request.message,
            intent_type=intent_type,
        )
        _log_structured(
            "chat.stream.request",
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            intent_type=intent_type,
            history_count=len(history),
            history_used=min(len(history), CHAT_HISTORY_LIMIT),
            current_question=_short_text(effective_message),
            learned_alias_count=len(learned_aliases),
            session_alias_count=len(alias_memory or {}),
        )

        try:
            handled = chat_memory_service.handle_memory_command(
                request_id=request_id,
                session_id=session_id,
                user_id=user_id,
                message=request.message,
                alias_memory=alias_memory,
                alias_confirmation=request.alias_confirmation,
            )
            if handled:
                memory_action_result = handled.get("memory_action_result")
                follow_up = memory_action_result.get("follow_up") if isinstance(memory_action_result, dict) else None
                if isinstance(follow_up, dict):
                    effective_message = chat_memory_service.apply_query_once_alias(
                        {
                            "action": "query_once",
                            "alias_text": follow_up.get("alias_text"),
                            "canonical_text": follow_up.get("canonical_text"),
                            "original_question": follow_up.get("original_question"),
                        },
                        follow_up.get("original_question") or effective_message,
                    )
                    intent_type = "data_query"
                else:
                    chat_memory_service.record_chat_message(
                        session_id=session_id,
                        user_id=user_id,
                        role="assistant",
                        message=handled.get("response") or "",
                        intent_type="memory_command",
                        message_meta={
                            "response": handled.get("response") or "",
                            "intent_type": handled.get("intent_type") or "memory_command",
                            "memory_action_result": handled.get("memory_action_result"),
                            "resolved_scope": handled.get("resolved_scope"),
                            "query_params": handled.get("query_params"),
                            "projects": handled.get("projects"),
                            "devices": handled.get("devices"),
                            "clarification_required": handled.get("clarification_required", False),
                            "clarification_candidates": handled.get("clarification_candidates"),
                        },
                    )
                    complete_payload = {
                        "type": "complete",
                        **handled,
                        "session_id": session_id,
                        "original_question": effective_message,
                        "intent_type": handled.get("intent_type") or intent_type,
                    }
                    yield f"data: {json.dumps(complete_payload, ensure_ascii=False)}\n\n"
                    return

            effective_message = chat_memory_service.apply_query_once_alias(request.alias_confirmation, effective_message)
            memory_apply = chat_memory_service.apply_user_memories(
                effective_message,
                user_id=user_id,
                alias_memory=alias_memory,
            )
            effective_message = memory_apply.get("effective_message") or effective_message
            alias_memory = memory_apply.get("alias_memory") or alias_memory
            memory_effect_payload = chat_memory_service.build_memory_effect_payload(
                memory_apply.get("applied_items") or [],
                memory_apply.get("invalid_items") or [],
            )

            agent = _create_chat_agent(alias_memory=alias_memory)
            message_with_history = _build_message_with_history(effective_message, history)

            for event in agent.run_with_progress(message_with_history):
                event_type = event.get("type")

                if event_type == "step_start":
                    step_start = {"type": "step_start", "step": event["step"]}
                    if event.get("node_name"):
                        step_start["node_name"] = event.get("node_name")
                    if event.get("timestamp_ms") is not None:
                        step_start["timestamp_ms"] = event.get("timestamp_ms")
                    yield f"data: {json.dumps(step_start, ensure_ascii=False)}\n\n"

                elif event_type == "step_done":
                    step_data = {
                        "type": "step_done",
                        "step": event["step"],
                        "info": event.get("info", ""),
                    }
                    if event.get("node_name"):
                        step_data["node_name"] = event.get("node_name")
                    if event.get("duration_ms") is not None:
                        step_data["duration_ms"] = event.get("duration_ms")
                    if event.get("query_info") is not None:
                        step_data["query_info"] = event.get("query_info")
                    _log_structured(
                        "chat.stream.step_done",
                        request_id=request_id,
                        session_id=session_id,
                        step=event.get("step"),
                        duration_ms=event.get("duration_ms"),
                        has_query_info=event.get("query_info") is not None,
                    )
                    yield f"data: {json.dumps(step_data, ensure_ascii=False)}\n\n"

                elif event_type == "final_answer":
                    response_text = event.get("response", "")
                    response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
                    analysis = event.get("analysis") or {}
                    chart_specs = event.get("chart_specs") or []
                    query_params = event.get("query_params")
                    resolved_scope = _build_resolved_scope(
                        query_params,
                        alias_memory=alias_memory,
                        learned_aliases=learned_aliases,
                    )
                    _log_structured(
                        "chat.stream.final",
                        request_id=request_id,
                        session_id=session_id,
                        user_id=user_id,
                        intent_type=intent_type,
                        show_table=event.get("show_table", False),
                        table_type=event.get("table_type", ""),
                        query_params=query_params,
                        resolved_scope_count=len((resolved_scope or {}).get("items") or []),
                        analysis_mode=analysis.get("mode") if isinstance(analysis, dict) else None,
                        chart_count=len(chart_specs) if isinstance(chart_specs, list) else 0,
                        show_charts=event.get("show_charts", False),
                        total_duration_ms=event.get("total_duration_ms"),
                        empty_result=response_text == EMPTY_RESULT_MESSAGE,
                    )

                    for chunk in _iter_answer_delta_chunks(response_text):
                        delta_payload = {
                            "type": "answer_delta",
                            "delta": chunk,
                        }
                        yield f"data: {json.dumps(delta_payload, ensure_ascii=False)}\n\n"
                        if ANSWER_STREAM_CHUNK_DELAY_MS > 0:
                            await asyncio.sleep(ANSWER_STREAM_CHUNK_DELAY_MS / 1000)

                    table_type = event.get("table_type", "")
                    projects = event.get("projects")
                    project_stats = event.get("project_stats")
                    devices = event.get("devices")
                    if isinstance(query_params, dict):
                        if table_type == "projects" and projects is None:
                            projects = query_params.get("projects")
                        elif table_type == "project_stats" and project_stats is None:
                            project_stats = query_params.get("stats") or query_params.get("project_stats")
                        elif table_type == "devices" and devices is None:
                            devices = query_params.get("devices")

                    if table_type == "projects" and projects is None:
                        try:
                            projects = metadata_engine.list_projects()
                        except Exception as exc:
                            logger.warning("chat.stream.projects_fallback.failed request_id=%s error=%s", request_id, exc)
                    elif table_type == "project_stats" and project_stats is None:
                        try:
                            project_stats = metadata_engine.get_project_device_stats()
                        except Exception as exc:
                            logger.warning("chat.stream.project_stats_fallback.failed request_id=%s error=%s", request_id, exc)

                    clarification_required = bool(event.get("clarification_required", False))
                    memory_suggestion = chat_memory_service.build_memory_suggestion(
                        request.message,
                        user_id=user_id,
                        alias_memory=alias_memory,
                        query_params=query_params,
                        clarification_required=clarification_required,
                    )
                    final_data = {
                        "type": "complete",
                        "success": True,
                        "request_id": request_id,
                        "response": response_text,
                        "original_question": effective_message,
                        "session_id": session_id,
                        "intent_type": intent_type,
                        "show_table": event.get("show_table", False),
                        "table_type": table_type,
                        "projects": projects,
                        "project_stats": project_stats,
                        "devices": devices,
                        "query_params": query_params,
                        "resolved_scope": resolved_scope,
                        "analysis": event.get("analysis"),
                        "chart_specs": event.get("chart_specs"),
                        "table_preview": event.get("table_preview"),
                        "total_duration_ms": event.get("total_duration_ms"),
                        "show_charts": event.get("show_charts", False),
                        "clarification_required": clarification_required,
                        "clarification_candidates": event.get("clarification_candidates"),
                        "memory_action_result": memory_action_result or memory_effect_payload,
                        "memory_suggestion": memory_suggestion,
                    }
                    chat_memory_service.record_chat_message(
                        session_id=session_id,
                        user_id=user_id,
                        role="assistant",
                        message=response_text,
                        intent_type=intent_type,
                        message_meta={
                            "response": response_text,
                            "original_question": effective_message,
                            "intent_type": intent_type,
                            "show_table": event.get("show_table", False),
                            "table_type": table_type,
                            "projects": projects,
                            "project_stats": project_stats,
                            "devices": devices,
                            "query_params": query_params,
                            "resolved_scope": resolved_scope,
                            "analysis": event.get("analysis"),
                            "chart_specs": event.get("chart_specs"),
                            "table_preview": event.get("table_preview"),
                            "total_duration_ms": event.get("total_duration_ms"),
                            "show_charts": event.get("show_charts", False),
                            "clarification_required": clarification_required,
                            "clarification_candidates": event.get("clarification_candidates"),
                            "memory_action_result": memory_action_result or memory_effect_payload,
                            "memory_suggestion": memory_suggestion,
                        },
                    )
                    yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"

        except Exception as e:
            _log_exception_event(
                "chat.stream.error",
                error=e,
                request_id=request_id,
                session_id=session_id,
                history_count=len(history),
                current_question=_short_text(request.message),
            )
            error_data = _build_error_payload(str(e), code="CHAT_STREAM_ERROR", request_id=request_id)
            error_data["type"] = "error"
            error_data["session_id"] = session_id
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """AI chat endpoint returning query params (non-streaming)."""
    request_id = uuid4().hex
    context = _prepare_chat_context(request)
    session_id = context["session_id"]
    history = context["history"]
    effective_message = context["effective_message"]
    alias_memory = context["alias_memory"]
    learned_aliases = context["learned_aliases"]
    user_id = chat_memory_service.resolve_chat_user_id(request.user_id, session_id)
    intent_type = chat_memory_service.classify_chat_intent(request.message)
    memory_action_result = None
    memory_suggestion = None
    chat_memory_service.record_chat_message(
        session_id=session_id,
        user_id=user_id,
        role="user",
        message=request.message,
        intent_type=intent_type,
    )
    _log_structured(
        "chat.request",
        request_id=request_id,
        session_id=session_id,
        user_id=user_id,
        intent_type=intent_type,
        history_count=len(history),
        history_used=min(len(history), CHAT_HISTORY_LIMIT),
        current_question=_short_text(effective_message),
        learned_alias_count=len(learned_aliases),
        session_alias_count=len(alias_memory or {}),
    )

    try:
        handled = chat_memory_service.handle_memory_command(
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            message=request.message,
            alias_memory=alias_memory,
            alias_confirmation=request.alias_confirmation,
        )
        if handled:
            memory_action_result = handled.get("memory_action_result")
            follow_up = memory_action_result.get("follow_up") if isinstance(memory_action_result, dict) else None
            if isinstance(follow_up, dict):
                effective_message = chat_memory_service.apply_query_once_alias(
                    {
                        "action": "query_once",
                        "alias_text": follow_up.get("alias_text"),
                        "canonical_text": follow_up.get("canonical_text"),
                        "original_question": follow_up.get("original_question"),
                    },
                    follow_up.get("original_question") or effective_message,
                )
                intent_type = "data_query"
            else:
                chat_memory_service.record_chat_message(
                    session_id=session_id,
                    user_id=user_id,
                    role="assistant",
                    message=handled.get("response") or "",
                    intent_type="memory_command",
                )
                return {
                    **handled,
                    "session_id": session_id,
                    "original_question": effective_message,
                    "intent_type": handled.get("intent_type") or intent_type,
                }

        effective_message = chat_memory_service.apply_query_once_alias(request.alias_confirmation, effective_message)
        memory_apply = chat_memory_service.apply_user_memories(
            effective_message,
            user_id=user_id,
            alias_memory=alias_memory,
        )
        effective_message = memory_apply.get("effective_message") or effective_message
        alias_memory = memory_apply.get("alias_memory") or alias_memory
        memory_effect_payload = chat_memory_service.build_memory_effect_payload(
            memory_apply.get("applied_items") or [],
            memory_apply.get("invalid_items") or [],
        )
        if memory_action_result is None:
            memory_action_result = memory_effect_payload

        agent = _create_chat_agent(alias_memory=alias_memory)

        steps = []
        response_text = ""
        query_params = None
        last_sensor_result = None
        projects_list = None
        devices_list = None
        final_analysis = None
        final_table_type = ""
        total_duration_ms = None
        clarification_required = False
        clarification_candidates = None

        message_with_history = _build_message_with_history(effective_message, history)

        for event in agent.run_with_progress(message_with_history):
            event_type = event.get("type")

            if event_type == "step_start":
                step = {"step": event["step"], "status": "running"}
                if event.get("timestamp_ms") is not None:
                    step["timestamp_ms"] = event.get("timestamp_ms")
                steps.append(step)

            elif event_type == "step_done":
                for s in steps:
                    if s["step"] == event["step"]:
                        s["status"] = "done"
                        s["info"] = event.get("info", "")
                        if event.get("duration_ms") is not None:
                            s["duration_ms"] = event.get("duration_ms")
                        query_info = event.get("query_info")
                        if query_info:
                            s["query_info"] = query_info
                            if isinstance(query_info, dict):
                                if query_info.get("query"):
                                    last_sensor_result = query_info
                                project_rows = query_info.get("projects")
                                if isinstance(project_rows, list):
                                    projects_list = project_rows
                                device_rows = query_info.get("devices")
                                if isinstance(device_rows, list):
                                    devices_list = device_rows[:50]

            elif event_type == "final_answer":
                response_text = event.get("response", "")
                response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
                final_analysis = event.get("analysis")
                final_table_type = str(event.get("table_type") or final_table_type or "")
                total_duration_ms = event.get("total_duration_ms")
                clarification_required = bool(event.get("clarification_required", False))
                clarification_candidates = event.get("clarification_candidates")
                if event.get("query_params"):
                    query_params = event.get("query_params")
                if event.get("projects") is not None:
                    projects_list = event.get("projects")
                if event.get("devices") is not None:
                    devices_list = event.get("devices")

        resolved_scope = _build_resolved_scope(
            query_params,
            alias_memory=alias_memory,
            learned_aliases=learned_aliases,
        )

        if not query_params and last_sensor_result and last_sensor_result.get("query"):
            mongo_query = last_sensor_result["query"]
            device_codes = mongo_query.get("device", {}).get("$in", [])
            log_time = mongo_query.get("logTime", {})
            start_time_str = log_time.get("$gte", "")
            end_time_str = log_time.get("$lte", "")
            data_type = _infer_data_type_from_context(request.message, last_sensor_result)

            tg_query = mongo_query.get("tg")
            tg_values = []
            if isinstance(tg_query, dict):
                tg_values = [str(value) for value in tg_query.get("$in", []) if str(value).strip()]
            elif isinstance(tg_query, list):
                tg_values = [str(value) for value in tg_query if str(value).strip()]
            elif tg_query not in (None, ""):
                tg_values = [str(tg_query)]

            query_params = {
                "device_codes": device_codes,
                "tg_values": tg_values,
                "start_time": start_time_str.split(" ")[0] if start_time_str else "",
                "end_time": end_time_str.split(" ")[0] if end_time_str else "",
                "data_type": data_type,
                "page": 1,
                "page_size": 50,
            }
            _log_structured("chat.query_params.derived", request_id=request_id, query_params=query_params)
            resolved_scope = _build_resolved_scope(
                query_params,
                alias_memory=alias_memory,
                learned_aliases=learned_aliases,
            )

        table_keywords = ["表格", "下表", "列表", "记录", "结果"]
        if final_table_type in {"projects", "project_stats", "devices"} or any(keyword in response_text for keyword in table_keywords):
            if final_table_type == "projects" or any(word in response_text for word in ["项目", "project"]):
                if not projects_list:
                    try:
                        projects_list = metadata_engine.list_projects()
                    except Exception as e:
                        logger.warning("chat.project_list_fallback.failed request_id=%s error=%s", request_id, e)

            if final_table_type == "devices" or any(word in response_text for word in ["设备", "device"]):
                if not devices_list:
                    try:
                        search_keywords = _extract_device_search_keywords_from_message(request.message)
                        if search_keywords:
                            from src.tools.device_tool import find_device_metadata_with_engine

                            for keyword in search_keywords:
                                devices_result = find_device_metadata_with_engine(keyword, metadata_engine)
                                if devices_result and isinstance(devices_result, list):
                                    devices_list = [
                                        d for d in devices_result
                                        if "_query_info" not in d and "error" not in d
                                    ][:50]
                                    if devices_list:
                                        break
                    except Exception as e:
                        logger.warning("chat.device_list_fallback.failed request_id=%s error=%s", request_id, e)

        memory_suggestion = chat_memory_service.build_memory_suggestion(
            request.message,
            user_id=user_id,
            alias_memory=alias_memory,
            query_params=query_params,
            clarification_required=clarification_required,
        )
        _log_structured(
            "chat.final",
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            intent_type=intent_type,
            query_params=query_params,
            resolved_scope_count=len((resolved_scope or {}).get("items") or []),
            project_count=len(projects_list or []),
            listed_device_count=len(devices_list or []),
            analysis_mode=final_analysis.get("mode") if isinstance(final_analysis, dict) else None,
            clarification_required=clarification_required,
            clarification_group_count=len(clarification_candidates or []),
            response_preview=_short_text(response_text),
            total_duration_ms=total_duration_ms,
            empty_result=response_text == EMPTY_RESULT_MESSAGE,
        )
        chat_memory_service.record_chat_message(
            session_id=session_id,
            user_id=user_id,
            role="assistant",
            message=response_text,
            intent_type=intent_type,
            message_meta={
                "response": response_text,
                "original_question": effective_message,
                "intent_type": intent_type,
                "query_params": query_params,
                "resolved_scope": resolved_scope,
                "projects": projects_list,
                "devices": devices_list,
                "analysis": final_analysis,
                "total_duration_ms": total_duration_ms,
                "clarification_required": clarification_required,
                "clarification_candidates": clarification_candidates,
                "memory_action_result": memory_action_result,
                "memory_suggestion": memory_suggestion,
            },
        )
        return {
            "success": True,
            "response": response_text,
            "original_question": effective_message,
            "session_id": session_id,
            "steps": steps,
            "query_params": query_params,
            "resolved_scope": resolved_scope,
            "projects": projects_list,
            "devices": devices_list,
            "total_duration_ms": total_duration_ms,
            "clarification_required": clarification_required,
            "clarification_candidates": clarification_candidates,
            "request_id": request_id,
            "intent_type": intent_type,
            "memory_action_result": memory_action_result,
            "memory_suggestion": memory_suggestion,
        }

    except Exception as e:
        _log_exception_event(
            "chat.error",
            error=e,
            request_id=request_id,
            session_id=session_id,
            history_count=len(history),
            current_question=_short_text(request.message),
        )
        return _build_error_payload(str(e), code="CHAT_ERROR", request_id=request_id)


@app.post("/api/query")
async def query_data(request: DataQueryRequest):
    """Direct data query endpoint with pagination."""
    request_id = uuid4().hex
    _log_structured(
        "query.request",
        request_id=request_id,
        device_count=len(request.device_codes),
        device_codes=_sample_items(request.device_codes, limit=8),
        tg_count=len(request.tg_values or []),
        tg_values=_sample_items(request.tg_values or [], limit=8),
        start_time=request.start_time,
        end_time=request.end_time,
        data_type=request.data_type,
        page=request.page,
        page_size=request.page_size,
        value_filter=request.value_filter,
        user_query=_short_text(request.user_query),
        query_plan_mode=(request.query_plan or {}).get("query_mode") if isinstance(request.query_plan, dict) else None,
    )

    try:
        if isinstance(request.comparison_scope_groups, dict) and request.comparison_scope_groups:
            result = _build_comparison_query_result(request)
        else:
            result = fetch_sensor_data_with_components(
                device_codes=request.device_codes,
                tg_values=request.tg_values,
                start_time=request.start_time,
                end_time=request.end_time,
                data_fetcher=data_fetcher,
                compressor=None,
                data_type=request.data_type,
                page=request.page,
                page_size=request.page_size,
                output_format="json",
                user_query=request.user_query,
                query_plan=request.query_plan,
                use_aggregation=False,
                value_filter=request.value_filter,
            )

        table_data = []
        is_aggregated = result.get("is_aggregated", False)

        if result.get("success") and result.get("data"):
            try:
                raw_data = result["data"]
                if isinstance(raw_data, str):
                    try:
                        raw_data = json.loads(raw_data)
                    except json.JSONDecodeError:
                        try:
                            raw_data = ast.literal_eval(raw_data)
                        except (ValueError, SyntaxError):
                            raw_data = []

                for item in raw_data:
                    if isinstance(item, dict):
                        if is_aggregated or "average" in item or "total" in item:
                            time_field = item.get("date") or item.get("month") or item.get("year") or item.get("time") or "-"
                            value_field = item.get("average") or item.get("total") or item.get("diff") or item.get("max") or item.get("min") or ""
                            table_data.append(
                                {
                                    "time": time_field,
                                    "device": item.get("device", "-"),
                                    "tag": item.get("tag", "-"),
                                    "value": round(value_field, 2) if isinstance(value_field, float) else value_field,
                                    "record_count": item.get("record_count", ""),
                                }
                            )
                        else:
                            table_data.append(
                                {
                                    "time": item.get("logTime") or item.get("time", ""),
                                    "device": item.get("device", "-"),
                                    "tag": item.get("tag", "-"),
                                    "value": item.get("val") if item.get("val") is not None else "",
                                }
                            )
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        table_data.append(
                            {
                                "time": item[0],
                                "device": "-",
                                "tag": "-",
                                "value": item[1],
                            }
                        )
            except Exception as e:
                logger.exception(
                    "query.parse.error %s",
                    json.dumps(
                        {
                            "event": "query.parse.error",
                            "request_id": request_id,
                            "error": str(e),
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                )

        response_payload = {
            "success": result.get("success", False),
            "data": table_data,
            "total_count": result.get("total_count", 0),
            "page": result.get("page", 1),
            "page_size": result.get("page_size", 0),
            "total_pages": result.get("total_pages", 1),
            "has_more": result.get("has_more", False),
            "statistics": result.get("statistics"),
            "analysis": result.get("analysis"),
            "focused_table": result.get("focused_table"),
            "chart_specs": result.get("chart_specs"),
            "show_charts": result.get("show_charts", False),
            "is_sampled": result.get("is_sampled", False),
            "is_aggregated": is_aggregated,
            "aggregation_type": result.get("aggregation_type"),
            "error": result.get("error"),
            "error_code": None,
            "request_id": request_id,
            "message": None,
            "empty_result": False,
        }

        if not response_payload["success"]:
            response_payload = _build_error_payload(
                response_payload.get("error") or "查询失败",
                code="QUERY_FAILED",
                request_id=request_id,
            )
        elif _is_empty_query_result(result, table_data):
            response_payload = _build_empty_result_payload(
                request_id=request_id,
                data=[],
                total_count=result.get("total_count", 0),
                page=result.get("page", 1),
                page_size=result.get("page_size", 0),
                total_pages=result.get("total_pages", 1),
                has_more=result.get("has_more", False),
                statistics=result.get("statistics"),
                analysis=result.get("analysis"),
                chart_specs=result.get("chart_specs"),
                show_charts=result.get("show_charts", False),
                is_sampled=result.get("is_sampled", False),
                is_aggregated=is_aggregated,
                aggregation_type=result.get("aggregation_type"),
            )

        analysis = response_payload.get("analysis") or {}
        chart_specs = response_payload.get("chart_specs") or []
        _log_structured(
            "query.response",
            request_id=request_id,
            success=response_payload.get("success"),
            total_count=response_payload.get("total_count"),
            table_rows=len(response_payload.get("data") or []),
            page=response_payload.get("page"),
            page_size=response_payload.get("page_size"),
            total_pages=response_payload.get("total_pages"),
            has_more=response_payload.get("has_more"),
            analysis_mode=analysis.get("mode") if isinstance(analysis, dict) else None,
            chart_count=len(chart_specs) if isinstance(chart_specs, list) else 0,
            is_sampled=response_payload.get("is_sampled"),
            is_aggregated=response_payload.get("is_aggregated"),
            error=response_payload.get("error"),
            error_code=response_payload.get("error_code"),
            empty_result=response_payload.get("empty_result", False),
        )
        return response_payload

    except Exception as e:
        _log_exception_event(
            "query.error",
            error=e,
            request_id=request_id,
            device_codes=_sample_items(request.device_codes, limit=8),
            start_time=request.start_time,
            end_time=request.end_time,
            data_type=request.data_type,
            page=request.page,
            page_size=request.page_size,
        )
        return _build_error_payload(str(e), code="QUERY_ERROR", request_id=request_id)


@app.get("/api/projects")
async def list_projects():
    """获取项目列表"""
    request_id = uuid4().hex
    _log_structured("projects.request", request_id=request_id)
    try:
        projects = metadata_engine.list_projects()
        _log_structured("projects.response", request_id=request_id, success=True, project_count=len(projects or []))
        return _build_success_payload(request_id=request_id, projects=projects)
    except Exception as e:
        _log_exception_event("projects.error", error=e, request_id=request_id)
        return _build_error_payload(str(e), code="PROJECTS_ERROR", request_id=request_id)


@app.get("/api/devices")
async def search_devices(keyword: str = ""):
    """搜索设备"""
    request_id = uuid4().hex
    _log_structured("devices.request", request_id=request_id, keyword=_short_text(keyword))
    try:
        if not keyword:
            _log_structured("devices.response", request_id=request_id, success=True, keyword="", device_count=0, resolver_used=False)
            return _build_success_payload(request_id=request_id, devices=[])

        try:
            resolution_result = entity_resolver.search_device_candidates(keyword, top_k=50)
            devices = resolution_result.to_dict_list()
            _log_structured(
                "devices.response",
                request_id=request_id,
                success=True,
                keyword=_short_text(keyword),
                device_count=len(devices[:50]),
                resolver_used=True,
                query_info=resolution_result.query_info,
            )
            return _build_success_payload(request_id=request_id, devices=devices[:50], query_info=resolution_result.query_info)
        except Exception as resolver_exc:
            logger.warning(
                "api.search_devices.entity_resolver_failed %s",
                json.dumps(
                    {
                        "event": "devices.entity_resolver_failed",
                        "request_id": request_id,
                        "keyword": _short_text(keyword),
                        "error": str(resolver_exc),
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            )

        from src.tools.device_tool import find_device_metadata_with_engine
        devices = find_device_metadata_with_engine(keyword, metadata_engine)
        devices = [d for d in devices if "_query_info" not in d and "error" not in d]

        _log_structured(
            "devices.response",
            request_id=request_id,
            success=True,
            keyword=_short_text(keyword),
            device_count=len(devices[:50]),
            resolver_used=False,
        )
        return _build_success_payload(request_id=request_id, devices=devices[:50])
    except Exception as e:
        _log_exception_event("devices.error", error=e, request_id=request_id, keyword=_short_text(keyword))
        return _build_error_payload(str(e), code="DEVICES_ERROR", request_id=request_id)


@app.get("/api/version")
async def get_version():
    """获取当前版本与更新摘要"""
    payload = build_version_payload()
    payload["timestamp"] = datetime.now().isoformat()
    return payload


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
