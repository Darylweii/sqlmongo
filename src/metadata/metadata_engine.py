"""
Metadata engine for SQL-backed device and project metadata.

This module provides `MetadataEngine`, which queries MySQL metadata tables and
adds lightweight in-memory caching for repeated lookups.

Expected table shape:
- `device.device_info`: `id, device_name, device_type, device, project_id, asset_number, tg`
- `project.project_info`: `id, project_name, project_code_name, enable, ...`
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Tuple
import logging
import re

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from src.exceptions import DatabaseConnectionError, MetadataEngineError


logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Device information data class."""
    device: str  # ???? (?? MongoDB ? device ??)
    name: str  # ????
    device_type: str  # ????
    project_id: str  # ??ID
    project_name: Optional[str] = None  # ????
    project_code_name: Optional[str] = None  # ????
    tg: Optional[str] = None
    score: float = 0.0
    matched_fields: List[str] = field(default_factory=list)
    match_reason: Optional[str] = None
    retrieval_source: str = "lexical"
    confidence_level: Optional[str] = None
    decision_mode: Optional[str] = None
    is_recommended: bool = False
    recommendation_rank: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "device": self.device,
            "name": self.name,
            "device_type": self.device_type,
            "project_id": self.project_id,
            "project_name": self.project_name,
            "project_code_name": self.project_code_name,
            "tg": self.tg,
            "match_score": round(float(self.score or 0.0), 2),
            "matched_fields": list(self.matched_fields or []),
            "match_reason": self.match_reason,
            "retrieval_source": self.retrieval_source,
            "confidence_level": self.confidence_level,
            "decision_mode": self.decision_mode,
            "is_recommended": bool(self.is_recommended),
            "recommendation_rank": self.recommendation_rank,
        }


class MetadataEngine:
    """
    SQL Metadata Retrieval Engine.
    
    Provides methods to search for devices and projects with LRU caching
    for optimized repeated query performance.
    
    实际表结构:
    - device.device_info: id, device_name, device_type, device, project_id, asset_number, tg
    - project.project_info: id, project_name, project_code_name, enable, ...
    """
    
    def __init__(self, db_connection_string: str, cache_size: int = 1000):
        """
        Initialize the Metadata Engine.
        
        Args:
            db_connection_string: SQLAlchemy database connection string
            cache_size: Maximum number of cached query results (default: 1000)
        """
        self._db_connection_string = db_connection_string
        self._cache_size = cache_size
        self._engine = None
        self._session_factory = None
        self._search_cache: dict = {}
        self._project_cache: dict = {}
        self._catalog_cache: Optional[List[DeviceInfo]] = None
        
    def _get_engine(self):
        """Get or create SQLAlchemy engine."""
        if self._engine is None:
            try:
                self._engine = create_engine(
                    self._db_connection_string,
                    pool_pre_ping=True,
                    pool_recycle=3600
                )
                self._session_factory = sessionmaker(bind=self._engine)
            except Exception as e:
                logger.error(f"Failed to create database engine: {e}")
                raise DatabaseConnectionError(f"数据库连接失败: {str(e)}")
        return self._engine
    
    def _get_session(self):
        """Get a new database session."""
        self._get_engine()
        return self._session_factory()

    def _normalize_match_text(self, value: Optional[str]) -> str:
        if value is None:
            return ""
        text = str(value).strip().lower()
        text = text.replace("（", "(").replace("）", ")")
        text = re.sub(r"[_\-/#()]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _compact_match_text(self, value: Optional[str]) -> str:
        return re.sub(r"\s+", "", self._normalize_match_text(value))

    def _replace_chinese_digits(self, value: str) -> str:
        mapping = str.maketrans({
            "零": "0", "一": "1", "二": "2", "两": "2", "三": "3", "四": "4",
            "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
        })
        return str(value or "").translate(mapping)

    def _replace_arabic_digits(self, value: str) -> str:
        mapping = str.maketrans({
            "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
            "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
        })
        return str(value or "").translate(mapping)

    def _extract_match_tokens(self, value: str) -> List[str]:
        normalized = self._normalize_match_text(value)
        compact = self._compact_match_text(value)
        tokens: List[str] = []
        seen = set()

        def add_token(token: str) -> None:
            text = str(token or "").strip().lower()
            if not text or text in seen:
                return
            seen.add(text)
            tokens.append(text)

        for token in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", normalized):
            add_token(token)
            add_token(self._replace_chinese_digits(token))
            add_token(self._replace_arabic_digits(token))
            if re.fullmatch(r"[\u4e00-\u9fff]{2,12}", token):
                for size in (2, 3):
                    if len(token) < size:
                        continue
                    for index in range(0, len(token) - size + 1):
                        add_token(token[index:index + size])

        add_token(compact)
        add_token(self._replace_chinese_digits(compact))
        add_token(self._replace_arabic_digits(compact))
        return tokens

    def _load_device_catalog(self) -> List[DeviceInfo]:
        if self._catalog_cache is not None:
            return self._catalog_cache

        session = self._get_session()
        try:
            query = text("""
                SELECT
                    d.device,
                    d.device_name,
                    d.device_type,
                    d.project_id,
                    d.tg,
                    p.project_name,
                    p.project_code_name
                FROM device.device_info d
                LEFT JOIN project.project_info p ON d.project_id = p.id
            """)
            result = session.execute(query)
            catalog: List[DeviceInfo] = []
            for row in result:
                if not row.device:
                    continue
                catalog.append(
                    DeviceInfo(
                        device=row.device,
                        name=row.device_name or "",
                        device_type=row.device_type or "",
                        project_id=str(row.project_id) if row.project_id else "",
                        project_name=row.project_name,
                        project_code_name=row.project_code_name,
                        tg=getattr(row, "tg", None),
                    )
                )
            self._catalog_cache = catalog
            return catalog
        finally:
            session.close()

    def _score_device_match(self, keyword: str, device: DeviceInfo) -> Tuple[float, List[str], str]:
        normalized_keyword = self._normalize_match_text(keyword)
        compact_keyword = self._compact_match_text(keyword)
        if not normalized_keyword:
            return 0.0, [], ""

        tokens = self._extract_match_tokens(normalized_keyword)
        fields = {
            "device": device.device,
            "name": device.name,
            "project_name": device.project_name,
            "project_code_name": device.project_code_name,
            "device_type": device.device_type,
            "tg": device.tg,
        }
        exact_weights = {
            "device": 120.0,
            "name": 105.0,
            "project_name": 72.0,
            "project_code_name": 64.0,
            "device_type": 36.0,
            "tg": 42.0,
        }
        contains_weights = {
            "device": 96.0,
            "name": 88.0,
            "project_name": 58.0,
            "project_code_name": 52.0,
            "device_type": 24.0,
            "tg": 30.0,
        }
        token_weights = {
            "device": 84.0,
            "name": 78.0,
            "project_name": 48.0,
            "project_code_name": 44.0,
            "device_type": 18.0,
            "tg": 24.0,
        }
        field_labels = {
            "device": "设备代号",
            "name": "设备名称",
            "project_name": "项目名称",
            "project_code_name": "项目代号",
            "device_type": "设备类型",
            "tg": "TG",
        }

        score = 0.0
        matched_fields: List[str] = []
        reasons: List[str] = []

        for field_name, raw_value in fields.items():
            normalized_value = self._normalize_match_text(raw_value)
            compact_value = self._compact_match_text(raw_value)
            if not normalized_value:
                continue

            field_score = 0.0
            field_reason = ""
            if normalized_value == normalized_keyword:
                field_score = exact_weights[field_name]
                field_reason = f"{field_labels[field_name]}精确匹配"
            elif compact_keyword and compact_keyword in compact_value:
                field_score = contains_weights[field_name]
                field_reason = f"{field_labels[field_name]}包含关键词"
            elif tokens:
                matched_tokens = [token for token in tokens if token and token in compact_value]
                coverage = len(matched_tokens) / max(len(tokens), 1)
                if coverage >= 0.78:
                    field_score = token_weights[field_name]
                    field_reason = f"{field_labels[field_name]}高覆盖关键词"
                elif coverage >= 0.5:
                    field_score = token_weights[field_name] * coverage
                    field_reason = f"{field_labels[field_name]}命中部分关键词"

            if field_score <= 0:
                continue

            score += field_score
            matched_fields.append(field_name)
            reasons.append(field_reason)

        if re.fullmatch(r"[a-zA-Z]\d*_[a-zA-Z0-9_]+", normalized_keyword or "") and normalized_keyword == self._normalize_match_text(device.device):
            score += 30.0
            if "device" not in matched_fields:
                matched_fields.insert(0, "device")
            reasons.insert(0, "设备代号格式命中")

        if len(matched_fields) > 1:
            score += min(18.0, 6.0 * (len(matched_fields) - 1))

        if score <= 0:
            return 0.0, [], ""

        return score, matched_fields, "；".join(dict.fromkeys(reasons))

    def search_devices(self, keyword: str) -> tuple[List[DeviceInfo], str]:
        """
        Search devices by keyword.

        Searches for devices where the device name or associated project name
        contains the given keyword (case-insensitive).

        Args:
            keyword: Device name or project name keyword to search for

        Returns:
            Tuple of (List of DeviceInfo objects, SQL query string)
        """
        if not keyword or not keyword.strip():
            return [], ""

        keyword = keyword.strip()

        cache_key = f"search:{keyword.lower()}"
        if cache_key in self._search_cache:
            cached = self._search_cache[cache_key]
            return cached["devices"], cached["sql"]

        try:
            catalog = self._load_device_catalog()
            deduped: dict = {}
            for catalog_device in catalog:
                score, matched_fields, match_reason = self._score_device_match(keyword, catalog_device)
                if score <= 0:
                    continue

                device = DeviceInfo(
                    device=catalog_device.device,
                    name=catalog_device.name,
                    device_type=catalog_device.device_type,
                    project_id=catalog_device.project_id,
                    project_name=catalog_device.project_name,
                    project_code_name=catalog_device.project_code_name,
                    tg=catalog_device.tg,
                    score=score,
                    matched_fields=matched_fields,
                    match_reason=match_reason,
                    retrieval_source="lexical",
                )
                dedupe_key = (device.device, device.project_id, device.name, device.tg)
                existing = deduped.get(dedupe_key)
                if existing is None or device.score > existing.score:
                    deduped[dedupe_key] = device

            devices = sorted(
                deduped.values(),
                key=lambda item: (
                    -float(item.score or 0.0),
                    str(item.device or ""),
                    str(item.name or ""),
                    str(item.project_id or ""),
                ),
            )

            actual_sql = f"catalog_search:{keyword}"
            self._update_cache(self._search_cache, cache_key, {"devices": devices, "sql": actual_sql})
            return devices, actual_sql

        except SQLAlchemyError as e:
            logger.error(f"Database query failed: {e}")
            raise DatabaseConnectionError(f"???????: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            raise MetadataEngineError(f"?????????: {str(e)}")

    def get_devices_by_project(self, project_id: str) -> List[DeviceInfo]:
        """
        Get all devices under a specific project.
        
        Args:
            project_id: Project ID to query devices for
        
        Returns:
            List of DeviceInfo objects for the specified project.
        """
        if not project_id or not str(project_id).strip():
            return []
        
        project_id = str(project_id).strip()
        
        # Check cache first
        cache_key = f"project:{project_id}"
        if cache_key in self._project_cache:
            return self._project_cache[cache_key]
        
        try:
            session = self._get_session()
            try:
                query = text("""
                    SELECT 
                        d.device,
                        d.device_name,
                        d.device_type,
                        d.project_id,
                        d.tg,
                        p.project_name,
                        p.project_code_name
                    FROM device.device_info d
                    LEFT JOIN project.project_info p ON d.project_id = p.id
                    WHERE d.project_id = :project_id
                """)
                
                result = session.execute(query, {"project_id": project_id})
                
                devices = []
                for row in result:
                    if not row.device:
                        continue
                    device = DeviceInfo(
                        device=row.device,
                        name=row.device_name or "",
                        device_type=row.device_type or "",
                        project_id=str(row.project_id) if row.project_id else "",
                        project_name=row.project_name,
                        project_code_name=row.project_code_name,
                        tg=row.tg,
                    )
                    devices.append(device)
                
                # Update cache
                self._update_cache(self._project_cache, cache_key, devices)
                
                return devices
                
            finally:
                session.close()
                
        except SQLAlchemyError as e:
            logger.error(f"Database query failed: {e}")
            raise DatabaseConnectionError(f"数据库查询失败: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error getting devices by project: {e}")
            raise MetadataEngineError(f"获取项目设备时发生错误: {str(e)}")

    def list_all_devices(self, force_refresh: bool = False) -> List[DeviceInfo]:
        """List the full device catalog used by the entity resolver."""
        if not force_refresh and self._catalog_cache is not None:
            return list(self._catalog_cache)

        try:
            session = self._get_session()
            try:
                query = text("""
                    SELECT
                        d.device,
                        d.device_name,
                        d.device_type,
                        d.project_id,
                        d.tg,
                        p.project_name,
                        p.project_code_name
                    FROM device.device_info d
                    LEFT JOIN project.project_info p ON d.project_id = p.id
                    WHERE d.device IS NOT NULL
                      AND d.device <> ''
                      AND (p.enable = 1 OR p.enable IS NULL)
                    ORDER BY d.device, d.project_id, d.device_name
                """)
                result = session.execute(query)

                deduped: dict = {}
                for row in result:
                    if not row.device:
                        continue
                    device = DeviceInfo(
                        device=row.device,
                        name=row.device_name or "",
                        device_type=row.device_type or "",
                        project_id=str(row.project_id) if row.project_id else "",
                        project_name=row.project_name,
                        project_code_name=row.project_code_name,
                        tg=row.tg,
                    )
                    dedupe_key = (device.device, device.project_id, device.name)
                    deduped[dedupe_key] = device

                catalog = list(deduped.values())
                self._catalog_cache = catalog
                return list(catalog)

            finally:
                session.close()

        except SQLAlchemyError as e:
            logger.error(f"Database query failed: {e}")
            raise DatabaseConnectionError(f"数据库查询失败: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error listing all devices: {e}")
            raise MetadataEngineError(f"获取设备目录时发生错误: {str(e)}")
    
    def list_projects(self) -> List[dict]:
        """
        List all available projects.
        
        Returns:
            List of project dictionaries with id, project_name, project_code_name
        """
        try:
            session = self._get_session()
            try:
                query = text("""
                    SELECT id, project_name, project_code_name 
                    FROM project.project_info 
                    WHERE enable = 1
                    ORDER BY id
                """)
                result = session.execute(query)
                
                projects = []
                for row in result:
                    projects.append({
                        "id": row.id,
                        "project_name": row.project_name,
                        "code_name": row.project_code_name
                    })
                return projects
                
            finally:
                session.close()
                
        except SQLAlchemyError as e:
            logger.error(f"Database query failed: {e}")
            raise DatabaseConnectionError(f"数据库查询失败: {str(e)}")
    
    def get_project_device_stats(self) -> List[dict]:
        """
        Get device count statistics for all projects.
        
        Returns:
            List of project dictionaries with id, project_name, device_count, sorted by device_count desc
        """
        try:
            session = self._get_session()
            try:
                query = text("""
                    SELECT 
                        p.id,
                        p.project_name,
                        p.project_code_name,
                        COUNT(d.id) as device_count
                    FROM project.project_info p
                    LEFT JOIN device.device_info d ON p.id = d.project_id
                    WHERE p.enable = 1
                    GROUP BY p.id, p.project_name, p.project_code_name
                    ORDER BY device_count DESC
                """)
                result = session.execute(query)
                
                stats = []
                for row in result:
                    stats.append({
                        "id": row.id,
                        "project_name": row.project_name,
                        "code_name": row.project_code_name,
                        "device_count": row.device_count
                    })
                return stats
                
            finally:
                session.close()
                
        except SQLAlchemyError as e:
            logger.error(f"Database query failed: {e}")
            raise DatabaseConnectionError(f"数据库查询失败: {str(e)}")
    
    def _update_cache(self, cache: dict, key: str, value: Any) -> None:
        """Update cache with LRU eviction policy."""
        if len(cache) >= self._cache_size:
            oldest_key = next(iter(cache))
            del cache[oldest_key]
        cache[key] = value
    
    def clear_cache(self) -> None:
        """Clear all cached query results."""
        self._search_cache.clear()
        self._project_cache.clear()
        self._catalog_cache = None
        logger.info("Metadata cache cleared")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "search_cache_size": len(self._search_cache),
            "project_cache_size": len(self._project_cache),
            "catalog_cache_size": len(self._catalog_cache or []),
            "max_cache_size": self._cache_size
        }
