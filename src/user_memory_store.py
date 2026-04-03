from __future__ import annotations

import os
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

DEFAULT_MEMORY_DB_PATH = Path(os.getenv("APP_MEMORY_DB_PATH", "data/app_memory.sqlite3"))


class UserMemoryStore:
    def __init__(self, db_path: Path | str = DEFAULT_MEMORY_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._lock = Lock()
        self._ensure_db()

    @staticmethod
    def _normalize_text(value: Any) -> str:
        text = str(value or "").strip().lower()
        text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        return re.sub(r"\s+", "", text)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        message TEXT NOT NULL,
                        intent_type TEXT NOT NULL DEFAULT '',
                        message_meta TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL
                    )
                    """
                )
                existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(chat_messages)").fetchall()}
                if "message_meta" not in existing_columns:
                    conn.execute("ALTER TABLE chat_messages ADD COLUMN message_meta TEXT NOT NULL DEFAULT ''")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created ON chat_messages(session_id, created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_user_created ON chat_messages(user_id, created_at)")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_alias_memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        alias_text TEXT NOT NULL,
                        normalized_alias_text TEXT NOT NULL,
                        canonical_text TEXT NOT NULL,
                        normalized_canonical_text TEXT NOT NULL,
                        target_type TEXT NOT NULL,
                        target_value TEXT NOT NULL,
                        scope_type TEXT NOT NULL,
                        scope_value TEXT NOT NULL,
                        normalized_scope_value TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'enabled',
                        source TEXT NOT NULL DEFAULT 'chat_command',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_user_alias_memories_lookup ON user_alias_memories(user_id, normalized_alias_text, status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_user_alias_memories_canonical ON user_alias_memories(user_id, normalized_canonical_text, status)")
                conn.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS uq_user_alias_memories_active
                    ON user_alias_memories(
                        user_id,
                        normalized_alias_text,
                        scope_type,
                        normalized_scope_value,
                        status
                    )
                    """
                )
                conn.commit()

    def record_chat_message(
        self,
        *,
        session_id: str,
        user_id: str,
        role: str,
        message: str,
        intent_type: str,
        message_meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        now = datetime.now().isoformat()
        serialized_meta = json.dumps(message_meta or {}, ensure_ascii=False) if isinstance(message_meta, dict) and message_meta else ""
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO chat_messages(session_id, user_id, role, message, intent_type, message_meta, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(session_id or "").strip(),
                        str(user_id or "").strip(),
                        str(role or "").strip(),
                        str(message or ""),
                        str(intent_type or ""),
                        serialized_meta,
                        now,
                    ),
                )
                conn.commit()
                return int(cursor.lastrowid)

    @staticmethod
    def _summarize_session_title(message: Any, limit: int = 24) -> str:
        text = re.sub(r"\s+", " ", str(message or "")).strip()
        if not text:
            return "新对话"
        return text if len(text) <= limit else text[: max(limit - 1, 1)] + "…"

    def list_chat_sessions(
        self,
        *,
        user_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            return []
        page_limit = max(1, min(int(limit or 50), 200))
        query = """
            SELECT session_id, user_id, role, message, intent_type, message_meta, created_at, id
            FROM chat_messages
            WHERE user_id = ?
            ORDER BY created_at ASC, id ASC
        """
        with self._connect() as conn:
            rows = conn.execute(query, (normalized_user_id,)).fetchall()
        sessions: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            item = dict(row)
            session_id = str(item.get("session_id") or "").strip()
            if not session_id:
                continue
            session = sessions.setdefault(
                session_id,
                {
                    "session_id": session_id,
                    "user_id": normalized_user_id,
                    "title": "新对话",
                    "first_user_message": "",
                    "last_message": "",
                    "last_message_at": item.get("created_at"),
                    "message_count": 0,
                },
            )
            session["message_count"] = int(session.get("message_count") or 0) + 1
            session["last_message"] = str(item.get("message") or "")
            session["last_message_at"] = item.get("created_at")
            if not session.get("first_user_message") and str(item.get("role") or "") == "user":
                first_user_message = str(item.get("message") or "")
                session["first_user_message"] = first_user_message
                session["title"] = self._summarize_session_title(first_user_message)
        ordered = sorted(
            sessions.values(),
            key=lambda item: (str(item.get("last_message_at") or ""), int(item.get("message_count") or 0)),
            reverse=True,
        )
        return ordered[:page_limit]

    def list_chat_history(
        self,
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: List[Any] = []
        if session_id:
            clauses.append("session_id = ?")
            params.append(str(session_id).strip())
        if user_id:
            clauses.append("user_id = ?")
            params.append(str(user_id).strip())
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        page_limit = max(1, min(int(limit or 50), 200))
        query = f"""
            SELECT id, session_id, user_id, role, message, intent_type, message_meta, created_at
            FROM chat_messages
            {where_sql}
            ORDER BY created_at DESC, id DESC
            LIMIT ?
        """
        params.append(page_limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        items: List[Dict[str, Any]] = []
        for row in reversed(rows):
            item = dict(row)
            raw_meta = str(item.get("message_meta") or "").strip()
            if raw_meta:
                try:
                    item["message_meta"] = json.loads(raw_meta)
                except Exception:
                    item["message_meta"] = None
            else:
                item["message_meta"] = None
            items.append(item)
        return items

    def clear_chat_history(
        self,
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> int:
        clauses = []
        params: List[Any] = []
        if session_id:
            clauses.append("session_id = ?")
            params.append(str(session_id).strip())
        if user_id:
            clauses.append("user_id = ?")
            params.append(str(user_id).strip())
        if not clauses:
            raise ValueError("session_id 或 user_id 至少提供一个")
        where_sql = f"WHERE {' AND '.join(clauses)}"
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(f"DELETE FROM chat_messages {where_sql}", params)
                conn.commit()
                return int(cursor.rowcount or 0)

    def upsert_alias_memory(
        self,
        *,
        user_id: str,
        alias_text: str,
        canonical_text: str,
        target_type: str,
        target_value: str,
        scope_type: str,
        scope_value: str,
        source: str = "chat_command",
    ) -> Dict[str, Any]:
        alias_text = str(alias_text or "").strip()
        canonical_text = str(canonical_text or "").strip()
        target_type = str(target_type or "canonical_term").strip()
        target_value = str(target_value or canonical_text).strip()
        scope_type = str(scope_type or "global").strip().lower()
        scope_value = str(scope_value or ("global" if scope_type == "global" else "")).strip()
        user_id = str(user_id or "").strip()
        if not user_id:
            raise ValueError("user_id 不能为空")
        if not alias_text:
            raise ValueError("alias_text 不能为空")
        if not canonical_text:
            raise ValueError("canonical_text 不能为空")
        if scope_type not in {"project", "global"}:
            raise ValueError("scope_type 仅支持 project / global")
        now = datetime.now().isoformat()
        normalized_alias = self._normalize_text(alias_text)
        normalized_canonical = self._normalize_text(canonical_text)
        normalized_scope_value = self._normalize_text(scope_value if scope_type == "project" else "global")
        with self._lock:
            with self._connect() as conn:
                existing = conn.execute(
                    """
                    SELECT *
                    FROM user_alias_memories
                    WHERE user_id = ?
                      AND normalized_alias_text = ?
                      AND scope_type = ?
                      AND normalized_scope_value = ?
                      AND status = 'enabled'
                    ORDER BY updated_at DESC, id DESC
                    LIMIT 1
                    """,
                    (user_id, normalized_alias, scope_type, normalized_scope_value),
                ).fetchone()
                if existing:
                    conn.execute(
                        """
                        UPDATE user_alias_memories
                        SET canonical_text = ?,
                            normalized_canonical_text = ?,
                            target_type = ?,
                            target_value = ?,
                            source = ?,
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (canonical_text, normalized_canonical, target_type, target_value, source, now, existing["id"]),
                    )
                    row_id = int(existing["id"])
                else:
                    cursor = conn.execute(
                        """
                        INSERT INTO user_alias_memories(
                            user_id, alias_text, normalized_alias_text, canonical_text, normalized_canonical_text,
                            target_type, target_value, scope_type, scope_value, normalized_scope_value,
                            status, source, created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'enabled', ?, ?, ?)
                        """,
                        (
                            user_id,
                            alias_text,
                            normalized_alias,
                            canonical_text,
                            normalized_canonical,
                            target_type,
                            target_value,
                            scope_type,
                            scope_value if scope_type == "project" else "global",
                            normalized_scope_value,
                            source,
                            now,
                            now,
                        ),
                    )
                    row_id = int(cursor.lastrowid)
                conn.commit()
        return self.get_alias_memory(row_id=row_id, user_id=user_id)

    def get_alias_memory(self, *, row_id: int, user_id: Optional[str] = None) -> Dict[str, Any]:
        clauses = ["id = ?"]
        params: List[Any] = [int(row_id)]
        if user_id:
            clauses.append("user_id = ?")
            params.append(str(user_id).strip())
        query = f"SELECT * FROM user_alias_memories WHERE {' AND '.join(clauses)} LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        if not row:
            raise KeyError("memory_not_found")
        return dict(row)

    def list_alias_memories(
        self,
        *,
        user_id: str,
        keyword: str = "",
        canonical_text: str = "",
        scope_type: str = "",
        status: str = "enabled",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        clauses = ["user_id = ?"]
        params: List[Any] = [str(user_id or "").strip()]
        if status:
            clauses.append("status = ?")
            params.append(str(status).strip())
        if scope_type:
            clauses.append("scope_type = ?")
            params.append(str(scope_type).strip().lower())
        if keyword:
            clauses.append("(normalized_alias_text LIKE ? OR normalized_canonical_text LIKE ? OR target_value LIKE ?)")
            normalized_keyword = f"%{self._normalize_text(keyword)}%"
            params.extend([normalized_keyword, normalized_keyword, f"%{str(keyword).strip()}%"])
        if canonical_text:
            clauses.append("normalized_canonical_text = ?")
            params.append(self._normalize_text(canonical_text))
        page_limit = max(1, min(int(limit or 100), 200))
        query = f"""
            SELECT *
            FROM user_alias_memories
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
        """
        params.append(page_limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def update_alias_memory(
        self,
        *,
        memory_id: int,
        user_id: str,
        alias_text: Optional[str] = None,
        canonical_text: Optional[str] = None,
        scope_type: Optional[str] = None,
        scope_value: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        existing = self.get_alias_memory(row_id=memory_id, user_id=user_id)
        next_alias_text = str(alias_text if alias_text is not None else existing.get("alias_text") or "").strip()
        next_canonical_text = str(canonical_text if canonical_text is not None else existing.get("canonical_text") or "").strip()
        next_scope_type = str(scope_type if scope_type is not None else existing.get("scope_type") or "global").strip().lower()
        next_scope_value = str(scope_value if scope_value is not None else existing.get("scope_value") or ("global" if next_scope_type == "global" else "")).strip()
        next_status = str(status if status is not None else existing.get("status") or "enabled").strip().lower()
        if next_scope_type not in {"project", "global"}:
            raise ValueError("scope_type 仅支持 project / global")
        if next_status not in {"enabled", "disabled"}:
            raise ValueError("status 仅支持 enabled / disabled")
        now = datetime.now().isoformat()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE user_alias_memories
                    SET alias_text = ?,
                        normalized_alias_text = ?,
                        canonical_text = ?,
                        normalized_canonical_text = ?,
                        scope_type = ?,
                        scope_value = ?,
                        normalized_scope_value = ?,
                        status = ?,
                        updated_at = ?
                    WHERE id = ? AND user_id = ?
                    """,
                    (
                        next_alias_text,
                        self._normalize_text(next_alias_text),
                        next_canonical_text,
                        self._normalize_text(next_canonical_text),
                        next_scope_type,
                        next_scope_value if next_scope_type == "project" else "global",
                        self._normalize_text(next_scope_value if next_scope_type == "project" else "global"),
                        next_status,
                        now,
                        int(memory_id),
                        str(user_id).strip(),
                    ),
                )
                conn.commit()
        return self.get_alias_memory(row_id=memory_id, user_id=user_id)

    def delete_alias_memory(self, *, memory_id: int, user_id: str) -> Dict[str, Any]:
        row = self.get_alias_memory(row_id=memory_id, user_id=user_id)
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM user_alias_memories WHERE id = ? AND user_id = ?", (int(memory_id), str(user_id).strip()))
                conn.commit()
        return row

    def resolve_message_memories(
        self,
        *,
        user_id: str,
        message: str,
        project_scope_value: str = "",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        normalized_message = self._normalize_text(message)
        if not user_id or not normalized_message:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM user_alias_memories
                WHERE user_id = ? AND status = 'enabled'
                ORDER BY updated_at DESC, id DESC
                """,
                (str(user_id).strip(),),
            ).fetchall()
        scored: List[Dict[str, Any]] = []
        project_key = self._normalize_text(project_scope_value or "")
        for row in rows:
            item = dict(row)
            alias_key = str(item.get("normalized_alias_text") or "")
            if not alias_key or alias_key not in normalized_message:
                continue
            scope_type = str(item.get("scope_type") or "global").strip().lower()
            scope_value_key = str(item.get("normalized_scope_value") or "")
            if scope_type == "project" and project_key and scope_value_key == project_key:
                scope_score = 200
            elif scope_type == "project":
                continue
            else:
                scope_score = 100
            item["memory_score"] = scope_score + len(alias_key)
            scored.append(item)
        scored.sort(key=lambda item: (-int(item.get("memory_score") or 0), -len(str(item.get("alias_text") or "")), -int(item.get("id") or 0)))
        return scored[: max(1, min(int(limit or 20), 50))]
