from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


DEVICE_CODE_PATTERN = re.compile(r"[a-zA-Z]\d*_[a-zA-Z0-9_]+", re.IGNORECASE)


@dataclass
class MemoryRewriteResult:
    alias_text: str
    target_text: str
    source: str = "rule"
    confidence: str = "high"
    reversed_mapping: bool = False
    raw_alias_text: str = ""
    raw_target_text: str = ""

    def to_command(self) -> Dict[str, Any]:
        return {
            "intent": "create_alias_memory",
            "alias_text": self.alias_text,
            "target_text": self.target_text,
            "rewrite_source": self.source,
            "rewrite_confidence": self.confidence,
            "reversed_mapping": self.reversed_mapping,
            "raw_alias_text": self.raw_alias_text or self.alias_text,
            "raw_target_text": self.raw_target_text or self.target_text,
        }


class MemoryCommandRewriter:
    COMMAND_FILLERS = [
        "请帮我记住以后查询",
        "请帮我记住以后",
        "请帮我记住",
        "帮我记住以后查询",
        "帮我记住以后",
        "帮我记住",
        "帮忙记住",
        "帮我记一下",
        "帮忙记一下",
        "请记住以后查询",
        "请记住以后",
        "请记住",
        "以后记住",
        "以后查询",
        "记一下",
        "查询",
        "以后",
        "请",
    ]
    TARGET_FILLERS = [
        "的意思",
        "这个意思",
        "这个叫法",
        "这个名称",
        "这个说法",
    ]

    def __init__(
        self,
        *,
        normalize_alias_key: Callable[[str], str],
        llm_rewrite: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    ) -> None:
        self.normalize_alias_key = normalize_alias_key
        self.llm_rewrite = llm_rewrite
        self.add_patterns = [
            re.compile(r"^(?:\u8bf7)?(?:\u628a)?(?P<alias>.+?)\u6dfb\u52a0\u4e3a(?P<target>.+?)\u7684\u5e38\u7528\u53eb\u6cd5[\uff1f?]?$", re.IGNORECASE),
            re.compile(r"^(?:\u8bf7)?(?:\u628a)?(?P<alias>.+?)\u8bb0\u6210(?P<target>.+?)[\uff1f?]?$", re.IGNORECASE),
            re.compile(r"^(?:\u8bf7)?(?:\u5e2e\u6211|\u5e2e\u5fd9)?(?:\u8bb0\u4e00\u4e0b|\u8bb0\u4f4f)(?P<alias>.+?)(?:\u5176\u5b9e)?(?:\u4ee3\u8868|\u5c31\u662f|\u53ef\u4ee5\u53eb|\u53eb\u505a|\u53eb)(?P<target>.+?)[\uff1f?]?$", re.IGNORECASE),
            re.compile(r"^(?:\u8bf7)?(?:\u4ee5\u540e)?\u8bb0\u4f4f(?P<alias>.+?)(?:\u4ee3\u8868|\u5c31\u662f|\u53ef\u4ee5\u53eb|\u53eb\u505a|\u53eb)(?P<target>.+?)[\uff1f?]?$", re.IGNORECASE),
            re.compile(r"^(?:\u4ee5\u540e)?(?P<alias>.+?)\u5c31\u662f(?P<target>.+?)[\uff1f?]?$", re.IGNORECASE),
        ]

    def clean_text(self, value: Any) -> str:
        return str(value or "").strip().strip('"').strip("'").replace("“", "").replace("”", "")

    def normalize_phrase(self, value: Any, *, is_target: bool = False) -> str:
        text = self.clean_text(value)
        if not text:
            return ""
        changed = True
        while changed:
            changed = False
            for filler in self.COMMAND_FILLERS:
                if text.startswith(filler) and len(text) > len(filler):
                    text = text[len(filler):].strip()
                    changed = True
        if is_target:
            for filler in self.TARGET_FILLERS:
                if text.endswith(filler) and len(text) > len(filler):
                    text = text[:-len(filler)].strip()
        text = re.sub(r"^(?:\u628a|\u5c06)", "", text).strip()
        text = re.sub(r"(?:\u5176\u5b9e)$", "", text).strip()
        return text.strip("\uFF0C\u3002,.\uFF1A:?\uFF1B;\"'\u201c\u201d")

    def is_device_code(self, value: str) -> bool:
        return bool(DEVICE_CODE_PATTERN.fullmatch(str(value or "").strip()))

    def is_memory_command_message(self, message: str) -> bool:
        normalized = str(message or "").strip()
        if not normalized:
            return False
        keywords = [
            "记住",
            "记一下",
            "常用叫法",
            "以后",
            "代表",
            "叫做",
            "可以叫",
            "记成",
            "就是",
        ]
        return any(keyword in normalized for keyword in keywords)

    def _finalize_result(self, alias_text: str, target_text: str, *, source: str, confidence: str) -> Optional[MemoryRewriteResult]:
        normalized_alias = self.normalize_phrase(alias_text)
        normalized_target = self.normalize_phrase(target_text, is_target=True)
        if not normalized_alias or not normalized_target:
            return None
        reversed_mapping = False
        if self.is_device_code(normalized_alias) and not self.is_device_code(normalized_target):
            normalized_alias, normalized_target = normalized_target, normalized_alias
            reversed_mapping = True
        return MemoryRewriteResult(
            alias_text=normalized_alias,
            target_text=normalized_target,
            source=source,
            confidence=confidence,
            reversed_mapping=reversed_mapping,
            raw_alias_text=self.clean_text(alias_text),
            raw_target_text=self.clean_text(target_text),
        )

    def rewrite_with_rules(self, message: str) -> Optional[MemoryRewriteResult]:
        content = str(message or "").strip()
        for pattern in self.add_patterns:
            match = pattern.search(content)
            if not match:
                continue
            result = self._finalize_result(
                str(match.group("alias") or ""),
                str(match.group("target") or ""),
                source="rule",
                confidence="high",
            )
            if result is not None:
                return result
        return None

    def rewrite_with_llm(self, message: str) -> Optional[MemoryRewriteResult]:
        if not callable(self.llm_rewrite):
            return None
        payload = self.llm_rewrite(str(message or "").strip())
        if not isinstance(payload, dict):
            return None
        alias_text = str(payload.get("alias_text") or "").strip()
        target_text = str(payload.get("target_text") or "").strip()
        if not alias_text or not target_text:
            return None
        confidence = str(payload.get("confidence") or "medium").strip().lower() or "medium"
        source = str(payload.get("source") or "llm").strip() or "llm"
        return self._finalize_result(alias_text, target_text, source=source, confidence=confidence)

    def rewrite_create_command(self, message: str) -> Optional[Dict[str, Any]]:
        content = str(message or "").strip()
        if not self.is_memory_command_message(content):
            return None
        result = self.rewrite_with_rules(content)
        if result is None:
            result = self.rewrite_with_llm(content)
        if result is None:
            return None
        return result.to_command()


def parse_memory_rewrite_json(content: Any) -> Optional[Dict[str, Any]]:
    text = str(content or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
