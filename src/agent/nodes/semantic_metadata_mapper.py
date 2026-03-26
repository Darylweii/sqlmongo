"""
Semantic Metadata Mapper Node - 语义增强的元数据映射节点

使用语义搜索替代传统的 LIKE 查询，实现更智能的设备匹配。

功能：
- 使用 FAISS 向量索引进行语义搜索
- 支持模糊匹配和同义词匹配
- 返回匹配置信度分数
- 支持直接使用设备代号查询（跳过 SQL 查询）
- 回退到传统 LIKE 查询（当语义搜索无结果时）
"""

import logging
import re
from typing import Optional, List, Dict, Any

from src.agent.types import GraphState, NODE_METADATA_MAPPER
from src.agent.query_entities import allows_explicit_multi_scope_aggregation
from src.agent.query_plan_state import (
    build_compat_intent_from_state,
    get_comparison_targets_from_state,
    get_project_hints_from_state,
    get_target_label_from_state,
    has_device_listing_intent_from_state,
    is_comparison_query,
)
from src.semantic_layer.device_search import DeviceSemanticSearch
from src.metadata.metadata_engine import MetadataEngine, DeviceInfo


logger = logging.getLogger(__name__)


# 设备代号的正则模式（如 a1_b1, B2_FCS13F5_PAU4_OA_T 等）
DEVICE_CODE_PATTERN = re.compile(r'^[a-zA-Z]\d*_[a-zA-Z0-9_]+$')


class SemanticMetadataMapperNode:
    """
    语义增强的元数据映射节点
    
    使用语义搜索从用户查询中匹配设备，支持：
    - 自然语言设备名称匹配
    - 模糊匹配和同义词
    - 匹配置信度评分
    
    当语义搜索无结果时，回退到传统 LIKE 查询。
    """
    
    def __init__(
        self,
        metadata_engine: Optional[MetadataEngine] = None,
        device_search: Optional[DeviceSemanticSearch] = None,
        min_score: float = 0.3,
        top_k: int = 10,
    ):
        """
        初始化语义元数据映射节点
        
        Args:
            metadata_engine: MetadataEngine 实例（用于回退查询）
            device_search: DeviceSemanticSearch 实例
            min_score: 最小匹配分数阈值
            top_k: 返回的最大设备数量
        """
        self.metadata_engine = metadata_engine
        self.min_score = min_score
        self.top_k = top_k
        
        # 初始化语义搜索
        if device_search:
            self.device_search = device_search
        else:
            self.device_search = DeviceSemanticSearch()
            if not self.device_search.initialize():
                logger.warning("语义搜索初始化失败，将使用回退模式")
                self.device_search = None
    
    def _is_device_code(self, target: str) -> bool:
        """
        检查目标是否是设备代号格式
        
        设备代号格式示例：
        - a1_b1, a1_b3, a9_b11
        - B2_FCS13F5_PAU4_OA_T
        - a20_B14
        
        Args:
            target: 查询目标
        
        Returns:
            是否是设备代号格式
        """
        return bool(DEVICE_CODE_PATTERN.match(target))
    
    def _get_device_name_from_metadata(self, device_code: str) -> Optional[str]:
        """
        从元数据中获取设备代号对应的中文名称
        
        Args:
            device_code: 设备代号
        
        Returns:
            设备中文名称，如果找不到返回 None
        """
        # 先从语义搜索的元数据中查找
        if self.device_search and self.device_search.is_initialized:
            metadata = self.device_search._metadata
            if metadata:
                for entry in metadata:
                    if entry['type'] == 'device' and entry['metadata'].get('device_id') == device_code:
                        return entry['metadata'].get('device_name', device_code)
        
        # 如果语义搜索没找到，尝试从 MetadataEngine 查找
        if self.metadata_engine:
            try:
                devices, _ = self.metadata_engine.search_devices(device_code)
                for d in devices:
                    if d.device == device_code:
                        return d.name
            except Exception as e:
                logger.warning(f"从 MetadataEngine 查找设备名称失败: {e}")
        
        return None
    
    def _normalize_scope_text(self, value: object) -> str:
        return "".join(str(value or "").strip().lower().split())

    def _get_alias_memory_row(self, state: GraphState, target: str) -> dict | None:
        alias_memory = state.get("alias_memory") if isinstance(state.get("alias_memory"), dict) else {}
        normalized_target = self._normalize_scope_text(target)
        if not normalized_target:
            return None
        entry = alias_memory.get(normalized_target)
        if not isinstance(entry, dict):
            return None
        device_code = str(entry.get("device") or "").strip()
        if not device_code:
            return None
        return {
            "device_id": device_code,
            "device_name": entry.get("name"),
            "device_type": entry.get("device_type", ""),
            "project_id": entry.get("project_id", ""),
            "project_name": entry.get("project_name"),
            "project_code_name": entry.get("project_code_name"),
            "tg": entry.get("tg"),
            "score": 1.0,
            "matched_fields": ["session_alias"],
            "match_reason": "session_alias",
        }

    def _find_project_match(self, project_hints: list[str]) -> dict | None:
        if not self.metadata_engine or not project_hints:
            return None
        try:
            projects = self.metadata_engine.list_projects()
        except Exception:
            return None
        best_project = None
        best_score = 0
        for hint in project_hints:
            normalized_hint = self._normalize_scope_text(hint)
            if not normalized_hint:
                continue
            for project in projects or []:
                project_name = self._normalize_scope_text(project.get("project_name"))
                project_code_name = self._normalize_scope_text(project.get("project_code_name"))
                score = 0
                if normalized_hint == project_name or normalized_hint == project_code_name:
                    score = 100
                elif normalized_hint and (normalized_hint in project_name or normalized_hint in project_code_name):
                    score = 80
                if score > best_score:
                    best_score = score
                    best_project = project
        return best_project if best_score > 0 else None


    def _get_project_devices(self, project_hints: list[str]) -> list[dict]:
        project = self._find_project_match(project_hints)
        if not project:
            return []
        devices = self.metadata_engine.get_devices_by_project(str(project.get("id") or ""))
        return [self._device_to_row(d) for d in devices]


    def _project_hint_score(self, row: dict, hint: str) -> int:
        normalized_hint = self._normalize_scope_text(hint)
        if not normalized_hint:
            return 0
        project_name = self._normalize_scope_text(row.get("project_name"))
        project_code_name = self._normalize_scope_text(row.get("project_code_name"))
        if normalized_hint == project_name:
            return 100
        if normalized_hint == project_code_name:
            return 95
        if normalized_hint and normalized_hint in project_name:
            return 80
        if normalized_hint and normalized_hint in project_code_name:
            return 75
        return 0

    def _select_clarification_candidates(self, rows: list[dict], limit: int = 10) -> list[dict]:
        result = []
        seen = set()
        for row in rows:
            match_type = row.get("match_type")
            if not match_type and row.get("score") is not None:
                match_type = "semantic"
            candidate = {
                "device": row.get("device_id"),
                "name": row.get("device_name"),
                "project_id": row.get("project_id"),
                "project_name": row.get("project_name"),
                "project_code_name": row.get("project_code_name"),
                "tg": row.get("tg"),
                "match_type": match_type,
                "match_score": row.get("score"),
                "match_reason": row.get("match_reason"),
                "matched_fields": row.get("matched_fields"),
                "scope_mode": row.get("scope_mode"),
            }
            dedupe_key = (candidate["device"], candidate["project_id"], candidate["tg"], candidate["name"])
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            result.append(candidate)
            if len(result) >= limit:
                break
        return result

    def _build_aggregate_scope_candidate(self, keyword: str, candidates: list[dict]) -> dict | None:
        if not candidates:
            return None
        device_codes = {str(item.get("device") or "").strip() for item in candidates if str(item.get("device") or "").strip()}
        if len(device_codes) != 1:
            return None
        device_code = next(iter(device_codes))
        scopes = {
            (
                str(item.get("project_id") or ""),
                str(item.get("project_name") or item.get("project_code_name") or ""),
                str(item.get("tg") or ""),
            )
            for item in candidates
        }
        if len(scopes) <= 1:
            return None
        return {
            "device": device_code,
            "name": f"\u6240\u6709 {device_code} \u8bbe\u5907",
            "project_id": "__aggregate_all__",
            "project_name": "\u8de8\u9879\u76ee\u6c47\u603b",
            "project_code_name": None,
            "tg": None,
            "match_type": "aggregate_all_option",
            "match_score": None,
            "match_reason": "\u6c47\u603b\u8be5\u8bbe\u5907\u5728\u6240\u6709\u5339\u914d\u9879\u76ee\u4e0b\u7684\u6570\u636e",
            "matched_fields": ["aggregate_scope"],
            "scope_mode": "aggregate_all",
        }


    def _build_clarification_message(self, groups: list[dict], resolved_rows: list[dict]) -> str:
        lines: list[str] = []
        for row in resolved_rows:
            device = str(row.get("device_id") or "").strip()
            name = str(row.get("device_name") or "").strip()
            project_name = str(row.get("project_name") or "").strip()
            if device and name and project_name:
                lines.append(f"已确认：{device} -> {device}（{name}，项目：{project_name}）")
            elif device and name:
                lines.append(f"已确认：{device} -> {device}（{name}）")

        for group in groups:
            keyword = str(group.get("keyword") or "").strip()
            candidates = list(group.get("candidates") or [])
            preview = "、".join(
                f"{item.get('device')}（{item.get('name') or item.get('project_name') or '未命名设备'}）"
                for item in candidates[:3]
            )
            if keyword:
                lines.append(f"“{keyword}”匹配到多个候选设备：{preview}。请确认你指的是哪一个。")

        return "\n".join(line for line in lines if line).strip() or "匹配到多个候选设备，请先确认。"

    def _resolve_exact_code_candidates(self, device_code: str, rows: list[dict], project_hints: list[str]) -> tuple[list[dict], dict | None]:
        normalized_code = self._normalize_scope_text(device_code)
        exact_rows = [row for row in rows if self._normalize_scope_text(row.get("device_id")) == normalized_code]
        if not exact_rows:
            return rows, None

        scopes = {
            (
                str(row.get("project_id") or ""),
                str(row.get("project_name") or ""),
                str(row.get("project_code_name") or ""),
                str(row.get("tg") or ""),
            )
            for row in exact_rows
        }
        if len(exact_rows) <= 1 or len(scopes) <= 1:
            return exact_rows[:1], None

        if project_hints:
            scored = []
            for row in exact_rows:
                best_score = max((self._project_hint_score(row, hint) for hint in project_hints), default=0)
                if best_score > 0:
                    scored.append((best_score, row))
            if scored:
                scored.sort(key=lambda item: item[0], reverse=True)
                top_score = scored[0][0]
                top_rows = [row for score, row in scored if score == top_score]
                top_scopes = {
                    (
                        str(row.get("project_id") or ""),
                        str(row.get("project_name") or ""),
                        str(row.get("project_code_name") or ""),
                        str(row.get("tg") or ""),
                    )
                    for row in top_rows
                }
                if len(top_rows) == 1 or len(top_scopes) == 1:
                    return [top_rows[0]], None

        clarification_rows = [
            {
                **row,
                "match_type": "exact_code_conflict",
                "score": None,
                "match_reason": "\u7cbe\u786e\u7801\u547d\u4e2d",
            }
            for row in exact_rows
        ]
        clarification_candidates = self._select_clarification_candidates(clarification_rows)
        aggregate_candidate = self._build_aggregate_scope_candidate(device_code, clarification_candidates)
        if aggregate_candidate:
            clarification_candidates.append(aggregate_candidate)
        return [], {"keyword": device_code, "candidates": clarification_candidates}

    def _narrow_rows_by_project_hints(self, rows: list[dict], project_hints: list[str]) -> list[dict]:
        if len(rows) <= 1 or not project_hints:
            return rows
        scored = []
        for row in rows:
            best_score = max((self._project_hint_score(row, hint) for hint in project_hints), default=0)
            if best_score > 0:
                scored.append((best_score, row))
        if not scored:
            return rows
        scored.sort(key=lambda item: item[0], reverse=True)
        top_score = scored[0][0]
        top_rows = [row for score, row in scored if score == top_score]
        return top_rows or rows

    def __call__(self, state: GraphState) -> GraphState:
        """执行语义设备查询，优先直接读取 query_plan。"""
        history = list(state.get("history", []))
        intent = build_compat_intent_from_state(state)

        comparison_targets = get_comparison_targets_from_state(state)
        is_comparison = is_comparison_query(state) and len(comparison_targets) > 1
        target = get_target_label_from_state(state) or str(intent.get("target") or "").strip()

        if not target and not comparison_targets:
            return {
                **state,
                "error": "意图解析结果为空，无法继续设备查询。",
                "error_node": NODE_METADATA_MAPPER,
                "history": history + [{
                    "node": NODE_METADATA_MAPPER,
                    "result": "错误: 未解析出查询目标",
                }],
            }

        targets = list(comparison_targets) if len(comparison_targets) > 1 else []

        if not targets:
            explicit_device_codes = re.findall(r"\b[a-zA-Z]\d*_[a-zA-Z0-9_]+\b", target)
            separator_pattern = r"[\s,，、;；/]+"
            if len(explicit_device_codes) > 1:
                seen = set()
                targets = []
                for code in explicit_device_codes:
                    key = code.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    targets.append(code)
            elif re.search(separator_pattern, target):
                targets = [item.strip() for item in re.split(separator_pattern, target) if item.strip()]
            else:
                connector_match = re.search(r"(.+?)(?:和|与|跟|同|vs|VS)(.+)", target)
                if connector_match:
                    left = connector_match.group(1).strip()
                    right = connector_match.group(2).strip()
                    if left and right:
                        targets = [left, right]

            if len(targets) > 1:
                is_comparison = True
            else:
                targets = []

        try:
            if is_comparison and len(targets) > 1:
                return self._handle_comparison_query(state, targets, history)
            return self._handle_single_query(state, target, history)
        except Exception as e:
            logger.exception(f"设备查询异常: {e}")
            return {
                **state,
                "error": f"设备查询异常: {str(e)}",
                "error_node": NODE_METADATA_MAPPER,
                "history": history + [{
                    "node": NODE_METADATA_MAPPER,
                    "result": f"异常: {str(e)}",
                }],
            }

    def _handle_single_query(self, state: GraphState, target: str, history: list) -> GraphState:
        """???????"""
        devices = []
        search_mode = "????"
        project_hints = get_project_hints_from_state(state)
        alias_row = self._get_alias_memory_row(state, target)

        # 1. ????????????
        if alias_row:
            devices = [dict(alias_row)]
            search_mode = "???????"

        # 2. ????????????? a1_b1?
        elif self._is_device_code(target):
            devices = self._fallback_search(target) if self.metadata_engine else []
            search_mode = "????????"

        # 3. ??????
        elif self.device_search and self.device_search.is_initialized:
            devices = self._semantic_search(target)

        # 4. ?????????????????
        if not devices and self.metadata_engine:
            search_mode = "LIKE ??"
            devices = self._fallback_search(target)
            if not devices and has_device_listing_intent_from_state(state) and project_hints:
                devices = self._get_project_devices(project_hints)
                if devices:
                    search_mode = "????????"

        if not devices:
            return {
                **state,
                "error": f"未找到与 '{target}' 匹配的设备",
                "error_node": NODE_METADATA_MAPPER,
                "history": history + [{
                    "node": NODE_METADATA_MAPPER,
                    "result": f"错误: 未找到与 '{target}' 匹配的设备"
                }]
            }
        
        # 提取设备代号列表
        device_codes = [d['device_id'] for d in devices]
        tg_values = [str(d.get('tg')).strip() for d in devices if d.get('tg')]

        # 构建设备代号到中文名称的映射
        device_names = {d['device_id']: d['device_name'] for d in devices}
        
        # 计算平均置信度
        avg_score = sum(d.get('score', 1.0) for d in devices) / len(devices)
        
        logger.info(f"设备查询成功 [{search_mode}]: 目标='{target}', 找到 {len(device_codes)} 个设备, 平均分数={avg_score:.3f}")

        if has_device_listing_intent_from_state(state):
            return {
                **state,
                "device_codes": device_codes,
                "device_names": device_names,
                "tg_values": tg_values,
                "devices": devices,
                "final_response": f"已找到 {len(devices)} 个相关设备，请查看下表。",
                "show_table": True,
                "table_type": "devices",
                "query_info": {
                    "devices": devices,
                    "target": target,
                    "search_mode": search_mode,
                },
                "is_comparison": False,
                "history": history + [{
                    "node": NODE_METADATA_MAPPER,
                    "result": f"找到 {len(device_codes)} 个设备候选"
                }]
            }

        if self._is_device_code(target):
            resolved_rows, clarification = self._resolve_exact_code_candidates(target, devices, project_hints)
            if clarification:
                return {
                    **state,
                    "clarification_required": True,
                    "clarification_candidates": [clarification],
                    "final_response": self._build_clarification_message([clarification], []),
                    "show_table": False,
                    "table_type": None,
                    "history": history + [{
                        "node": NODE_METADATA_MAPPER,
                        "result": f"找到 {len(clarification.get('candidates') or [])} 个候选设备，等待用户确认",
                    }],
                }
            devices = resolved_rows
        else:
            devices = self._narrow_rows_by_project_hints(devices, project_hints)
            if len(devices) > 1:
                clarification = {"keyword": target, "candidates": self._select_clarification_candidates(devices)}
                return {
                    **state,
                    "clarification_required": True,
                    "clarification_candidates": [clarification],
                    "final_response": self._build_clarification_message([clarification], []),
                    "show_table": False,
                    "table_type": None,
                    "history": history + [{
                        "node": NODE_METADATA_MAPPER,
                        "result": f"找到 {len(clarification.get('candidates') or [])} 个候选设备，等待用户确认",
                    }],
                }

        device_codes = [d['device_id'] for d in devices]
        tg_values = [str(d.get('tg')).strip() for d in devices if d.get('tg')]
        device_names = {d['device_id']: d['device_name'] for d in devices}

        return {
            **state,
            "device_codes": device_codes,
            "device_names": device_names,
            "tg_values": tg_values,
            "is_comparison": False,
            "history": history + [{
                "node": NODE_METADATA_MAPPER,
                "result": f"找到 {len(device_codes)} 个设备 [{search_mode}]"
            }]
        }
    
    def _handle_comparison_query(self, state: GraphState, targets: List[str], history: list) -> GraphState:
        """处理对比查询（多目标）"""
        all_device_codes = []
        all_device_names = {}
        comparison_device_groups = {}
        comparison_scope_groups = {}
        devices_with_metadata = {}  # 保存完整的设备元数据用于过滤
        resolved_rows = []
        clarification_groups = []
        project_hints = get_project_hints_from_state(state)

        query_text = str(getattr(state.get("query_plan"), "current_question", None) or state.get("query") or "")
        for target in targets:
            devices = []
            alias_row = self._get_alias_memory_row(state, target)
            if alias_row:
                devices = [dict(alias_row)]

            # 1. explicit code lookup
            if self._is_device_code(target) and not devices:
                devices = self._fallback_search(target) if self.metadata_engine else []
            # 2. semantic search
            elif not devices and self.device_search and self.device_search.is_initialized:
                devices = self._semantic_search(target)
            # 3. fallback fuzzy lookup
            if not devices and self.metadata_engine:
                devices = self._fallback_search(target)

            if devices:
                if self._is_device_code(target):
                    allow_multi_scope_aggregation = allows_explicit_multi_scope_aggregation(query_text, target)
                    narrowed_rows, clarification = self._resolve_exact_code_candidates(target, devices, project_hints)
                    if clarification and not allow_multi_scope_aggregation:
                        clarification_groups.append(clarification)
                        continue
                    devices = devices if (clarification and allow_multi_scope_aggregation) else narrowed_rows
                else:
                    devices = self._narrow_rows_by_project_hints(devices, project_hints)
                    if len(devices) > 1:
                        clarification_groups.append(
                            {
                                "keyword": target,
                                "candidates": self._select_clarification_candidates(devices),
                            }
                        )
                        continue

                device_codes = [d['device_id'] for d in devices]
                comparison_device_groups[target] = device_codes
                comparison_scope_groups[target] = [
                    {
                        'device': d.get('device_id'),
                        'name': d.get('device_name') or d.get('name'),
                        'project_id': d.get('project_id'),
                        'project_name': d.get('project_name'),
                        'project_code_name': d.get('project_code_name'),
                        'tg': d.get('tg'),
                    }
                    for d in devices if isinstance(d, dict)
                ]
                all_device_codes.extend(device_codes)

                # 保存设备元数据（包含 project_id, project_name 等）
                devices_with_metadata[target] = devices
                resolved_rows.extend(devices)

                for d in devices:
                    all_device_names[d['device_id']] = d['device_name']

        if clarification_groups:
            return {
                **state,
                "clarification_required": True,
                "clarification_candidates": clarification_groups,
                "device_codes": list(dict.fromkeys(all_device_codes)),
                "device_names": all_device_names,
                "tg_values": list(dict.fromkeys([str(row.get('tg')).strip() for row in resolved_rows if row.get('tg')])),
                "resolved_devices": resolved_rows,
                "final_response": self._build_clarification_message(clarification_groups, resolved_rows),
                "show_table": False,
                "table_type": None,
                "history": history + [{
                    "node": NODE_METADATA_MAPPER,
                    "result": f"找到 {sum(len(group.get('candidates') or []) for group in clarification_groups)} 个候选设备，等待用户确认",
                }],
            }

        if not all_device_codes:
            return {
                **state,
                "error": f"未找到与任何目标匹配的设备: {targets}",
                "error_node": NODE_METADATA_MAPPER,
                "history": history + [{
                    "node": NODE_METADATA_MAPPER,
                    "result": f"错误: 未找到匹配设备"
                }]
            }

        # 【新增】智能设备过滤
        from src.agent.utils.smart_device_filter import SmartDeviceFilter

        filtered_devices, filter_info = SmartDeviceFilter.filter_comparison_devices(
            devices_with_metadata
        )

        # 更新 comparison_device_groups 为过滤后的设备
        comparison_device_groups = {}
        all_device_codes = []
        for target, devices in filtered_devices.items():
            device_codes = [d['device_id'] for d in devices]
            comparison_device_groups[target] = device_codes
            all_device_codes.extend(device_codes)

        # 去重
        all_device_codes = list(dict.fromkeys(all_device_codes))

        # 【增强】构建详细的日志信息
        filter_strategy = filter_info.get('strategy', '无过滤')
        filter_details = filter_info.get('details', '')

        # 构建每个目标的设备数量信息
        target_device_counts = {
            target: len(devices)
            for target, devices in comparison_device_groups.items()
        }
        device_count_str = ", ".join([f"{t}={c}个" for t, c in target_device_counts.items()])

        logger.info(
            f"对比查询成功: {len(targets)} 个目标, 共 {len(all_device_codes)} 个设备 "
            f"({device_count_str}), 过滤策略={filter_strategy}, {filter_details}"
        )

        # 构建更详细的历史记录
        history_result = (
            f"对比查询: {len(targets)} 个目标, 共 {len(all_device_codes)} 个设备 "
            f"[{filter_strategy}]"
        )

        return {
            **state,
            "device_codes": all_device_codes,
            "device_names": all_device_names,
            "tg_values": [str(d.get('tg')).strip() for devices in filtered_devices.values() for d in devices if d.get('tg')],
            "is_comparison": True,
            "comparison_targets": targets,
            "comparison_device_groups": comparison_device_groups,
            "comparison_scope_groups": comparison_scope_groups,
            "filter_info": filter_info,  # 【新增】过滤信息
            "history": history + [{
                "node": NODE_METADATA_MAPPER,
                "result": history_result
            }]
        }
    
    def _semantic_search(self, target: str) -> List[Dict[str, Any]]:
        """
        使用语义搜索查找设备
        
        Args:
            target: 查询目标
        
        Returns:
            设备信息列表
        """
        results = self.device_search.search_devices(
            query=target,
            top_k=self.top_k,
            min_score=self.min_score,
        )
        
        return results
    
    def _fallback_search(self, target: str) -> List[Dict[str, Any]]:
        """
        回退到传统 LIKE 查询
        
        Args:
            target: 查询目标
        
        Returns:
            设备信息列表
        """
        devices, _ = self.metadata_engine.search_devices(target)
        
        return [
            {
                'device_id': d.device,
                'device_name': d.name,
                'device_type': d.device_type,
                'project_id': d.project_id,
                'project_name': d.project_name,
                'project_code_name': getattr(d, 'project_code_name', None),
                'tg': getattr(d, 'tg', None),
                'score': 1.0,  # LIKE 查询没有分数
            }
            for d in devices
        ]


__all__ = ["SemanticMetadataMapperNode"]
