from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.query_plan import QueryPlan, coerce_query_plan, fallback_query_plan

logger = logging.getLogger(__name__)


PLANNER_SYSTEM_PROMPT = """You are a Query Planner for an energy data assistant.
Your job is to transform a natural-language question into a structured query plan.
Return JSON only. Do not explain your reasoning.

Required fields:
- query_mode: general | sensor_query | comparison | detect_data_types | project_listing | project_stats | device_listing | trend_decision | ranked_timepoints | ranked_buckets | anomaly_points
- inferred_data_type: metric id such as ep / u_line / i / p / qf / t, or null if unknown
- explicit_device_codes: array of explicitly mentioned device codes; never invent them
- search_targets: array of entities that still require entity resolution / retrieval
- project_hints: array of project/platform/system hints
- time_start: YYYY-MM-DD or null
- time_end: YYYY-MM-DD or null
- has_sensor_intent / has_detect_data_types_intent / has_project_listing_intent / has_project_stats_intent / has_device_listing_intent / has_comparison_intent / has_pagination_intent / has_time_reference / has_ranked_point_intent / has_trend_decision_intent / has_anomaly_point_intent: booleans
- ranking_order: desc | asc | null
- ranking_limit: positive integer or null
- ranking_granularity: hour | day | week | month | null
- aggregation: raw | bucket | delta | avg | sum | max | min | trend_window | period_compare | compare | null
- response_style: direct_answer | structured_analysis | clarify | list | compare
- period_compare_targets: if the user is explicitly comparing two periods, return them as an array, otherwise []
- confidence: float between 0 and 1

Guidelines:
1. Prefer semantic understanding over surface keyword matching.
2. Preserve explicit device codes and user-provided dates.
3. Do not invent projects, devices, dates, or metric ids.
4. Only refine fields that are still missing or ambiguous in the fallback parse.
"""

LOCAL_ONLY_QUERY_MODES = {"project_listing", "project_stats", "device_listing", "detect_data_types"}
SINGLE_SENSOR_QUERY_MODES = {"sensor_query", "ranked_timepoints", "ranked_buckets", "trend_decision", "anomaly_points"}


class LLMQueryPlanner:
    def __init__(self, llm: Any):
        self.llm = llm

    def plan(self, user_query: str) -> QueryPlan:
        fallback = fallback_query_plan(user_query)
        if not self._should_use_llm_completion(fallback):
            return fallback

        try:
            payload = self._invoke_llm(user_query, fallback)
        except Exception as exc:  # noqa: BLE001
            logger.warning("query.planner.llm_failed query=%s error=%s", fallback.current_question, exc)
            return fallback

        llm_plan = coerce_query_plan(payload)
        if llm_plan is None:
            return fallback
        return self._merge_with_fallback(llm_plan, fallback)

    def _has_target_hints(self, plan: QueryPlan) -> bool:
        return bool(plan.explicit_device_codes or plan.search_targets or plan.project_hints)

    def _has_metric(self, plan: QueryPlan) -> bool:
        return bool(str(plan.inferred_data_type or "").strip())

    def _has_comparison_targets(self, plan: QueryPlan) -> bool:
        explicit_count = len(plan.explicit_device_codes or ())
        search_count = len(plan.search_targets or ())
        return explicit_count > 1 or search_count > 1

    def _is_ranked_plan_ready(self, plan: QueryPlan) -> bool:
        return self._has_target_hints(plan) and self._has_metric(plan) and bool(plan.ranking_order)

    def _is_single_sensor_plan_ready(self, plan: QueryPlan) -> bool:
        if plan.query_mode in {"ranked_timepoints", "ranked_buckets"}:
            return self._is_ranked_plan_ready(plan)
        return self._has_target_hints(plan) and self._has_metric(plan)

    def _should_use_llm_completion(self, fallback: QueryPlan) -> bool:
        if not self.llm or not hasattr(self.llm, "invoke"):
            return False
        if fallback.query_mode in LOCAL_ONLY_QUERY_MODES:
            return False
        if fallback.query_mode == "comparison":
            return not (self._has_comparison_targets(fallback) and self._has_metric(fallback))
        if fallback.query_mode in SINGLE_SENSOR_QUERY_MODES:
            return not self._is_single_sensor_plan_ready(fallback)
        return fallback.query_mode == "general"

    def _invoke_llm(self, user_query: str, fallback: QueryPlan) -> Dict[str, Any]:
        fallback_payload = fallback.to_dict()
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"User question: {fallback.current_question or user_query}\n"
                    f"Fallback parse for reference only: {json.dumps(fallback_payload, ensure_ascii=False)}\n"
                    "Return the final QueryPlan JSON."
                )
            ),
        ]
        response = self.llm.invoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)
        return self._parse_json(response_text)

    def _parse_json(self, response_text: str) -> Dict[str, Any]:
        cleaned = re.sub(r"<think>.*?</think>", "", str(response_text or ""), flags=re.DOTALL).strip()
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _merge_with_fallback(self, llm_plan: QueryPlan, fallback: QueryPlan) -> QueryPlan:
        query_mode = llm_plan.query_mode if llm_plan.query_mode != "general" or fallback.query_mode == "general" else fallback.query_mode
        explicit_device_codes = llm_plan.explicit_device_codes or fallback.explicit_device_codes
        search_targets = llm_plan.search_targets or fallback.search_targets
        project_hints = llm_plan.project_hints or fallback.project_hints
        ranking_order = llm_plan.ranking_order or fallback.ranking_order
        ranking_limit = llm_plan.ranking_limit or fallback.ranking_limit
        ranking_granularity = llm_plan.ranking_granularity or fallback.ranking_granularity
        inferred_data_type = llm_plan.inferred_data_type or fallback.inferred_data_type
        period_compare_targets = llm_plan.period_compare_targets or fallback.period_compare_targets
        response_style = llm_plan.response_style or fallback.response_style
        aggregation = llm_plan.aggregation or fallback.aggregation
        time_start = llm_plan.time_start or fallback.time_start
        time_end = llm_plan.time_end or fallback.time_end
        confidence = max(llm_plan.confidence, fallback.confidence)

        merged_raw_plan = dict(fallback.raw_plan or {})
        merged_raw_plan.update(dict(llm_plan.raw_plan or {}))

        return QueryPlan(
            current_question=llm_plan.current_question or fallback.current_question,
            source="llm",
            query_mode=query_mode,
            inferred_data_type=inferred_data_type,
            explicit_device_codes=explicit_device_codes,
            search_targets=search_targets,
            project_hints=project_hints,
            time_start=time_start,
            time_end=time_end,
            has_sensor_intent=llm_plan.has_sensor_intent or fallback.has_sensor_intent or query_mode not in {"general", "project_listing", "project_stats", "device_listing"},
            has_detect_data_types_intent=llm_plan.has_detect_data_types_intent or fallback.has_detect_data_types_intent or query_mode == "detect_data_types",
            has_project_listing_intent=llm_plan.has_project_listing_intent or fallback.has_project_listing_intent or query_mode == "project_listing",
            has_project_stats_intent=llm_plan.has_project_stats_intent or fallback.has_project_stats_intent or query_mode == "project_stats",
            has_device_listing_intent=llm_plan.has_device_listing_intent or fallback.has_device_listing_intent or query_mode == "device_listing",
            has_comparison_intent=llm_plan.has_comparison_intent or fallback.has_comparison_intent or query_mode == "comparison",
            has_pagination_intent=llm_plan.has_pagination_intent or fallback.has_pagination_intent,
            has_time_reference=llm_plan.has_time_reference or fallback.has_time_reference,
            has_ranked_point_intent=llm_plan.has_ranked_point_intent or fallback.has_ranked_point_intent or query_mode in {"ranked_timepoints", "ranked_buckets"},
            ranking_order=ranking_order,
            ranking_limit=ranking_limit,
            ranking_granularity=ranking_granularity,
            has_trend_decision_intent=llm_plan.has_trend_decision_intent or fallback.has_trend_decision_intent or query_mode == "trend_decision",
            has_anomaly_point_intent=llm_plan.has_anomaly_point_intent or fallback.has_anomaly_point_intent or query_mode == "anomaly_points",
            aggregation=aggregation,
            response_style=response_style,
            period_compare_targets=period_compare_targets,
            confidence=confidence,
            raw_plan=merged_raw_plan,
        )
