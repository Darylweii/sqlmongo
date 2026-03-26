"""
Collection Router module for dynamic MongoDB collection routing.

This module provides functionality to calculate target MongoDB collection names
based on date ranges, following the naming convention: source_data_{type}_YYYYMM

Requirements:
- 5.2: 使用语义层进行数据类型解析（可选）
- 5.3: 提供与现有接口兼容的语义层支持
"""

from datetime import datetime
from typing import List, Dict, Optional, TYPE_CHECKING
from dateutil.relativedelta import relativedelta
import logging

from src.exceptions import CircuitBreakerError, InvalidDateRangeError

if TYPE_CHECKING:
    from src.semantic_layer.semantic_layer import SemanticLayer

logger = logging.getLogger(__name__)


# 数据类型到集合前缀的映射（包含中文别名）
DATA_TYPE_PREFIXES: Dict[str, str] = {
    # === 电量/电能相关 ===
    "ep": "source_data_ep_",
    "电量": "source_data_ep_",
    "电能": "source_data_ep_",
    "用电量": "source_data_ep_",
    "发电量": "source_data_ep_",
    "elec": "source_data_ep_",  # day_data_elec 映射到 ep
    
    "epzyz": "source_data_epzyz_",
    "正向有功电能": "source_data_epzyz_",
    
    "fz-ep": "source_data_fz-ep_",
    "反向电能": "source_data_fz-ep_",
    "负载电量": "source_data_fz-ep_",
    
    "gffddl": "source_data_gffddl_",
    "光伏发电量": "source_data_gffddl_",
    
    # === 电流相关 ===
    "i": "source_data_i_",
    "电流": "source_data_i_",
    "三相电流": "source_data_i_",
    "ia": "source_data_i_",
    "ib": "source_data_i_",
    "ic": "source_data_i_",
    "a相电流": "source_data_i_",
    "b相电流": "source_data_i_",
    "c相电流": "source_data_i_",
    
    "Ia": "source_data_Ia_",
    "Ib": "source_data_Ib_",
    "Ic": "source_data_Ic_",
    
    "bms_i": "source_data_bms_i_",
    "储能电流": "source_data_bms_i_",
    "电池电流": "source_data_bms_i_",
    
    "dw_i": "source_data_dw_i_",
    "fz_i": "source_data_fz_i_",
    "pv_i": "source_data_pv_i_",
    "光伏电流": "source_data_pv_i_",
    
    "ibph": "source_data_ibph_",
    
    # === 电压相关 ===
    "u": "source_data_u_",
    "电压": "source_data_u_line_",
    "三相电压": "source_data_u_line_",
    "相电压": "source_data_u_line_",
    "ua": "source_data_u_line_",
    "ub": "source_data_u_line_",
    "uc": "source_data_u_line_",
    "a相电压": "source_data_u_line_",
    "b相电压": "source_data_u_line_",
    "c相电压": "source_data_u_line_",
    
    "Ua": "source_data_Ua_",
    "Ub": "source_data_Ub_",
    "Uc": "source_data_Uc_",
    
    "u_line": "source_data_u_line_",
    "线电压": "source_data_u_line_",
    
    "u_phase": "source_data_u_phase_",
    "相间电压": "source_data_u_phase_",
    "uab": "source_data_u_phase_",
    "ubc": "source_data_u_phase_",
    "uca": "source_data_u_phase_",
    
    "bms_u": "source_data_bms_u_",
    "储能电压": "source_data_bms_u_",
    "电池电压": "source_data_bms_u_",
    
    "dw_u": "source_data_dw_u_",
    "fz_u": "source_data_fz_u_",
    "pv_u": "source_data_pv_u_",
    "光伏电压": "source_data_pv_u_",
    
    "ubph": "source_data_ubph_",
    
    # === 功率相关 ===
    "p": "source_data_p_",
    "P": "source_data_P_",
    "功率": "source_data_p_",
    "有功功率": "source_data_p_",
    
    "qf": "source_data_qf_",
    "功率因数": "source_data_qf_",
    "pf": "source_data_qf_",
    
    "loadrate": "source_data_loadrate_",
    "负载率": "source_data_loadrate_",
    
    # === 频率相关 ===
    "f": "source_data_f_",
    "频率": "source_data_f_",
    
    # === 温度/湿度相关 ===
    "t": "source_data_t_",
    "温度": "source_data_t_",
    "ta": "source_data_t_",
    "tb": "source_data_t_",
    
    "sd": "source_data_sd_",
    "s": "source_data_sd_",
    "湿度": "source_data_sd_",
    
    # === 谐波相关 ===
    "thd_i": "source_data_thd_i_",
    "电流谐波": "source_data_thd_i_",
    "thd-ia": "source_data_thd_i_",
    "thd-ib": "source_data_thd_i_",
    "thd-ic": "source_data_thd_i_",
    
    "thd_v": "source_data_thd_v_",
    "电压谐波": "source_data_thd_v_",
    "thd-va": "source_data_thd_v_",
    "thd-vb": "source_data_thd_v_",
    "thd-vc": "source_data_thd_v_",
    
    # === 储能/电池相关 ===
    "soc": "source_data_soc_",
    "电池容量": "source_data_soc_",
    "荷电状态": "source_data_soc_",
    
    "cbm": "source_data_cbm_",
    "cbmui": "source_data_cbmui_",
    
    # === 直流相关 ===
    "dcdcdljl": "source_data_dcdcdljl_",
    "低压直流电量": "source_data_dcdcdljl_",
    "直流充电电量": "source_data_dcdcdljl_",
    
    "dcdfdljl": "source_data_dcdfdljl_",
    "低压直流反向电量": "source_data_dcdfdljl_",
    "直流放电电量": "source_data_dcdfdljl_",
    
    "tzddc": "source_data_tzddc_",
    "tzgdc": "source_data_tzgdc_",
}

# 数据类型对应的 tag 字段值
DATA_TYPE_TAGS: Dict[str, List[str]] = {
    "source_data_Ia_": ["ia"],
    "source_data_Ib_": ["ib"],
    "source_data_Ic_": ["ic"],
    "source_data_P_": ["p"],
    "source_data_Ua_": ["ua"],
    "source_data_Ub_": ["ub"],
    "source_data_Uc_": ["uc"],
    "source_data_bms_i_": ["i"],
    "source_data_bms_u_": ["u"],
    "source_data_cbm_": ["u", "i"],
    "source_data_cbmui_": ["u", "i"],
    "source_data_dcdcdljl_": ["dcdcdljl"],
    "source_data_dcdfdljl_": ["dcdfdljl"],
    "source_data_dw_i_": ["dw_i"],
    "source_data_dw_u_": ["dw_u"],
    "source_data_ep_": ["ep"],
    "source_data_epzyz_": ["epzyz"],
    "source_data_f_": ["f"],  # 频率
    "source_data_fz-ep_": ["fz-ep"],
    "source_data_fz_i_": ["fz_i"],
    "source_data_fz_u_": ["fz_u"],
    "source_data_gffddl_": ["gffddl"],
    "source_data_i_": ["ia", "ib", "ic"],
    "source_data_ibph_": ["ibph"],
    "source_data_loadrate_": ["loadrate"],
    "source_data_p_": ["p"],
    "source_data_pv_i_": ["pv_i"],
    "source_data_pv_u_": ["pv_u"],
    "source_data_qf_": ["qf"],
    "source_data_sd_": ["sd"],  # 湿度
    "source_data_soc_": ["soc"],
    "source_data_t_": ["ta", "tb"],
    "source_data_thd_i_": ["thd-ia", "thd-ib", "thd-ic"],
    "source_data_thd_v_": ["thd-va", "thd-vb", "thd-vc"],
    "source_data_tzddc_": ["tzddc"],
    "source_data_tzgdc_": ["tzgdc"],
    "source_data_u_": ["u"],
    "source_data_u_line_": ["ua", "ub", "uc"],
    "source_data_u_phase_": ["uab", "ubc", "uca"],
    "source_data_ubph_": ["ubph"],
}

# 预聚合表数据类型映射 (用于 day_data_*, month_data_*, year_data_*)
# 这些类型有对应的预聚合表
AGGREGATED_DATA_TYPES = {
    # 电量相关
    "ep": "ep",
    "电量": "ep",
    "电能": "ep",
    "elec": "elec",  # day_data_elec 是综合电气表（含 ep, ia, ib, ic, p, ua, ub, uc）
    "综合电气": "elec",
    "电气综合": "elec",
    "综合用电": "elec",
    
    # 电流
    "i": "i",
    "电流": "i",
    
    # 电压
    "u_line": "u_line",
    "线电压": "u_line",
    "电压": "u_line",
    
    # 功率
    "p": "p",
    "功率": "p",
    
    # 功率因数
    "qf": "qf",
    "功率因数": "qf",
    
    # 负载率
    "loadrate": "loadrate",
    "负载率": "loadrate",
    
    # 温度
    "t": "t",
    "温度": "t",
    
    # 湿度
    "sd": "sd",
    "湿度": "sd",
    
    # 频率
    "f": "f",
    "频率": "f",
    
    # 光伏发电量
    "gffddl": "gffddl",
    "光伏发电量": "gffddl",
    
    # 反向电能
    "fz-ep": "fz-ep",
    "反向电能": "fz-ep",
    
    # 正向有功电能
    "epzyz": "epzyz",
    "正向有功电能": "epzyz",
    
    # 直流充放电
    "dcdcdljl": "dcdcdljl",
    "直流充电电量": "dcdcdljl",
    "dcdfdljl": "dcdfdljl",
    "直流放电电量": "dcdfdljl",
}


def get_aggregated_data_type(data_type: str) -> str:
    """
    获取预聚合表的数据类型
    
    Args:
        data_type: 用户输入的数据类型（如 "电流", "电量", "i" 等）
    
    Returns:
        预聚合表数据类型（如 "i", "ep" 等），如果没有对应的预聚合表则返回原类型
    """
    data_type_lower = data_type.lower().strip()
    # 先尝试小写匹配
    if data_type_lower in AGGREGATED_DATA_TYPES:
        return AGGREGATED_DATA_TYPES[data_type_lower]
    # 再尝试原始大小写匹配
    if data_type in AGGREGATED_DATA_TYPES:
        return AGGREGATED_DATA_TYPES[data_type]
    # 默认返回原类型
    return data_type_lower


def get_collection_prefix(
    data_type: str,
    semantic_layer: Optional["SemanticLayer"] = None,
) -> str:
    """
    根据数据类型获取集合前缀
    
    当提供语义层时，使用语义匹配来解析口语化的数据类型查询。
    如果语义层未提供或未初始化，则回退到字典匹配。
    
    Args:
        data_type: 数据类型（如 "电流", "电量", "ia", "吃电情况" 等）
        semantic_layer: 语义层实例（可选，为 None 时使用字典匹配）
    
    Returns:
        集合前缀（如 "source_data_i_"）
    
    需求引用:
        - 需求 5.2: 使用语义层进行数据类型解析
        - 需求 5.3: 提供与现有接口兼容的语义层支持
    
    Example:
        # 传统模式（向后兼容）
        prefix = get_collection_prefix("电量")
        # -> "source_data_ep_"
        
        # 语义层模式
        prefix = get_collection_prefix("吃电情况", semantic_layer=sl)
        # -> "source_data_ep_" (通过语义匹配)
    """
    # 尝试使用语义层进行语义匹配
    if semantic_layer is not None:
        try:
            if semantic_layer.is_initialized:
                prefix = semantic_layer.get_collection_prefix(data_type)
                if prefix:
                    logger.debug(f"语义层匹配: '{data_type}' -> '{prefix}'")
                    return prefix
        except Exception as e:
            logger.warning(f"语义层匹配失败，回退到字典匹配: {e}")
    
    # 回退到字典匹配
    return _get_collection_prefix_from_dict(data_type)


def _get_collection_prefix_from_dict(data_type: str) -> str:
    """
    使用字典进行数据类型到集合前缀的匹配（内部函数）
    
    Args:
        data_type: 数据类型
    
    Returns:
        集合前缀
    """
    data_type_lower = data_type.lower().strip()
    # 先尝试小写匹配
    if data_type_lower in DATA_TYPE_PREFIXES:
        return DATA_TYPE_PREFIXES[data_type_lower]
    # 再尝试原始大小写匹配
    if data_type in DATA_TYPE_PREFIXES:
        return DATA_TYPE_PREFIXES[data_type]
    # 默认返回电量
    return "source_data_ep_"


def get_data_tags(collection_prefix: str) -> List[str]:
    """
    根据集合前缀获取对应的 tag 值列表
    
    Args:
        collection_prefix: 集合前缀
    
    Returns:
        tag 值列表
    """
    return DATA_TYPE_TAGS.get(collection_prefix, [])


def get_target_collections(
    start_date: str,
    end_date: str,
    collection_prefix: str = "source_data_ep_",
    max_collections: int = 50
) -> List[str]:
    """
    Calculate target MongoDB collection names based on date range.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        collection_prefix: Collection name prefix (default: "source_data_ep_")
        max_collections: Maximum allowed collections (circuit breaker threshold)
    
    Returns:
        List of collection names in chronological order
    
    Raises:
        InvalidDateRangeError: If date format is invalid or start_date > end_date
        CircuitBreakerError: If collection count exceeds max_collections
    
    Examples:
        >>> get_target_collections("2024-01-15", "2024-03-20")
        ['source_data_ep_202401', 'source_data_ep_202402', 'source_data_ep_202403']
    """
    # Parse dates
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        raise InvalidDateRangeError(
            f"无效的开始日期格式：'{start_date}'，请使用 YYYY-MM-DD 格式"
        )
    
    try:
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise InvalidDateRangeError(
            f"无效的结束日期格式：'{end_date}'，请使用 YYYY-MM-DD 格式"
        )
    
    # Validate date range
    if start > end:
        raise InvalidDateRangeError(
            f"开始日期 ({start_date}) 不能晚于结束日期 ({end_date})"
        )
    
    # Generate collection names for each month in range
    collections = []
    current = start.replace(day=1)  # Normalize to first day of month
    end_month = end.replace(day=1)
    
    while current <= end_month:
        collection_name = f"{collection_prefix}{current.strftime('%Y%m')}"
        collections.append(collection_name)
        current += relativedelta(months=1)
    
    # Circuit breaker check
    if len(collections) > max_collections:
        raise CircuitBreakerError(
            collection_count=len(collections),
            max_allowed=max_collections
        )
    
    return collections
