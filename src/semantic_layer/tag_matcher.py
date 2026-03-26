"""
指标标签匹配器

基于关键词的快速指标匹配，不依赖向量搜索。
用于从用户查询中提取数据类型标签(tag)。
"""

import re
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# 指标同义词映射表
TAG_SYNONYMS = {
    # 基础电力指标
    'p': ['功率', '有功功率', '实时功率', '瞬时功率', '负载'],
    'i': ['电流', '三相电流', 'A相电流', 'B相电流', 'C相电流', '相电流', '安培'],
    'u_line': ['线电压', '三相电压', 'AB电压', 'BC电压', 'CA电压', '三相线电压', '电压稳不稳定', '电压稳定'],
    'u_phase': ['相电压', 'A相电压', 'B相电压', 'C相电压', '单相电压'],
    'ep': ['电量', '用电量', '电费', '耗电量', '正向有功电能', '累计电量', '总电量', 
           '用电情况', '吃电', '用了多少电', '电能', '度数'],
    'qf': ['功率因数', '功因', 'PF', 'cos', '力率', 'pf值'],
    
    # 温湿度
    't': ['温度', '温度值', '环境温度', '室温', '气温', '多少度', '温湿度'],
    'sd': ['湿度', '环境湿度', '相对湿度', '空气湿度'],
    
    # 谐波
    'thd_i': ['电流谐波', '谐波电流', 'THD-I', '电流畸变'],
    'thd_v': ['电压谐波', '谐波电压', 'THD-V', '电压畸变'],
    
    # 电池/储能
    'bms_i': ['电池电流', 'BMS电流', '储能电流'],
    'bms_u': ['电池电压', 'BMS电压', '储能电压'],
    'soc': ['电池容量', '荷电状态', 'SOC', '剩余电量', '电池电量', '电量百分比'],
    
    # 光伏
    'gffddl': ['光伏发电量', '发电量', '光伏发电', '太阳能发电', '光伏'],
    'pv_i': ['光伏电流', '太阳能电流', 'PV电流'],
    'pv_u': ['光伏电压', '太阳能电压', 'PV电压'],
    
    # 充放电
    'dcdcdljl': ['充电电量', '直流充电', '充电量', '充电'],
    'dcdfdljl': ['放电电量', '直流放电', '放电量', '放电'],
    
    # 其他
    'loadrate': ['负载率', '负荷率', '负载情况', '负荷'],
}

# 指标信息
TAG_INFO = {
    'p': {'name': '有功功率', 'unit': 'kW'},
    'i': {'name': '电流', 'unit': 'A'},
    'u_line': {'name': '线电压', 'unit': 'V'},
    'u_phase': {'name': '相电压', 'unit': 'V'},
    'ep': {'name': '正向有功电能', 'unit': 'kWh'},
    'qf': {'name': '功率因数', 'unit': ''},
    't': {'name': '温度', 'unit': '℃'},
    'sd': {'name': '湿度', 'unit': '%'},
    'thd_i': {'name': '电流谐波', 'unit': '%'},
    'thd_v': {'name': '电压谐波', 'unit': '%'},
    'bms_i': {'name': 'BMS电流', 'unit': 'A'},
    'bms_u': {'name': 'BMS电压', 'unit': 'V'},
    'soc': {'name': '荷电状态', 'unit': '%'},
    'gffddl': {'name': '光伏发电量', 'unit': 'kWh'},
    'pv_i': {'name': '光伏电流', 'unit': 'A'},
    'pv_u': {'name': '光伏电压', 'unit': 'V'},
    'dcdcdljl': {'name': '直流充电电量', 'unit': 'kWh'},
    'dcdfdljl': {'name': '直流放电电量', 'unit': 'kWh'},
    'loadrate': {'name': '负载率', 'unit': '%'},
}

# 构建反向索引：同义词 -> tag
_SYNONYM_TO_TAG = {}
for tag, synonyms in TAG_SYNONYMS.items():
    for synonym in synonyms:
        _SYNONYM_TO_TAG[synonym.lower()] = tag


def match_tag(query: str) -> Optional[Dict]:
    """
    从查询中匹配指标标签
    
    Args:
        query: 用户查询文本
    
    Returns:
        匹配结果字典，包含 tag, name, unit, matched_keyword
        如果没有匹配返回 None
    """
    query_lower = query.lower()
    
    # 按关键词长度降序排序，优先匹配更长的关键词
    sorted_synonyms = sorted(_SYNONYM_TO_TAG.keys(), key=len, reverse=True)
    
    for synonym in sorted_synonyms:
        if synonym in query_lower:
            tag = _SYNONYM_TO_TAG[synonym]
            info = TAG_INFO.get(tag, {})
            return {
                'tag': tag,
                'name': info.get('name', tag),
                'unit': info.get('unit', ''),
                'matched_keyword': synonym,
            }
    
    # 特殊处理：电压（不带前缀时默认线电压）
    if '电压' in query_lower and '相电压' not in query_lower:
        return {
            'tag': 'u_line',
            'name': '线电压',
            'unit': 'V',
            'matched_keyword': '电压',
        }
    
    return None


def match_all_tags(query: str) -> List[Dict]:
    """
    从查询中匹配所有可能的指标标签
    
    Args:
        query: 用户查询文本
    
    Returns:
        所有匹配结果的列表
    """
    query_lower = query.lower()
    results = []
    matched_tags = set()
    
    sorted_synonyms = sorted(_SYNONYM_TO_TAG.keys(), key=len, reverse=True)
    
    for synonym in sorted_synonyms:
        if synonym in query_lower:
            tag = _SYNONYM_TO_TAG[synonym]
            if tag not in matched_tags:
                matched_tags.add(tag)
                info = TAG_INFO.get(tag, {})
                results.append({
                    'tag': tag,
                    'name': info.get('name', tag),
                    'unit': info.get('unit', ''),
                    'matched_keyword': synonym,
                })
    
    return results


def get_tag_info(tag: str) -> Optional[Dict]:
    """获取指标信息"""
    if tag in TAG_INFO:
        return {
            'tag': tag,
            **TAG_INFO[tag],
            'synonyms': TAG_SYNONYMS.get(tag, []),
        }
    return None


def list_all_tags() -> List[Dict]:
    """列出所有支持的指标"""
    return [
        {'tag': tag, **info, 'synonyms': TAG_SYNONYMS.get(tag, [])}
        for tag, info in TAG_INFO.items()
    ]


# 测试
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    test_queries = [
        "功率",
        "电流",
        "电压稳不稳定",
        "用电情况",
        "功率因数",
        "房间温度",
        "机房湿度",
        "光伏发电量",
        "电池容量",
        "负载率",
    ]

    logger.info("指标匹配测试:")
    logger.info("%s", "-" * 60)
    for query in test_queries:
        result = match_tag(query)
        if result:
            logger.info(
                "%s -> %s (%s, 匹配: %s)",
                query,
                result["tag"],
                result["name"],
                result["matched_keyword"],
            )
        else:
            logger.info("%s -> 未匹配", query)
