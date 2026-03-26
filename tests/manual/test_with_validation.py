"""
带答案验证的测试脚本
先查询正确答案，再测试AI回答，对比结果
"""
import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(key, None)
os.environ['NO_PROXY'] = '*'

from dotenv import load_dotenv
load_dotenv()

from pymongo import MongoClient
from src.fetcher.data_fetcher import DataFetcher
from src.metadata.metadata_engine import MetadataEngine
from src.compressor.context_compressor import ContextCompressor
from src.agent.orchestrator import create_agent_with_streaming
from src.config import load_config
from langchain_openai import ChatOpenAI
import httpx

# 初始化组件
config = load_config()
metadata_engine = MetadataEngine(config.mysql.connection_string)
mongo_client = MongoClient(config.mongodb.uri)
data_fetcher = DataFetcher(
    mongo_client=mongo_client,
    database_name=config.mongodb.database_name,
    max_records=2000
)
compressor = ContextCompressor(max_tokens=4000)

# 初始化 LLM
vllm_base = os.getenv("VLLM_API_BASE") or os.getenv("LLM_BASE_URL")
http_client = httpx.Client(trust_env=False, timeout=httpx.Timeout(300.0))
llm = ChatOpenAI(
    model=os.getenv("VLLM_MODEL") or os.getenv("LLM_MODEL", "/models/Qwen3-32B-AWQ"),
    openai_api_base=vllm_base,
    openai_api_key=os.getenv("VLLM_API_KEY") or os.getenv("LLM_API_KEY") or "not-needed",
    temperature=0.7,
    max_tokens=16384,
    http_client=http_client,
    request_timeout=300.0,
)

# 创建 Agent
agent = create_agent_with_streaming(
    llm=llm,
    llm_non_streaming=llm,
    metadata_engine=metadata_engine,
    data_fetcher=data_fetcher,
    cache_manager=None,
    compressor=compressor
)

def get_ground_truth(test_case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """获取正确答案（Ground Truth）"""
    try:
        query_type = test_case.get("expected_type")
        
        # 设备搜索类
        if query_type == "device_list":
            keyword = test_case.get("expected_keyword", "")
            devices = metadata_engine.search_devices(keyword)
            return {
                "type": "device_list",
                "count": len(devices),
                "devices": devices[:10]  # 只取前10个
            }
        
        # 项目查询类
        elif query_type == "project_list":
            projects = metadata_engine.list_projects()
            return {
                "type": "project_list",
                "count": len(projects),
                "projects": projects
            }
        
        elif query_type == "project_stats":
            projects = metadata_engine.list_projects()
            stats = []
            for proj in projects:
                devices = metadata_engine.get_devices_by_project(proj["id"])
                stats.append({
                    "project": proj["project_name"],
                    "device_count": len(devices)
                })
            stats.sort(key=lambda x: x["device_count"], reverse=True)
            return {
                "type": "project_stats",
                "stats": stats[:5]  # 前5个
            }
        
        # 传感器数据查询类
        elif query_type == "sensor_data" or query_type == "comparison":
            devices = test_case.get("expected_devices", [])
            data_type = test_case.get("expected_data_type", "ep")
            
            if not devices:
                return None
            
            # 解析时间范围
            query = test_case["query"]
            start_time, end_time = parse_time_range(query)
            
            if not start_time or not end_time:
                return None
            
            # 查询数据
            collections = get_collections(data_type, start_time, end_time)
            result = data_fetcher.fetch_sync(
                collections=collections,
                devices=devices,
                start_time=start_time,
                end_time=end_time,
                tags=None,
                page=1,
                page_size=0  # 获取全部数据用于统计
            )
            
            return {
                "type": "sensor_data",
                "total_count": result.total_count,
                "statistics": result.statistics,
                "devices": devices,
                "time_range": f"{start_time.date()} 至 {end_time.date()}"
            }
        
        return None
    except Exception as e:
        print(f"  [警告] 获取正确答案失败: {e}")
        return None

def parse_time_range(query: str) -> tuple:
    """从查询中解析时间范围"""
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    if "今天" in query or "今日" in query:
        return today, datetime.now()
    elif "昨天" in query or "昨日" in query:
        yesterday = today - timedelta(days=1)
        return yesterday, today
    elif "最近7天" in query or "近7天" in query:
        return today - timedelta(days=7), datetime.now()
    elif "本周" in query:
        # 本周一到现在
        weekday = today.weekday()
        monday = today - timedelta(days=weekday)
        return monday, datetime.now()
    elif "上周" in query:
        weekday = today.weekday()
        last_monday = today - timedelta(days=weekday+7)
        last_sunday = last_monday + timedelta(days=6, hours=23, minutes=59, seconds=59)
        return last_monday, last_sunday
    elif "本月" in query:
        first_day = today.replace(day=1)
        return first_day, datetime.now()
    elif "2024年1月" in query or "2024年 1月" in query:
        return datetime(2024, 1, 1), datetime(2024, 1, 31, 23, 59, 59)
    elif "1月5日到1月10日" in query or "1月5日至1月10日" in query:
        return datetime(2024, 1, 5), datetime(2024, 1, 10, 23, 59, 59)
    
    return None, None

def get_collections(data_type: str, start_time: datetime, end_time: datetime) -> List[str]:
    """根据数据类型和时间范围获取集合名"""
    collections = []
    current = start_time.replace(day=1)
    
    while current <= end_time:
        coll_name = f"source_data_{data_type}_{current.strftime('%Y%m')}"
        if coll_name not in collections:
            collections.append(coll_name)
        
        # 下个月
        if current.month == 12:
            current = current.replace(year=current.year+1, month=1)
        else:
            current = current.replace(month=current.month+1)
    
    return collections

def validate_answer(ground_truth: Dict[str, Any], ai_response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """验证AI答案的正确性"""
    validation = {
        "correct": False,
        "score": 0.0,
        "details": []
    }
    
    if not ground_truth:
        validation["details"].append("无法获取正确答案，跳过验证")
        return validation
    
    gt_type = ground_truth.get("type")
    
    # 设备列表验证
    if gt_type == "device_list":
        gt_count = ground_truth["count"]
        # 检查AI回答中是否提到了设备数量
        if str(gt_count) in ai_response or f"{gt_count}个" in ai_response:
            validation["score"] += 0.5
            validation["details"].append(f"✓ 设备数量正确: {gt_count}个")
        else:
            validation["details"].append(f"✗ 设备数量不匹配: 期望{gt_count}个")
        
        # 检查是否提到了设备
        if "设备" in ai_response:
            validation["score"] += 0.5
            validation["details"].append("✓ 提到了设备")
    
    # 项目列表验证
    elif gt_type == "project_list":
        gt_count = ground_truth["count"]
        if str(gt_count) in ai_response or f"{gt_count}个" in ai_response:
            validation["score"] += 0.5
            validation["details"].append(f"✓ 项目数量正确: {gt_count}个")
        else:
            validation["details"].append(f"✗ 项目数量不匹配: 期望{gt_count}个")
        
        if "项目" in ai_response:
            validation["score"] += 0.5
            validation["details"].append("✓ 提到了项目")
    
    # 传感器数据验证
    elif gt_type == "sensor_data":
        gt_count = ground_truth["total_count"]
        gt_stats = ground_truth.get("statistics", {})
        
        # 验证数据量
        if gt_count > 0:
            if str(gt_count) in ai_response or f"{gt_count}条" in ai_response:
                validation["score"] += 0.3
                validation["details"].append(f"✓ 数据量正确: {gt_count}条")
            else:
                validation["details"].append(f"✗ 数据量不匹配: 期望{gt_count}条")
        
        # 验证统计值（允许一定误差）
        if gt_stats:
            gt_min = gt_stats.get("min")
            gt_max = gt_stats.get("max")
            gt_avg = gt_stats.get("avg")
            
            if gt_min and (str(int(gt_min)) in ai_response or str(round(gt_min, 1)) in ai_response):
                validation["score"] += 0.2
                validation["details"].append(f"✓ 最小值正确: {gt_min}")
            
            if gt_max and (str(int(gt_max)) in ai_response or str(round(gt_max, 1)) in ai_response):
                validation["score"] += 0.2
                validation["details"].append(f"✓ 最大值正确: {gt_max}")
            
            if gt_avg and (str(int(gt_avg)) in ai_response or str(round(gt_avg, 1)) in ai_response or str(round(gt_avg, 2)) in ai_response):
                validation["score"] += 0.3
                validation["details"].append(f"✓ 平均值正确: {gt_avg}")
    
    validation["correct"] = validation["score"] >= 0.6  # 60分及格
    return validation

# 选择5个代表性测试用例
SAMPLE_TEST_CASES = [
    {
        "id": 2,
        "category": "基础查询",
        "complexity": "简单",
        "query": "a1_b9 设备 2024年1月的电量",
        "expected_type": "sensor_data",
        "expected_devices": ["a1_b9"],
        "expected_data_type": "ep"
    },
    {
        "id": 6,
        "category": "设备搜索",
        "complexity": "简单",
        "query": "搜索包含 b9 的设备",
        "expected_type": "device_list",
        "expected_keyword": "b9"
    },
    {
        "id": 9,
        "category": "项目查询",
        "complexity": "简单",
        "query": "有哪些项目可用？",
        "expected_type": "project_list"
    },
    {
        "id": 15,
        "category": "统计分析",
        "complexity": "中等",
        "query": "a1_b9 设备2024年1月的平均用电量是多少？",
        "expected_type": "sensor_data",
        "expected_devices": ["a1_b9"],
        "expected_data_type": "ep",
        "needs_statistics": True
    },
    {
        "id": 16,
        "category": "统计分析",
        "complexity": "复杂",
        "query": "a1_b9 设备在2024年1月哪天用电最多？",
        "expected_type": "sensor_data",
        "expected_devices": ["a1_b9"],
        "expected_data_type": "ep",
        "needs_statistics": True
    }
]

def run_test_with_validation(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """运行带验证的测试"""
    print(f"\n{'='*70}")
    print(f"测试 #{test_case['id']}: {test_case['query']}")
    print(f"分类: {test_case['category']} | 复杂度: {test_case['complexity']}")
    print('='*70)
    
    # 1. 获取正确答案
    print("\n[步骤1] 获取正确答案...")
    ground_truth = get_ground_truth(test_case)
    
    if ground_truth:
        print(f"  正确答案: {json.dumps(ground_truth, ensure_ascii=False, indent=2)}")
    else:
        print("  无法获取正确答案")
    
    # 2. 让AI回答
    print("\n[步骤2] AI回答...")
    start_time = time.time()
    
    try:
        events = list(agent.run_with_progress(test_case["query"]))
        end_time = time.time()
        response_time = round(end_time - start_time, 2)
        
        # 查找最终响应
        ai_response = None
        for event in reversed(events):
            if event.get("type") in ["final_answer", "direct_response"]:
                ai_response = event.get("response", "")
                break
        
        if ai_response:
            print(f"  AI回答: {ai_response[:200]}...")
            print(f"  响应时间: {response_time}s")
        else:
            print("  未获取到AI回答")
            return {
                "id": test_case["id"],
                "success": False,
                "error": "未获取到AI回答"
            }
        
        # 3. 验证答案
        print("\n[步骤3] 验证答案...")
        validation = validate_answer(ground_truth, ai_response, test_case)
        
        print(f"  验证结果: {'✓ 正确' if validation['correct'] else '✗ 错误'}")
        print(f"  得分: {validation['score']*100:.0f}分")
        for detail in validation["details"]:
            print(f"    {detail}")
        
        return {
            "id": test_case["id"],
            "category": test_case["category"],
            "query": test_case["query"],
            "success": True,
            "response_time": response_time,
            "ai_response": ai_response,
            "ground_truth": ground_truth,
            "validation": validation
        }
        
    except Exception as e:
        import traceback
        print(f"  [错误] {e}")
        traceback.print_exc()
        return {
            "id": test_case["id"],
            "success": False,
            "error": str(e)
        }

def main():
    """主函数"""
    print("="*70)
    print("AI Data Router Agent - 带答案验证的测试")
    print("="*70)
    print(f"测试用例数: {len(SAMPLE_TEST_CASES)}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = []
    
    for test_case in SAMPLE_TEST_CASES:
        result = run_test_with_validation(test_case)
        results.append(result)
        time.sleep(1)
    
    # 统计结果
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    total = len(results)
    success = sum(1 for r in results if r.get("success"))
    correct = sum(1 for r in results if r.get("validation", {}).get("correct"))
    
    print(f"总测试数: {total}")
    print(f"成功执行: {success}/{total} ({success/total*100:.1f}%)")
    print(f"答案正确: {correct}/{total} ({correct/total*100:.1f}%)")
    
    # 保存结果
    result_file = f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存: {result_file}")
    
    mongo_client.close()

if __name__ == "__main__":
    main()
