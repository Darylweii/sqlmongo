"""
完整的答案验证测试
1. 先查询数据库获取正确答案
2. 让AI回答同样的问题
3. 对比两个答案
"""
import os
import sys
import json
import time
from datetime import datetime, timedelta

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
from src.tools.device_tool import find_device_metadata_with_engine
from langchain_openai import ChatOpenAI
import httpx

# 初始化组件
config = load_config()
metadata_engine = MetadataEngine(config.mysql.connection_string)
mongo_client = MongoClient(config.mongodb.uri)
db = mongo_client[config.mongodb.database_name]
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

# 20个测试用例
TEST_CASES = [
    # 基础查询（1-5）
    {
        "id": 1,
        "query": "a1_b9 设备 2024年1月的电量",
        "type": "sensor_data",
        "device": "a1_b9",
        "data_type": "ep",
        "time_range": ("2024-01-01", "2024-01-31")
    },
    {
        "id": 2,
        "query": "查询 b1_b14 设备 2024年1月的电流数据",
        "type": "sensor_data",
        "device": "b1_b14",
        "data_type": "i",
        "time_range": ("2024-01-01", "2024-01-31")
    },
    {
        "id": 3,
        "query": "a1_b9 设备 2024年1月5日到1月10日的电量",
        "type": "sensor_data",
        "device": "a1_b9",
        "data_type": "ep",
        "time_range": ("2024-01-05", "2024-01-10")
    },
    {
        "id": 4,
        "query": "a2_b9 设备 2024年1月的电量数据",
        "type": "sensor_data",
        "device": "a2_b9",
        "data_type": "ep",
        "time_range": ("2024-01-01", "2024-01-31")
    },
    {
        "id": 5,
        "query": "b1_b9 设备 2024年1月的电量",
        "type": "sensor_data",
        "device": "b1_b9",
        "data_type": "ep",
        "time_range": ("2024-01-01", "2024-01-31")
    },
    
    # 设备搜索（6-8）
    {
        "id": 6,
        "query": "搜索包含 b9 的设备",
        "type": "device_search",
        "keyword": "b9"
    },
    {
        "id": 7,
        "query": "搜索包含 b14 的设备",
        "type": "device_search",
        "keyword": "b14"
    },
    {
        "id": 8,
        "query": "有哪些电梯设备？",
        "type": "device_search",
        "keyword": "电梯"
    },
    
    # 项目查询（9-10）
    {
        "id": 9,
        "query": "有哪些项目可用？",
        "type": "project_list"
    },
    {
        "id": 10,
        "query": "哪个项目的设备最多？",
        "type": "project_stats"
    },
    
    # 统计分析（11-15）
    {
        "id": 11,
        "query": "a1_b9 设备2024年1月的平均用电量是多少？",
        "type": "sensor_data",
        "device": "a1_b9",
        "data_type": "ep",
        "time_range": ("2024-01-01", "2024-01-31"),
        "stat_type": "avg"
    },
    {
        "id": 12,
        "query": "a1_b9 设备2024年1月的最大用电量",
        "type": "sensor_data",
        "device": "a1_b9",
        "data_type": "ep",
        "time_range": ("2024-01-01", "2024-01-31"),
        "stat_type": "max"
    },
    {
        "id": 13,
        "query": "a1_b9 设备2024年1月的最小用电量",
        "type": "sensor_data",
        "device": "a1_b9",
        "data_type": "ep",
        "time_range": ("2024-01-01", "2024-01-31"),
        "stat_type": "min"
    },
    {
        "id": 14,
        "query": "b1_b14 设备2024年1月的平均电流",
        "type": "sensor_data",
        "device": "b1_b14",
        "data_type": "i",
        "time_range": ("2024-01-01", "2024-01-31"),
        "stat_type": "avg"
    },
    {
        "id": 15,
        "query": "a2_b9 设备2024年1月的电量统计",
        "type": "sensor_data",
        "device": "a2_b9",
        "data_type": "ep",
        "time_range": ("2024-01-01", "2024-01-31"),
        "stat_type": "all"
    },
    
    # 对比查询（16-18）
    {
        "id": 16,
        "query": "对比 a1_b9 和 b1_b14 的用电量",
        "type": "comparison",
        "devices": ["a1_b9", "b1_b14"],
        "data_type": "ep",
        "time_range": ("2024-01-01", "2024-01-31")
    },
    {
        "id": 17,
        "query": "a1_b9 与 a2_b9 哪个用电更多？",
        "type": "comparison",
        "devices": ["a1_b9", "a2_b9"],
        "data_type": "ep",
        "time_range": ("2024-01-01", "2024-01-31")
    },
    {
        "id": 18,
        "query": "比较 a1_b9、a2_b9、b1_b9 三个设备的用电情况",
        "type": "comparison",
        "devices": ["a1_b9", "a2_b9", "b1_b9"],
        "data_type": "ep",
        "time_range": ("2024-01-01", "2024-01-31")
    },
    
    # 复杂查询（19-20）
    {
        "id": 19,
        "query": "a1_b9 设备2024年1月的总用电量",
        "type": "sensor_data",
        "device": "a1_b9",
        "data_type": "ep",
        "time_range": ("2024-01-01", "2024-01-31"),
        "stat_type": "total"
    },
    {
        "id": 20,
        "query": "b1_b14 设备2024年1月电流数据的记录数",
        "type": "sensor_data",
        "device": "b1_b14",
        "data_type": "i",
        "time_range": ("2024-01-01", "2024-01-31"),
        "stat_type": "count"
    }
]

def normalize_number(text):
    """标准化数字：移除千位分隔符、小数点后多余的0"""
    import re
    # 移除千位分隔符
    text = text.replace(',', '').replace('，', '')
    # 提取数字
    numbers = re.findall(r'\d+\.?\d*', text)
    return numbers

def number_match(expected, text, tolerance=0.01):
    """
    检查数字是否匹配（允许格式化差异）
    tolerance: 允许的相对误差（默认1%）
    """
    if expected is None:
        return False
    
    # 标准化文本中的数字
    numbers_in_text = normalize_number(text)
    
    expected_float = float(expected)
    
    for num_str in numbers_in_text:
        try:
            num = float(num_str)
            # 检查是否在误差范围内
            if abs(num - expected_float) / max(abs(expected_float), 1) <= tolerance:
                return True
            # 也检查整数形式
            if abs(int(num) - int(expected_float)) <= 1:
                return True
        except:
            continue
    
    return False
    """步骤1: 查询数据库获取正确答案"""
    query = test_case["query"]
    test_type = test_case["type"]
    
    print(f"\n[步骤1] 查询数据库获取正确答案...")
    
    if test_type == "sensor_data":
        # 传感器数据查询
        collection = db["source_data_ep_202401"]
        mongo_query = {
            "device": "a1_b9",
            "logTime": {"$gte": "2024-01-01 00:00:00", "$lte": "2024-01-31 23:59:59"}
        }
        
        cursor = list(collection.find(mongo_query).sort([("logTime", 1), ("_id", 1)]))
        
        if cursor:
            values = [doc["val"] for doc in cursor if "val" in doc]
            ground_truth = {
                "count": len(cursor),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "avg": round(sum(values) / len(values), 2) if values else None,
                "first_time": cursor[0]["logTime"],
                "last_time": cursor[-1]["logTime"]
            }
            
            # 如果问的是哪天用电最多
            if "哪天" in query and "最多" in query:
                # 按天聚合
                daily_data = {}
                for doc in cursor:
                    date = doc["logTime"][:10]  # 取日期部分
                    if date not in daily_data:
                        daily_data[date] = []
                    daily_data[date].append(doc["val"])
                
                # 计算每天的总用电量（最大值-最小值）
                daily_consumption = {}
                for date, vals in daily_data.items():
                    daily_consumption[date] = max(vals) - min(vals)
                
                max_date = max(daily_consumption, key=daily_consumption.get)
                ground_truth["max_consumption_date"] = max_date
                ground_truth["max_consumption_value"] = round(daily_consumption[max_date], 2)
            
            print(f"  数据库结果: {json.dumps(ground_truth, ensure_ascii=False, indent=2)}")
            return ground_truth
    
    elif test_type == "device_search":
        # 设备搜索
        devices = find_device_metadata_with_engine("b9", metadata_engine)
        device_list = [d for d in devices if isinstance(d, dict) and 'device' in d and 'error' not in d]
        
        ground_truth = {
            "count": len(device_list),
            "devices": [d["device"] for d in device_list[:5]]  # 前5个
        }
        
        print(f"  数据库结果: {json.dumps(ground_truth, ensure_ascii=False, indent=2)}")
        return ground_truth
    
    elif test_type == "project_list":
        # 项目列表
        projects = metadata_engine.list_projects()
        
        ground_truth = {
            "count": len(projects),
            "projects": [p["project_name"] for p in projects[:5]]  # 前5个
        }
        
        print(f"  数据库结果: {json.dumps(ground_truth, ensure_ascii=False, indent=2)}")
        return ground_truth
    
    return None

def get_ai_answer(query):
    """步骤2: 让AI回答"""
    print(f"\n[步骤2] AI回答问题...")
    
    start_time = time.time()
    
    try:
        events = list(agent.run_with_progress(query))
        end_time = time.time()
        response_time = round(end_time - start_time, 2)
        
        # 查找最终响应
        ai_response = None
        for event in reversed(events):
            if event.get("type") in ["final_answer", "direct_response"]:
                ai_response = event.get("response", "")
                break
        
        if ai_response:
            print(f"  AI回答: {ai_response[:300]}...")
            print(f"  响应时间: {response_time}s")
            return ai_response, response_time
        else:
            print("  未获取到AI回答")
            return None, response_time
    
    except Exception as e:
        print(f"  错误: {e}")
        return None, 0

def compare_answers(ground_truth, ai_response, test_case):
    """步骤3: 对比答案"""
    print(f"\n[步骤3] 对比答案...")
    
    if not ground_truth or not ai_response:
        print("  无法对比（缺少数据）")
        return {"match": False, "score": 0, "details": []}
    
    test_type = test_case["type"]
    query = test_case["query"]
    details = []
    score = 0
    
    if test_type == "sensor_data":
        # 检查数据量
        gt_count = ground_truth.get("count")
        if gt_count and str(gt_count) in ai_response:
            details.append(f"✓ 数据量匹配: {gt_count}条")
            score += 25
        else:
            details.append(f"✗ 数据量不匹配: 期望{gt_count}条")
        
        # 检查最小值
        gt_min = ground_truth.get("min")
        if gt_min and (str(int(gt_min)) in ai_response or str(gt_min) in ai_response):
            details.append(f"✓ 最小值匹配: {gt_min}")
            score += 25
        else:
            details.append(f"✗ 最小值不匹配: 期望{gt_min}")
        
        # 检查最大值
        gt_max = ground_truth.get("max")
        if gt_max and (str(int(gt_max)) in ai_response or str(gt_max) in ai_response):
            details.append(f"✓ 最大值匹配: {gt_max}")
            score += 25
        else:
            details.append(f"✗ 最大值不匹配: 期望{gt_max}")
        
        # 检查平均值
        gt_avg = ground_truth.get("avg")
        if gt_avg and (str(int(gt_avg)) in ai_response or str(gt_avg) in ai_response):
            details.append(f"✓ 平均值匹配: {gt_avg}")
            score += 25
        else:
            details.append(f"✗ 平均值不匹配: 期望{gt_avg}")
        
        # 如果问的是哪天最多
        if "哪天" in query and "最多" in query:
            max_date = ground_truth.get("max_consumption_date")
            if max_date:
                # 提取日期（可能是 1月X日 或 2024-01-X）
                date_num = max_date.split("-")[-1]  # 取日期数字
                if date_num in ai_response or f"1月{int(date_num)}日" in ai_response:
                    details.append(f"✓ 最大用电日期匹配: {max_date}")
                    score += 25
                else:
                    details.append(f"✗ 最大用电日期不匹配: 期望{max_date}")
    
    elif test_type == "device_search":
        gt_count = ground_truth.get("count")
        if gt_count and (str(gt_count) in ai_response or f"{gt_count}个" in ai_response):
            details.append(f"✓ 设备数量匹配: {gt_count}个")
            score += 50
        else:
            details.append(f"✗ 设备数量不匹配: 期望{gt_count}个")
        
        # 检查是否提到了设备
        if "设备" in ai_response:
            details.append("✓ 提到了设备")
            score += 50
    
    elif test_type == "project_list":
        gt_count = ground_truth.get("count")
        if gt_count and (str(gt_count) in ai_response or f"{gt_count}个" in ai_response):
            details.append(f"✓ 项目数量匹配: {gt_count}个")
            score += 50
        else:
            details.append(f"✗ 项目数量不匹配: 期望{gt_count}个")
        
        # 检查是否提到了项目
        if "项目" in ai_response:
            details.append("✓ 提到了项目")
            score += 50
    
    match = score >= 60  # 60分及格
    
    print(f"  匹配度: {score}分 ({'✓ 通过' if match else '✗ 不通过'})")
    for detail in details:
        print(f"    {detail}")
    
    return {
        "match": match,
        "score": score,
        "details": details
    }

def run_test(test_case):
    """运行单个测试"""
    print("\n" + "="*70)
    print(f"测试 #{test_case['id']}: {test_case['query']}")
    print("="*70)
    
    # 步骤1: 获取正确答案
    ground_truth = get_ground_truth(test_case)
    
    # 步骤2: AI回答
    ai_response, response_time = get_ai_answer(test_case["query"])
    
    # 步骤3: 对比答案
    comparison = compare_answers(ground_truth, ai_response, test_case)
    
    return {
        "id": test_case["id"],
        "query": test_case["query"],
        "ground_truth": ground_truth,
        "ai_response": ai_response,
        "response_time": response_time,
        "comparison": comparison
    }

def main():
    """主函数"""
    print("="*70)
    print("AI答案验证测试 - 完整流程")
    print("="*70)
    print(f"测试用例数: {len(TEST_CASES)}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = []
    
    for test_case in TEST_CASES:
        result = run_test(test_case)
        results.append(result)
        time.sleep(1)  # 避免请求过快
    
    # 统计结果
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    total = len(results)
    passed = sum(1 for r in results if r["comparison"]["match"])
    avg_score = sum(r["comparison"]["score"] for r in results) / total if total > 0 else 0
    avg_time = sum(r["response_time"] for r in results) / total if total > 0 else 0
    
    print(f"总测试数: {total}")
    print(f"通过数: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"平均得分: {avg_score:.1f}分")
    print(f"平均响应时间: {avg_time:.2f}s")
    
    print("\n详细结果:")
    for r in results:
        status = "✓" if r["comparison"]["match"] else "✗"
        print(f"  {status} 测试{r['id']}: {r['comparison']['score']}分 - {r['query'][:30]}...")
    
    # 保存结果
    result_file = f"ground_truth_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存: {result_file}")
    
    mongo_client.close()

if __name__ == "__main__":
    main()
