"""
SQL Generator Node - SQL 生成节点

使用代码模型（如 qwen2.5-coder）根据用户意图生成 SQL 查询语句。

功能：
- 根据自然语言意图生成 SQL 查询
- 支持设备查询、项目查询等场景
- 防止 SQL 注入

需求引用：
- 需求 3.2: 查询 MySQL 数据库获取设备信息
"""

import logging
import re
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from src.agent.types import GraphState, NODE_METADATA_MAPPER


logger = logging.getLogger(__name__)


# 数据库 Schema 信息，供 LLM 参考
DATABASE_SCHEMA = """
## 数据库表结构

### 设备表: device.device_info
| 字段 | 类型 | 说明 |
|------|------|------|
| id | INT | 主键 |
| device_name | VARCHAR | 设备名称（中文） |
| device_type | VARCHAR | 设备类型 |
| device | VARCHAR | 设备代号（用于 MongoDB 查询） |
| project_id | INT | 关联项目ID |
| asset_number | VARCHAR | 资产编号 |
| tg | VARCHAR | 设备分组 |

### 项目表: project.project_info
| 字段 | 类型 | 说明 |
|------|------|------|
| id | INT | 主键 |
| project_name | VARCHAR | 项目名称 |
| project_code_name | VARCHAR | 项目代号 |
| enable | INT | 是否启用 (1=启用) |

### 关联关系
- device.device_info.project_id -> project.project_info.id
"""


class SQLGeneratorNode:
    """
    SQL 生成节点 - 使用代码模型生成 SQL
    
    根据用户的自然语言意图，生成合适的 SQL 查询语句。
    
    Attributes:
        llm: 代码模型 LLM 实例（推荐使用 qwen2.5-coder）
    """
    
    SYSTEM_PROMPT = f"""你是一个 SQL 专家。根据用户的查询意图，生成 MySQL 查询语句。

{DATABASE_SCHEMA}

## 输出要求
1. 只输出 SQL 语句，不要有其他解释
2. 必须返回以下字段：device, device_name, device_type, project_id, project_name, project_code_name
3. 使用 LEFT JOIN 连接设备表和项目表
4. 不要使用 DELETE、UPDATE、INSERT、DROP 等危险操作
5. 查询结果限制在 100 条以内

## 示例

用户意图: 查找电梯
SQL:
```sql
SELECT d.device, d.device_name, d.device_type, d.project_id, p.project_name, p.project_code_name
FROM device.device_info d
LEFT JOIN project.project_info p ON d.project_id = p.id
WHERE d.device_name LIKE '%电梯%' OR d.device_type LIKE '%电梯%'
LIMIT 100
```

用户意图: 查找项目A的所有设备
SQL:
```sql
SELECT d.device, d.device_name, d.device_type, d.project_id, p.project_name, p.project_code_name
FROM device.device_info d
LEFT JOIN project.project_info p ON d.project_id = p.id
WHERE p.project_name LIKE '%项目A%' OR p.project_code_name LIKE '%项目A%'
LIMIT 100
```
"""
    
    def __init__(self, llm: BaseChatModel):
        """
        初始化 SQL 生成节点
        
        Args:
            llm: 代码模型 LLM 实例
        """
        self.llm = llm
    
    def generate_sql(self, intent: dict) -> tuple[Optional[str], Optional[str]]:
        """
        根据意图生成 SQL 查询
        
        Args:
            intent: 解析后的用户意图
        
        Returns:
            (sql, error) - SQL 语句或错误信息
        """
        target = intent.get("target", "")
        if not target:
            return None, "未指定查询目标"
        
        # 构建查询描述
        query_desc = f"查找 {target}"
        
        # 如果有额外条件，添加到描述中
        if intent.get("device_type"):
            query_desc += f"，设备类型为 {intent['device_type']}"
        if intent.get("project"):
            query_desc += f"，属于项目 {intent['project']}"
        
        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=f"用户意图: {query_desc}\nSQL:")
            ]
            
            response = self.llm.invoke(messages)
            sql = self._extract_sql(response.content)
            
            if not sql:
                return None, "无法生成有效的 SQL 查询"
            
            # 安全检查
            if not self._is_safe_sql(sql):
                return None, "生成的 SQL 包含不安全的操作"
            
            logger.info(f"生成 SQL: {sql[:100]}...")
            return sql, None
            
        except Exception as e:
            logger.error(f"SQL 生成失败: {e}")
            return None, f"SQL 生成失败: {str(e)}"
    
    def _extract_sql(self, response: str) -> Optional[str]:
        """从 LLM 响应中提取 SQL 语句"""
        # 尝试从代码块中提取
        code_block_pattern = r'```(?:sql)?\s*([\s\S]*?)```'
        matches = re.findall(code_block_pattern, response, re.IGNORECASE)
        if matches:
            return matches[0].strip()
        
        # 尝试直接提取 SELECT 语句
        select_pattern = r'(SELECT[\s\S]+?(?:LIMIT\s+\d+|;|$))'
        matches = re.findall(select_pattern, response, re.IGNORECASE)
        if matches:
            return matches[0].strip().rstrip(';')
        
        return None
    
    def _is_safe_sql(self, sql: str) -> bool:
        """检查 SQL 是否安全"""
        sql_upper = sql.upper()
        
        # 禁止的关键词
        dangerous_keywords = [
            'DELETE', 'UPDATE', 'INSERT', 'DROP', 'TRUNCATE',
            'ALTER', 'CREATE', 'GRANT', 'REVOKE', 'EXEC',
            'EXECUTE', 'UNION', '--', '/*', '*/'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                logger.warning(f"SQL 包含危险关键词: {keyword}")
                return False
        
        # 必须是 SELECT 语句
        if not sql_upper.strip().startswith('SELECT'):
            return False
        
        return True
