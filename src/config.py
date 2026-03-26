"""
Configuration management for AI Data Router Agent.

This module provides centralized configuration management for all components
of the AI Data Router Agent system, including database connections, cache settings,
and agent parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import logging

from src.semantic_layer.config import SemanticLayerConfig

logger = logging.getLogger(__name__)

# 自动加载 .env 文件
try:
    from dotenv import load_dotenv
    # 查找项目根目录的 .env 文件
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info("Loaded environment file: %s", env_path)
except ImportError:
    pass  # python-dotenv 未安装时跳过



@dataclass
class MongoDBConfig:
    """MongoDB connection configuration."""
    host: str = "localhost"
    port: int = 27017
    user: Optional[str] = None
    password: Optional[str] = None
    database_name: str = "sensor_db"
    auth_source: str = "admin"
    max_pool_size: int = 100
    min_pool_size: int = 10
    connect_timeout_ms: int = 5000
    server_selection_timeout_ms: int = 5000
    
    @property
    def uri(self) -> str:
        """Generate MongoDB connection URI with proper encoding."""
        from urllib.parse import quote_plus
        if self.user and self.password:
            # URL编码用户名和密码中的特殊字符
            encoded_user = quote_plus(self.user)
            encoded_password = quote_plus(self.password)
            return f"mongodb://{encoded_user}:{encoded_password}@{self.host}:{self.port}/?authSource={self.auth_source}"
        return f"mongodb://{self.host}:{self.port}"


@dataclass
class MySQLConfig:
    """MySQL connection configuration."""
    host: str = "localhost"
    port: int = 3306
    database: str = ""  # 可选，留空则可跨库查询
    user: str = "root"
    password: str = ""
    pool_size: int = 10
    pool_recycle: int = 3600
    
    @property
    def connection_string(self) -> str:
        """Generate SQLAlchemy connection string with proper encoding."""
        from urllib.parse import quote_plus
        # URL编码密码中的特殊字符
        encoded_password = quote_plus(self.password) if self.password else ""
        base_url = f"mysql+pymysql://{self.user}:{encoded_password}@{self.host}:{self.port}"
        if self.database:
            return f"{base_url}/{self.database}"
        return base_url


@dataclass
class RedisConfig:
    """Redis cache configuration."""
    url: str = "redis://localhost:6379"
    default_ttl: int = 3600  # 1 hour
    max_connections: int = 50
    enabled: bool = True


@dataclass
class AgentConfig:
    """Agent behavior configuration."""
    max_collections: int = 50  # Circuit breaker threshold
    max_records: int = 2000  # Downsampling threshold
    max_tokens: int = 4000  # Context compression limit
    long_query_threshold_days: int = 30  # Degradation threshold
    cache_recent_days: int = 7  # Hot data cache window
    metadata_cache_size: int = 1000  # LRU cache size
    orchestrator_type: str = "dag"  # Orchestrator type: "react" or "dag"
    use_llm_sql: bool = False  # 是否使用 LLM 生成 SQL 查询


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: str = "openai"  # openai, anthropic, etc.
    model_name: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # 自定义 API 地址（如 vLLM）
    temperature: float = 0.0
    max_tokens: int = 2000
    timeout: int = 300  # 增加到 5 分钟


@dataclass
class MultiLLMConfig:
    """
    多模型配置 - 支持不同节点使用不同模型
    
    使用场景：
    - 意图解析/总结呈现：使用大模型（如 qwen3-32b）处理自然语言
    - 代码/SQL 生成：使用代码模型（如 qwen2.5-coder-7b）
    """
    # 主模型（用于意图解析和总结呈现）
    main_provider: str = "openai"
    main_model: str = "gpt-3.5-turbo"
    main_base_url: Optional[str] = None
    main_api_key: Optional[str] = None
    
    # 代码模型（用于 SQL/代码生成）
    coder_provider: str = "openai"
    coder_model: str = "gpt-3.5-turbo"
    coder_base_url: Optional[str] = None
    coder_api_key: Optional[str] = None

    # 是否启用多模型路由
    enabled: bool = False

    # 通用参数
    temperature: float = 0.0
    max_tokens: int = 2000
    timeout: int = 300  # 增加到 5 分钟


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None


@dataclass
class AppConfig:
    """
    Main application configuration.
    
    Aggregates all component configurations into a single configuration object.
    Supports loading from environment variables.
    """
    mongodb: MongoDBConfig = field(default_factory=MongoDBConfig)
    mysql: MySQLConfig = field(default_factory=MySQLConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    multi_llm: MultiLLMConfig = field(default_factory=MultiLLMConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    semantic_layer: SemanticLayerConfig = field(default_factory=SemanticLayerConfig)
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """
        Create configuration from environment variables.
        
        Environment variables:
            MONGODB_URI: MongoDB connection URI
            MONGODB_DATABASE: MongoDB database name
            MYSQL_HOST: MySQL host
            MYSQL_PORT: MySQL port
            MYSQL_DATABASE: MySQL database name
            MYSQL_USER: MySQL username
            MYSQL_PASSWORD: MySQL password
            REDIS_URL: Redis connection URL
            REDIS_ENABLED: Enable/disable Redis cache (true/false)
            LLM_PROVIDER: LLM provider name
            LLM_MODEL: LLM model name
            LLM_API_KEY: LLM API key
            LOG_LEVEL: Logging level
            LOG_FILE: Log file path
            ORCHESTRATOR_TYPE: Orchestrator type ("react" or "dag", default "dag")
            SEMANTIC_LAYER_ENABLED: Enable/disable semantic layer (true/false)
            DASHSCOPE_API_KEY: Alibaba Cloud DashScope API key for semantic layer
        
        Returns:
            AppConfig instance with values from environment
        """
        mongodb = MongoDBConfig(
            host=os.getenv("MONGODB_HOST", "localhost"),
            port=int(os.getenv("MONGODB_PORT", "27017")),
            user=os.getenv("MONGODB_USER"),
            password=os.getenv("MONGODB_PASSWORD"),
            database_name=os.getenv("MONGODB_DATABASE", "sensor_db"),
            auth_source=os.getenv("MONGODB_AUTH_SOURCE", "admin"),
        )
        
        mysql = MySQLConfig(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            database=os.getenv("MYSQL_DATABASE", ""),  # 可选
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
        )
        
        redis = RedisConfig(
            url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            enabled=os.getenv("REDIS_ENABLED", "true").lower() == "true",
        )
        
        llm = LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            model_name=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        )
        
        # 多模型配置
        multi_llm = MultiLLMConfig(
            enabled=os.getenv("MULTI_LLM_ENABLED", "false").lower() == "true",
            # 主模型（自然语言处理）
            main_provider=os.getenv("MAIN_LLM_PROVIDER", "openai"),
            main_model=os.getenv("MAIN_LLM_MODEL", "qwen3-32b"),
            main_base_url=os.getenv("MAIN_LLM_BASE_URL"),
            main_api_key=os.getenv("MAIN_LLM_API_KEY") or os.getenv("LLM_API_KEY"),
            # 代码模型
            coder_provider=os.getenv("CODER_LLM_PROVIDER", "openai"),
            coder_model=os.getenv("CODER_LLM_MODEL", "qwen2.5-coder-7b"),
            coder_base_url=os.getenv("CODER_LLM_BASE_URL"),
            coder_api_key=os.getenv("CODER_LLM_API_KEY") or os.getenv("LLM_API_KEY"),
        )
        
        logging_config = LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            file_path=os.getenv("LOG_FILE"),
        )
        
        # Agent configuration with orchestrator type
        agent = AgentConfig(
            orchestrator_type=os.getenv("ORCHESTRATOR_TYPE", "dag"),
            use_llm_sql=os.getenv("USE_LLM_SQL", "false").lower() == "true",
        )
        
        # Semantic layer configuration
        semantic_layer = SemanticLayerConfig.from_env()
        
        return cls(
            mongodb=mongodb,
            mysql=mysql,
            redis=redis,
            agent=agent,
            llm=llm,
            multi_llm=multi_llm,
            logging=logging_config,
            semantic_layer=semantic_layer,
        )
    
    def validate(self) -> bool:
        """
        Validate configuration values.
        
        Returns:
            True if configuration is valid
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.mongodb.host:
            raise ValueError("MongoDB host is required")
        
        if not self.mysql.host:
            raise ValueError("MySQL host is required")
        
        if self.agent.max_collections <= 0:
            raise ValueError("max_collections must be positive")
        
        if self.agent.max_records <= 0:
            raise ValueError("max_records must be positive")
        
        if self.agent.orchestrator_type not in ("react", "dag"):
            raise ValueError(f"orchestrator_type must be 'react' or 'dag', got '{self.agent.orchestrator_type}'")
        
        return True


def load_config() -> AppConfig:
    """
    Load application configuration.
    
    Attempts to load from environment variables first,
    falls back to defaults if not available.
    
    Returns:
        AppConfig instance
    """
    config = AppConfig.from_env()
    config.validate()
    return config


def setup_logging(config: LoggingConfig) -> None:
    """
    Setup logging based on configuration.
    
    Args:
        config: LoggingConfig instance
    """
    handlers = [logging.StreamHandler()]
    
    if config.file_path:
        handlers.append(logging.FileHandler(config.file_path))
    
    logging.basicConfig(
        level=getattr(logging, config.level.upper(), logging.INFO),
        format=config.format,
        handlers=handlers,
    )
    
    logger.info(f"Logging configured at level: {config.level}")
