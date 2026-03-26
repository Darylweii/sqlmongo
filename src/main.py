"""
Main application entry point for AI Data Router Agent.

This module provides the main entry point for the AI Data Router Agent system,
initializing all components and providing both CLI and programmatic interfaces.
"""

import asyncio
import logging
from typing import Optional, Union

from pymongo import MongoClient

from src.config import AppConfig, load_config, setup_logging
from src.agent.orchestrator import AgentOrchestrator, LLMAgent, create_agent
from src.agent.dag_orchestrator import DAGOrchestrator
from src.agent import create_orchestrator
from src.metadata.metadata_engine import MetadataEngine
from src.fetcher.data_fetcher import DataFetcher
from src.cache.cache_manager import CacheManager
from src.compressor.context_compressor import ContextCompressor
from src.tools.device_tool import configure_metadata_engine
from src.tools.sensor_tool import configure_sensor_tool


logger = logging.getLogger(__name__)


class AIDataRouterAgent:
    """
    Main application class for AI Data Router Agent.
    
    Provides a unified interface to initialize and run the agent system,
    managing all component lifecycles.
    
    Attributes:
        config: Application configuration
        metadata_engine: MetadataEngine instance
        data_fetcher: DataFetcher instance
        cache_manager: CacheManager instance (optional)
        compressor: ContextCompressor instance
        agent: AgentOrchestrator, LLMAgent, or DAGOrchestrator instance
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize the AI Data Router Agent.
        
        Args:
            config: Application configuration (loads from env if None)
        """
        self.config = config or load_config()
        self._mongo_client: Optional[MongoClient] = None
        self._metadata_engine: Optional[MetadataEngine] = None
        self._data_fetcher: Optional[DataFetcher] = None
        self._cache_manager: Optional[CacheManager] = None
        self._compressor: Optional[ContextCompressor] = None
        self._agent: Optional[Union[LLMAgent, DAGOrchestrator]] = None
        self._initialized = False
    
    def initialize(self) -> None:
        """
        Initialize all components.
        
        Sets up database connections, cache, and agent orchestrator.
        Must be called before using the agent.
        """
        if self._initialized:
            logger.warning("Agent already initialized")
            return
        
        # Setup logging
        setup_logging(self.config.logging)
        logger.info("Initializing AI Data Router Agent...")
        
        # Initialize MongoDB client
        logger.info(f"Connecting to MongoDB: {self.config.mongodb.uri}")
        self._mongo_client = MongoClient(
            self.config.mongodb.uri,
            maxPoolSize=self.config.mongodb.max_pool_size,
            minPoolSize=self.config.mongodb.min_pool_size,
            connectTimeoutMS=self.config.mongodb.connect_timeout_ms,
            serverSelectionTimeoutMS=self.config.mongodb.server_selection_timeout_ms,
        )
        
        # Initialize MetadataEngine
        logger.info(f"Connecting to MySQL: {self.config.mysql.host}")
        self._metadata_engine = MetadataEngine(
            db_connection_string=self.config.mysql.connection_string,
            cache_size=self.config.agent.metadata_cache_size,
        )
        
        # Configure global metadata engine for tools
        configure_metadata_engine(
            db_connection_string=self.config.mysql.connection_string,
            cache_size=self.config.agent.metadata_cache_size,
        )
        
        # Initialize DataFetcher
        self._data_fetcher = DataFetcher(
            mongo_client=self._mongo_client,
            database_name=self.config.mongodb.database_name,
            max_records=self.config.agent.max_records,
        )
        
        # Initialize CacheManager (optional)
        if self.config.redis.enabled:
            logger.info(f"Connecting to Redis: {self.config.redis.url}")
            try:
                self._cache_manager = CacheManager(
                    redis_url=self.config.redis.url,
                    default_ttl=self.config.redis.default_ttl,
                )
            except Exception as e:
                logger.warning(f"Failed to connect to Redis, caching disabled: {e}")
                self._cache_manager = None
        
        # Initialize ContextCompressor
        self._compressor = ContextCompressor(
            max_tokens=self.config.agent.max_tokens,
        )
        
        # Configure global sensor tool
        configure_sensor_tool(
            mongo_uri=self.config.mongodb.uri,
            database_name=self.config.mongodb.database_name,
            redis_url=self.config.redis.url if self.config.redis.enabled else None,
            max_records=self.config.agent.max_records,
            max_tokens=self.config.agent.max_tokens,
        )
        
        # Initialize LLM(s)
        llm = self._create_main_llm()
        coder_llm = self._create_coder_llm()
        
        # Initialize Agent Orchestrator using factory function
        # Supports both ReAct and DAG orchestrators based on configuration
        orchestrator_type = self.config.agent.orchestrator_type
        
        if self.config.multi_llm.enabled:
            logger.info(f"Creating {orchestrator_type.upper()} orchestrator with multi-model support...")
            logger.info(f"  Main LLM: {self.config.multi_llm.main_model}")
            logger.info(f"  Coder LLM: {self.config.multi_llm.coder_model}")
        else:
            logger.info(f"Creating {orchestrator_type.upper()} orchestrator...")
        
        self._agent = create_orchestrator(
            llm=llm,
            metadata_engine=self._metadata_engine,
            data_fetcher=self._data_fetcher,
            cache_manager=self._cache_manager,
            compressor=self._compressor,
            orchestrator_type=orchestrator_type,
            coder_llm=coder_llm,
        )
        
        self._initialized = True
        logger.info(f"AI Data Router Agent initialized successfully with {orchestrator_type.upper()} orchestrator")
    
    def _create_llm(self, provider: str = None, model: str = None, base_url: str = None, api_key: str = None):
        """
        Create LLM instance based on configuration.
        
        Args:
            provider: LLM provider (openai, anthropic, etc.)
            model: Model name
            base_url: Custom API base URL (for vLLM, etc.)
            api_key: API key
        
        Returns:
            LLM instance
        """
        provider = (provider or self.config.llm.provider).lower()
        model = model or self.config.llm.model_name
        base_url = base_url or self.config.llm.base_url
        api_key = api_key or self.config.llm.api_key
        
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            request_timeout = min(max(self.config.llm.timeout, 10), 20)
            kwargs = {
                "model": model,
                "temperature": self.config.llm.temperature,
                "max_tokens": self.config.llm.max_tokens,
                "timeout": request_timeout,
                "max_retries": 0,
            }
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
            return ChatOpenAI(**kwargs)
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model,
                api_key=api_key,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )
        else:
            # Fallback to a mock LLM for testing
            from langchain_core.language_models import FakeListChatModel
            logger.warning(f"Unknown LLM provider '{provider}', using mock LLM")
            return FakeListChatModel(responses=["Mock response"])
    
    def _create_coder_llm(self):
        """
        Create coder LLM instance for code/SQL generation.
        
        Returns:
            LLM instance for code generation, or None if not configured
        """
        multi_llm = self.config.multi_llm
        if not multi_llm.enabled:
            return None
        
        logger.info(f"Creating coder LLM: {multi_llm.coder_model}")
        return self._create_llm(
            provider=multi_llm.coder_provider,
            model=multi_llm.coder_model,
            base_url=multi_llm.coder_base_url,
            api_key=multi_llm.coder_api_key,
        )
    
    def _create_main_llm(self):
        """
        Create main LLM instance for natural language processing.
        
        Returns:
            LLM instance
        """
        multi_llm = self.config.multi_llm
        if multi_llm.enabled:
            logger.info(f"Creating main LLM: {multi_llm.main_model}")
            return self._create_llm(
                provider=multi_llm.main_provider,
                model=multi_llm.main_model,
                base_url=multi_llm.main_base_url,
                api_key=multi_llm.main_api_key,
            )
        else:
            return self._create_llm()
    
    @property
    def agent(self) -> Union[LLMAgent, DAGOrchestrator]:
        """Get the agent orchestrator instance."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self._agent
    
    @property
    def metadata_engine(self) -> MetadataEngine:
        """Get the metadata engine instance."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self._metadata_engine
    
    @property
    def data_fetcher(self) -> DataFetcher:
        """Get the data fetcher instance."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self._data_fetcher
    
    @property
    def cache_manager(self) -> Optional[CacheManager]:
        """Get the cache manager instance (may be None)."""
        return self._cache_manager
    
    def query(self, user_query: str) -> str:
        """
        Process a user query synchronously.
        
        Args:
            user_query: Natural language query from user
        
        Returns:
            Agent response string
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        logger.info(f"Processing query: {user_query[:100]}...")
        return self._agent.run(user_query)
    
    async def query_async(self, user_query: str) -> str:
        """
        Process a user query asynchronously.
        
        Args:
            user_query: Natural language query from user
        
        Returns:
            Agent response string
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        logger.info(f"Processing query (async): {user_query[:100]}...")
        return await self._agent.run_async(user_query)
    
    def close(self) -> None:
        """
        Close all connections and cleanup resources.
        """
        logger.info("Shutting down AI Data Router Agent...")
        
        if self._cache_manager:
            self._cache_manager.close()
        
        if self._mongo_client:
            self._mongo_client.close()
        
        self._initialized = False
        logger.info("AI Data Router Agent shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def create_app(config: Optional[AppConfig] = None) -> AIDataRouterAgent:
    """
    Factory function to create an AIDataRouterAgent instance.
    
    Args:
        config: Optional configuration (loads from env if None)
    
    Returns:
        AIDataRouterAgent instance (not initialized)
    """
    return AIDataRouterAgent(config)


def run_cli():
    """
    Run the agent in CLI mode.
    
    Provides an interactive command-line interface for querying the agent.
    """
    print("=" * 60)
    print("AI Data Router Agent - Interactive CLI")
    print("=" * 60)
    print("Type 'quit' or 'exit' to exit")
    print("Type 'help' for usage information")
    print("-" * 60)
    
    try:
        app = create_app()
        app.initialize()
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ("quit", "exit"):
                    print("Goodbye!")
                    break
                
                if user_input.lower() == "help":
                    print("""
Available commands:
  - Type any natural language query to search devices or data
  - 'quit' or 'exit': Exit the CLI
  - 'help': Show this help message

Example queries:
  - "查找温度传感器"
  - "列出项目A的所有设备"
  - "查看设备temp_001最近一周的数据"
""")
                    continue
                
                response = app.query(user_input)
                print(f"\n{response}")
                
            except KeyboardInterrupt:
                print("\nInterrupted. Type 'quit' to exit.")
            except Exception as e:
                print(f"\nError: {e}")
        
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        return 1
    finally:
        if 'app' in locals():
            app.close()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(run_cli())
