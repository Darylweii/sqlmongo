"""
Virtual Schema management for the Semantic Layer.

This module provides data type definitions and schema management functionality,
including YAML configuration loading, validation, and hot reload support.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.semantic_layer.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class DataTypeDefinition:
    """
    Data type definition for the semantic layer.
    
    Represents a single data type with its metadata, synonyms, and relationships.
    This structure maps to the YAML configuration format.
    
    Attributes:
        id: Unique identifier for the data type (e.g., "ep", "i", "u_line")
        collection_prefix: MongoDB collection prefix (e.g., "source_data_ep_")
        name: Human-readable Chinese name (e.g., "电量", "电流")
        synonyms: List of alternative names and colloquial expressions
        unit: Optional measurement unit (e.g., "kWh", "A", "V")
        description: Optional detailed description
        tags: List of associated tag field values for MongoDB queries
        related_types: List of related data type IDs for cross-reference
    """
    
    id: str
    collection_prefix: str
    name: str
    synonyms: List[str] = field(default_factory=list)
    unit: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    related_types: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate required fields after initialization."""
        if not self.id:
            raise ValueError("DataTypeDefinition.id cannot be empty")
        if not self.collection_prefix:
            raise ValueError("DataTypeDefinition.collection_prefix cannot be empty")
        if not self.name:
            raise ValueError("DataTypeDefinition.name cannot be empty")
    
    def get_all_searchable_terms(self) -> List[str]:
        """
        Get all searchable terms for this data type.
        
        Returns a list containing the name and all synonyms,
        which can be used for vector indexing.
        
        Returns:
            List of all searchable terms
        """
        terms = [self.name]
        terms.extend(self.synonyms)
        return terms
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for YAML serialization.
        
        Returns:
            Dictionary representation of the data type
        """
        result: Dict[str, Any] = {
            "id": self.id,
            "collection_prefix": self.collection_prefix,
            "name": self.name,
        }
        
        if self.synonyms:
            result["synonyms"] = self.synonyms
        if self.unit is not None:
            result["unit"] = self.unit
        if self.description is not None:
            result["description"] = self.description
        if self.tags:
            result["tags"] = self.tags
        if self.related_types:
            result["related_types"] = self.related_types
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataTypeDefinition":
        """
        Create a DataTypeDefinition from a dictionary.
        
        Args:
            data: Dictionary containing data type fields
        
        Returns:
            DataTypeDefinition instance
        
        Raises:
            ConfigurationError: If required fields are missing
        """
        required_fields = ["id", "collection_prefix", "name"]
        missing_fields = [f for f in required_fields if f not in data or not data[f]]
        
        if missing_fields:
            raise ConfigurationError(
                f"Missing required fields in data type definition: {missing_fields}",
                details={"data": data, "missing_fields": missing_fields}
            )
        
        return cls(
            id=data["id"],
            collection_prefix=data["collection_prefix"],
            name=data["name"],
            synonyms=data.get("synonyms", []),
            unit=data.get("unit"),
            description=data.get("description"),
            tags=data.get("tags", []),
            related_types=data.get("related_types", []),
        )



@dataclass
class VirtualSchemaSettings:
    """
    Global settings for the virtual schema.
    
    Attributes:
        similarity_threshold: Minimum similarity score for valid matches
        top_k_retrieval: Number of candidates from vector search
        top_n_rerank: Number of results after reranking
    """
    
    similarity_threshold: float = 0.5
    top_k_retrieval: int = 10
    top_n_rerank: int = 5
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VirtualSchemaSettings":
        """Create settings from dictionary."""
        return cls(
            similarity_threshold=data.get("similarity_threshold", 0.5),
            top_k_retrieval=data.get("top_k_retrieval", 10),
            top_n_rerank=data.get("top_n_rerank", 5),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "similarity_threshold": self.similarity_threshold,
            "top_k_retrieval": self.top_k_retrieval,
            "top_n_rerank": self.top_n_rerank,
        }


class VirtualSchemaManager:
    """
    Virtual Schema Manager for loading and managing data type configurations.
    
    This class handles:
    - Loading YAML configuration files
    - Validating configuration structure
    - Providing access to data type definitions
    - Hot reload support for configuration changes
    
    Thread-safe implementation for concurrent access.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the VirtualSchemaManager.
        
        Args:
            config_path: Path to the YAML configuration file.
                        If None, uses default path "config/semantic_layer.yaml"
        """
        self._config_path = config_path or "config/semantic_layer.yaml"
        self._data_types: Dict[str, DataTypeDefinition] = {}
        self._settings: VirtualSchemaSettings = VirtualSchemaSettings()
        self._version: str = "1.0"
        self._lock = threading.RLock()
        self._last_modified: float = 0.0
        self._hot_reload_thread: Optional[threading.Thread] = None
        self._hot_reload_stop_event = threading.Event()
        self._initialized: bool = False
    
    @property
    def config_path(self) -> str:
        """Get the configuration file path."""
        return self._config_path
    
    @property
    def settings(self) -> VirtualSchemaSettings:
        """Get the current settings."""
        with self._lock:
            return self._settings
    
    @property
    def version(self) -> str:
        """Get the configuration version."""
        with self._lock:
            return self._version
    
    @property
    def is_initialized(self) -> bool:
        """Check if the manager has been initialized."""
        return self._initialized
    
    def load_config(self, config_path: Optional[str] = None) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Optional path to override the default config path
        
        Raises:
            ConfigurationError: If the file cannot be loaded or parsed
        """
        path = config_path or self._config_path
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigurationError(
                f"Configuration file not found: {path}",
                details={"path": path}
            )
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML syntax in configuration file: {path}",
                details={"path": path, "error": str(e)}
            )
        except PermissionError:
            raise ConfigurationError(
                f"Permission denied reading configuration file: {path}",
                details={"path": path}
            )
        
        if raw_config is None:
            raise ConfigurationError(
                f"Configuration file is empty: {path}",
                details={"path": path}
            )
        
        # Parse and validate the configuration
        self._parse_config(raw_config)
        
        # Update last modified time
        try:
            self._last_modified = os.path.getmtime(path)
        except OSError:
            self._last_modified = time.time()
        
        self._initialized = True
        logger.info(f"Loaded configuration from {path} with {len(self._data_types)} data types")
    
    def _parse_config(self, raw_config: Dict[str, Any]) -> None:
        """
        Parse raw configuration dictionary.
        
        Args:
            raw_config: Raw configuration from YAML
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        with self._lock:
            # Parse version
            self._version = str(raw_config.get("version", "1.0"))
            
            # Parse settings
            settings_data = raw_config.get("settings", {})
            self._settings = VirtualSchemaSettings.from_dict(settings_data)
            
            # Parse data types
            data_types_list = raw_config.get("data_types", [])
            if not isinstance(data_types_list, list):
                raise ConfigurationError(
                    "data_types must be a list",
                    details={"type": type(data_types_list).__name__}
                )
            
            new_data_types: Dict[str, DataTypeDefinition] = {}
            errors: List[str] = []
            
            for idx, dt_data in enumerate(data_types_list):
                try:
                    dt = DataTypeDefinition.from_dict(dt_data)
                    if dt.id in new_data_types:
                        errors.append(f"Duplicate data type ID: {dt.id}")
                    else:
                        new_data_types[dt.id] = dt
                except (ConfigurationError, ValueError) as e:
                    errors.append(f"Data type at index {idx}: {str(e)}")
            
            if errors:
                raise ConfigurationError(
                    f"Configuration validation failed with {len(errors)} error(s)",
                    details={"errors": errors}
                )
            
            self._data_types = new_data_types
    
    def reload_config(self) -> bool:
        """
        Reload configuration from file.
        
        Implements graceful handling: if the new configuration is invalid,
        the previous valid configuration is retained.
        
        Returns:
            True if reload was successful, False otherwise
        """
        try:
            # Store current state for rollback
            with self._lock:
                old_data_types = self._data_types.copy()
                old_settings = self._settings
                old_version = self._version
            
            # Attempt to load new configuration
            self.load_config()
            logger.info("Configuration reloaded successfully")
            return True
            
        except ConfigurationError as e:
            # Rollback to previous configuration
            with self._lock:
                self._data_types = old_data_types
                self._settings = old_settings
                self._version = old_version
            logger.error(f"Configuration reload failed, keeping previous config: {e}")
            return False
        except Exception as e:
            # Rollback to previous configuration
            with self._lock:
                self._data_types = old_data_types
                self._settings = old_settings
                self._version = old_version
            logger.error(f"Unexpected error during config reload: {e}")
            return False
    
    def get_data_type(self, type_id: str) -> Optional[DataTypeDefinition]:
        """
        Get a data type definition by ID.
        
        Args:
            type_id: The data type identifier
        
        Returns:
            DataTypeDefinition if found, None otherwise
        """
        with self._lock:
            return self._data_types.get(type_id)
    
    def get_all_data_types(self) -> List[DataTypeDefinition]:
        """
        Get all data type definitions.
        
        Returns:
            List of all DataTypeDefinition instances
        """
        with self._lock:
            return list(self._data_types.values())
    
    def get_collection_prefix(self, type_id: str) -> str:
        """
        Get the collection prefix for a data type.
        
        This method provides backward compatibility with the existing
        get_collection_prefix() function in collection_router.py.
        
        Args:
            type_id: The data type identifier
        
        Returns:
            Collection prefix string, defaults to "source_data_ep_" if not found
        """
        with self._lock:
            dt = self._data_types.get(type_id)
            if dt:
                return dt.collection_prefix
            return "source_data_ep_"
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """
        Validate the current configuration.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors: List[str] = []
        
        with self._lock:
            if not self._data_types:
                errors.append("No data types defined")
            
            # Check for required fields in each data type
            for type_id, dt in self._data_types.items():
                if not dt.id:
                    errors.append(f"Data type missing id")
                if not dt.collection_prefix:
                    errors.append(f"Data type '{type_id}' missing collection_prefix")
                if not dt.name:
                    errors.append(f"Data type '{type_id}' missing name")
                
                # Validate related_types references
                for related_id in dt.related_types:
                    if related_id not in self._data_types:
                        errors.append(
                            f"Data type '{type_id}' references unknown related_type: {related_id}"
                        )
            
            # Validate settings
            if self._settings.similarity_threshold < 0 or self._settings.similarity_threshold > 1:
                errors.append(
                    f"similarity_threshold must be between 0 and 1, got {self._settings.similarity_threshold}"
                )
            if self._settings.top_k_retrieval <= 0:
                errors.append(f"top_k_retrieval must be positive, got {self._settings.top_k_retrieval}")
            if self._settings.top_n_rerank <= 0:
                errors.append(f"top_n_rerank must be positive, got {self._settings.top_n_rerank}")
        
        return (len(errors) == 0, errors)
    
    def start_hot_reload(self, interval_seconds: int = 60) -> None:
        """
        Start the hot reload background thread.
        
        Args:
            interval_seconds: Interval between config file checks
        """
        if self._hot_reload_thread is not None and self._hot_reload_thread.is_alive():
            logger.warning("Hot reload thread is already running")
            return
        
        self._hot_reload_stop_event.clear()
        self._hot_reload_thread = threading.Thread(
            target=self._hot_reload_loop,
            args=(interval_seconds,),
            daemon=True,
            name="VirtualSchemaHotReload"
        )
        self._hot_reload_thread.start()
        logger.info(f"Started hot reload thread with {interval_seconds}s interval")
    
    def stop_hot_reload(self) -> None:
        """Stop the hot reload background thread."""
        if self._hot_reload_thread is None:
            return
        
        self._hot_reload_stop_event.set()
        self._hot_reload_thread.join(timeout=5.0)
        self._hot_reload_thread = None
        logger.info("Stopped hot reload thread")
    
    def _hot_reload_loop(self, interval_seconds: int) -> None:
        """
        Background loop for hot reload.
        
        Args:
            interval_seconds: Interval between checks
        """
        while not self._hot_reload_stop_event.is_set():
            try:
                # Check if file has been modified
                current_mtime = os.path.getmtime(self._config_path)
                if current_mtime > self._last_modified:
                    logger.info("Configuration file changed, reloading...")
                    self.reload_config()
            except FileNotFoundError:
                logger.warning(f"Configuration file not found: {self._config_path}")
            except Exception as e:
                logger.error(f"Error in hot reload loop: {e}")
            
            # Wait for next check or stop event
            self._hot_reload_stop_event.wait(timeout=interval_seconds)
    
    def to_yaml(self) -> str:
        """
        Serialize current configuration to YAML string.
        
        Returns:
            YAML string representation of the configuration
        """
        with self._lock:
            config = {
                "version": self._version,
                "settings": self._settings.to_dict(),
                "data_types": [dt.to_dict() for dt in self._data_types.values()],
            }
            return yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    def save_config(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            path: Optional path to save to, defaults to config_path
        """
        save_path = path or self._config_path
        
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        yaml_content = self.to_yaml()
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        
        logger.info(f"Saved configuration to {save_path}")
