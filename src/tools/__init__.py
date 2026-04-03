"""Lazy exports for tool modules."""

from __future__ import annotations


def __getattr__(name: str):
    if name in {
        "find_device_metadata",
        "find_device_metadata_with_engine",
        "configure_metadata_engine",
        "get_metadata_engine",
    }:
        from src.tools import device_tool as device_tool_module

        return getattr(device_tool_module, name)
    if name in {
        "fetch_sensor_data",
        "fetch_sensor_data_with_components",
        "configure_sensor_tool",
        "get_data_fetcher",
        "get_cache_manager",
        "get_context_compressor",
    }:
        from src.tools import sensor_tool as sensor_tool_module

        return getattr(sensor_tool_module, name)
    raise AttributeError(name)


__all__ = [
    "find_device_metadata",
    "find_device_metadata_with_engine",
    "configure_metadata_engine",
    "get_metadata_engine",
    "fetch_sensor_data",
    "fetch_sensor_data_with_components",
    "configure_sensor_tool",
    "get_data_fetcher",
    "get_cache_manager",
    "get_context_compressor",
]
