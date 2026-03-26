"""Context Compressor for LLM-friendly data formatting.

This module provides functionality to compress and format data for efficient
consumption by Large Language Models within their context window limits.
"""

import csv
import io
from enum import Enum
from typing import Any, Dict, List, Optional


class OutputFormat(Enum):
    """Supported output formats for data compression."""
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"
    MINIMAL = "minimal"  # Only keeps [time, val]


class ContextCompressor:
    """上下文压缩器 - Compresses data to fit LLM context windows.
    
    This class provides methods to compress query results into various formats
    suitable for LLM consumption, removing redundant fields and converting
    to token-efficient representations.
    """
    
    def __init__(self, max_tokens: int = 4000):
        """Initialize the compressor with token limit.
        
        Args:
            max_tokens: Maximum estimated tokens for output (default: 4000)
        """
        self.max_tokens = max_tokens
    
    def compress(
        self,
        data: List[Dict[str, Any]],
        output_format: OutputFormat = OutputFormat.MINIMAL,
        fields_to_keep: Optional[List[str]] = None
    ) -> str:
        """Compress data to fit LLM context window.
        
        Args:
            data: Original data list of dictionaries
            output_format: Desired output format
            fields_to_keep: Optional list of fields to retain (overrides format defaults)
        
        Returns:
            Compressed string representation of the data
        """
        if not data:
            return ""
        
        # Filter fields based on format or explicit field list
        filtered_data = self._filter_fields(data, output_format, fields_to_keep)
        
        # Convert to requested format
        if output_format == OutputFormat.MINIMAL:
            result = self._to_minimal(filtered_data)
        elif output_format == OutputFormat.CSV:
            result = self._to_csv(filtered_data)
        elif output_format == OutputFormat.MARKDOWN:
            result = self._to_markdown(filtered_data)
        else:  # JSON
            result = self._to_json(filtered_data)
        
        # Check if result exceeds token estimate and further compress if needed
        estimated_tokens = self._estimate_tokens(result)
        if estimated_tokens > self.max_tokens and len(data) > 0:
            return self.to_statistics_summary(data)
        
        return result
    
    def _filter_fields(
        self,
        data: List[Dict[str, Any]],
        output_format: OutputFormat,
        fields_to_keep: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Filter data to keep only relevant fields.
        
        Args:
            data: Original data list
            output_format: Output format (determines default fields)
            fields_to_keep: Explicit fields to keep (overrides defaults)
        
        Returns:
            Filtered data list
        """
        if fields_to_keep:
            keep_fields = set(fields_to_keep)
        elif output_format == OutputFormat.MINIMAL:
            # 支持多种时间字段名: time, logTime
            keep_fields = {"time", "logTime", "val"}
        else:
            # For other formats, keep all fields
            return data
        
        return [
            {k: v for k, v in record.items() if k in keep_fields}
            for record in data
        ]
    
    def _to_minimal(self, data: List[Dict[str, Any]]) -> str:
        """Convert to minimal [time, val] array format.
        
        Args:
            data: Filtered data list
        
        Returns:
            String representation of minimal data
        """
        minimal_array = []
        for record in data:
            # 支持多种时间字段名: time, logTime
            time_val = record.get("time") or record.get("logTime") or ""
            val = record.get("val", "")
            minimal_array.append([str(time_val), val])
        
        return str(minimal_array)
    
    def _to_csv(self, data: List[Dict[str, Any]]) -> str:
        """Convert to CSV format.
        
        Args:
            data: Data list to convert
        
        Returns:
            Valid CSV string
        """
        if not data:
            return ""
        
        output = io.StringIO()
        
        # Get all unique fields from data
        all_fields = set()
        for record in data:
            all_fields.update(record.keys())
        fieldnames = sorted(all_fields)
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
        
        return output.getvalue()
    
    def _to_markdown(self, data: List[Dict[str, Any]]) -> str:
        """Convert to Markdown table format.
        
        Args:
            data: Data list to convert
        
        Returns:
            Markdown table string
        """
        if not data:
            return ""
        
        # Get all unique fields from data
        all_fields = set()
        for record in data:
            all_fields.update(record.keys())
        headers = sorted(all_fields)
        
        # Build header row
        header_row = "| " + " | ".join(headers) + " |"
        
        # Build separator row
        separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        
        # Build data rows
        data_rows = []
        for record in data:
            row_values = [str(record.get(h, "")) for h in headers]
            data_rows.append("| " + " | ".join(row_values) + " |")
        
        return "\n".join([header_row, separator_row] + data_rows)
    
    def _to_json(self, data: List[Dict[str, Any]]) -> str:
        """Convert to JSON string format.
        
        Args:
            data: Data list to convert
        
        Returns:
            JSON string representation
        """
        import json
        return json.dumps(data, default=str, ensure_ascii=False)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.
        
        Uses a simple heuristic: ~4 characters per token on average.
        
        Args:
            text: Text to estimate tokens for
        
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    def to_statistics_summary(self, data: List[Dict[str, Any]]) -> str:
        """Convert data to statistics summary when it's too large.
        
        Computes min, max, avg, count statistics for numeric 'val' field.
        
        Args:
            data: Original data list
        
        Returns:
            Statistics summary string
        """
        if not data:
            return "No data available"
        
        # Extract numeric values
        values = []
        for record in data:
            val = record.get("val")
            if val is not None:
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    continue
        
        if not values:
            return f"Data count: {len(data)}, No numeric values found"
        
        # Compute statistics
        min_val = min(values)
        max_val = max(values)
        avg_val = sum(values) / len(values)
        count = len(values)
        
        # Get time range if available (支持 time 和 logTime)
        times = []
        for record in data:
            t = record.get("time") or record.get("logTime")
            if t:
                times.append(str(t))
        
        time_range = ""
        if times:
            time_range = f", Time range: {min(times)} to {max(times)}"
        
        return (
            f"Statistics Summary:\n"
            f"- Count: {count}\n"
            f"- Min: {min_val}\n"
            f"- Max: {max_val}\n"
            f"- Avg: {avg_val:.2f}{time_range}"
        )
