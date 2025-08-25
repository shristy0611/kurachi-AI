#!/usr/bin/env python3
"""
Production Telemetry Logger for Intelligent Response Formatter
Tracks query patterns, confusion pairs, and performance metrics
"""
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from utils.logger import get_logger

logger = get_logger("formatter_telemetry")


@dataclass
class FormatterTelemetry:
    """Telemetry data for formatter operations"""
    query: str
    detected_type: str
    final_type: str
    confidence: float
    fallback_used: bool
    processing_time_ms: int
    rule_matched: Optional[str]
    smart_mapping_applied: bool
    timestamp: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None


class FormatterTelemetryLogger:
    """Production telemetry logger for monitoring formatter performance"""
    
    def __init__(self):
        self.confusion_pairs = {}
        self.performance_stats = {
            "rule_matched_queries": [],
            "llm_fallback_queries": [],
            "smart_mappings": []
        }
    
    def log_formatter_operation(
        self,
        query: str,
        detected_type: str,
        final_type: str,
        confidence: float,
        processing_time_ms: int,
        fallback_used: bool = False,
        rule_matched: Optional[str] = None,
        smart_mapping_applied: bool = False,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None
    ):
        """Log formatter operation with full telemetry"""
        
        telemetry = FormatterTelemetry(
            query=query[:100],  # Truncate for privacy
            detected_type=detected_type,
            final_type=final_type,
            confidence=confidence,
            fallback_used=fallback_used,
            processing_time_ms=processing_time_ms,
            rule_matched=rule_matched,
            smart_mapping_applied=smart_mapping_applied,
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        # Log structured telemetry
        logger.info("formatter_operation", extra=asdict(telemetry))
        
        # Track confusion pairs
        if detected_type != final_type:
            pair = f"{detected_type}→{final_type}"
            self.confusion_pairs[pair] = self.confusion_pairs.get(pair, 0) + 1
        
        # Track performance stats
        if rule_matched:
            self.performance_stats["rule_matched_queries"].append(processing_time_ms)
        elif fallback_used:
            self.performance_stats["llm_fallback_queries"].append(processing_time_ms)
        
        if smart_mapping_applied:
            self.performance_stats["smart_mappings"].append({
                "detected": detected_type,
                "final": final_type,
                "query_pattern": query[:50]
            })
    
    def get_confusion_report(self) -> Dict[str, Any]:
        """Generate confusion pairs report for model improvement"""
        return {
            "confusion_pairs": self.confusion_pairs,
            "total_mappings": sum(self.confusion_pairs.values()),
            "most_common_confusions": sorted(
                self.confusion_pairs.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        rule_times = self.performance_stats["rule_matched_queries"]
        llm_times = self.performance_stats["llm_fallback_queries"]
        
        return {
            "rule_matched_avg_ms": sum(rule_times) / len(rule_times) if rule_times else 0,
            "llm_fallback_avg_ms": sum(llm_times) / len(llm_times) if llm_times else 0,
            "rule_efficiency": len(rule_times) / (len(rule_times) + len(llm_times)) if (rule_times or llm_times) else 0,
            "smart_mappings_count": len(self.performance_stats["smart_mappings"]),
            "smart_mappings": self.performance_stats["smart_mappings"][-10:]  # Last 10
        }


# Global telemetry logger instance
formatter_telemetry = FormatterTelemetryLogger()