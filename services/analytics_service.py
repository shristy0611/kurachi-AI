"""
Analytics and Insights Service for Kurachi AI
Provides comprehensive usage tracking, metrics collection, and knowledge gap analysis
"""
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import uuid
import re

from config import config
from models.database import db_manager, ChatMessage, Document
from services.sota_translation_orchestrator import sota_translation_orchestrator, Language
from utils.logger import get_logger

logger = get_logger("analytics_service")


@dataclass
class UsageMetrics:
    """Usage metrics data structure"""
    user_id: str
    session_id: str
    action_type: str  # query, upload, translation, etc.
    timestamp: datetime
    duration_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    

@dataclass
class QueryAnalytics:
    """Query analytics data structure"""
    query_text: str
    query_language: str
    response_time_ms: int
    source_count: int
    confidence_score: float
    timestamp: datetime
    user_id: str
    conversation_id: str
    bilingual_search_used: bool = False
    

@dataclass
class KnowledgeGap:
    """Knowledge gap identification"""
    query_text: str
    query_frequency: int
    avg_confidence: float
    low_confidence_responses: int
    suggested_documents: List[str]
    languages_involved: List[str]
    first_seen: datetime
    last_seen: datetime


class AnalyticsService:
    """Comprehensive analytics and insights service"""
    
    def __init__(self):
        self.usage_metrics: List[UsageMetrics] = []
        self.query_analytics: List[QueryAnalytics] = []
        self.knowledge_gaps: List[KnowledgeGap] = []
        self._load_persisted_data()
        logger.info("Analytics service initialized")
    
    def _load_persisted_data(self):
        """Load persisted analytics data (placeholder for now)"""
        # In a real implementation, this would load from a database or file
        pass
    
    def track_usage(self, user_id: str, action_type: str, 
                   duration_ms: Optional[int] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track user actions and usage patterns"""
        try:
            metric = UsageMetrics(
                user_id=user_id,
                session_id=str(uuid.uuid4()),  # Simple session tracking
                action_type=action_type,
                timestamp=datetime.utcnow(),
                duration_ms=duration_ms,
                metadata=metadata or {}
            )
            
            self.usage_metrics.append(metric)
            
            # Keep only recent metrics to prevent memory bloat
            if len(self.usage_metrics) > 10000:
                self.usage_metrics = self.usage_metrics[-5000:]
            
            logger.debug(f"Tracked usage: {action_type} for user {user_id}")
            
        except Exception as e:
            logger.error("Failed to track usage", error=e)
    
    def track_query(self, query_text: str, response_metadata: Dict[str, Any], 
                   response_time_ms: int, user_id: str, conversation_id: str) -> None:
        """Track query analytics for insight generation"""
        try:
            query_language = response_metadata.get("query_language", "unknown")
            source_count = response_metadata.get("source_count", 0)
            confidence = response_metadata.get("confidence", 0.5)
            bilingual_search = response_metadata.get("bilingual_search_performed", False)
            
            analytics = QueryAnalytics(
                query_text=query_text,
                query_language=query_language,
                response_time_ms=response_time_ms,
                source_count=source_count,
                confidence_score=confidence,
                timestamp=datetime.utcnow(),
                user_id=user_id,
                conversation_id=conversation_id,
                bilingual_search_used=bilingual_search
            )
            
            self.query_analytics.append(analytics)
            
            # Analyze for knowledge gaps
            self._analyze_knowledge_gaps(analytics)
            
            # Keep only recent analytics
            if len(self.query_analytics) > 5000:
                self.query_analytics = self.query_analytics[-2500:]
            
            logger.debug(f"Tracked query analytics for user {user_id}")
            
        except Exception as e:
            logger.error("Failed to track query analytics", error=e)
    
    def _analyze_knowledge_gaps(self, query_analytics: QueryAnalytics) -> None:
        """Analyze queries to identify knowledge gaps"""
        try:
            # Low confidence threshold
            LOW_CONFIDENCE_THRESHOLD = 0.6
            
            if query_analytics.confidence_score < LOW_CONFIDENCE_THRESHOLD:
                # Normalize query for gap analysis
                normalized_query = self._normalize_query(query_analytics.query_text)
                
                # Find existing gap or create new one
                existing_gap = None
                for gap in self.knowledge_gaps:
                    if self._queries_similar(normalized_query, gap.query_text):
                        existing_gap = gap
                        break
                
                if existing_gap:
                    # Update existing gap
                    existing_gap.query_frequency += 1
                    existing_gap.low_confidence_responses += 1
                    existing_gap.avg_confidence = (
                        (existing_gap.avg_confidence * (existing_gap.query_frequency - 1) + 
                         query_analytics.confidence_score) / existing_gap.query_frequency
                    )
                    existing_gap.last_seen = query_analytics.timestamp
                    
                    if query_analytics.query_language not in existing_gap.languages_involved:
                        existing_gap.languages_involved.append(query_analytics.query_language)
                else:
                    # Create new gap
                    new_gap = KnowledgeGap(
                        query_text=normalized_query,
                        query_frequency=1,
                        avg_confidence=query_analytics.confidence_score,
                        low_confidence_responses=1,
                        suggested_documents=[],
                        languages_involved=[query_analytics.query_language],
                        first_seen=query_analytics.timestamp,
                        last_seen=query_analytics.timestamp
                    )
                    self.knowledge_gaps.append(new_gap)
                
                # Keep only significant gaps
                if len(self.knowledge_gaps) > 1000:
                    # Sort by frequency and recency, keep top gaps
                    sorted_gaps = sorted(
                        self.knowledge_gaps,
                        key=lambda g: (g.query_frequency, g.last_seen),
                        reverse=True
                    )
                    self.knowledge_gaps = sorted_gaps[:500]
            
        except Exception as e:
            logger.error("Failed to analyze knowledge gaps", error=e)
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for gap analysis"""
        # Remove punctuation, lowercase, basic stemming
        normalized = re.sub(r'[^\w\s]', '', query.lower())
        return ' '.join(normalized.split())
    
    def _queries_similar(self, query1: str, query2: str) -> bool:
        """Check if two queries are similar enough to be considered the same gap"""
        # Simple similarity check - can be enhanced with more sophisticated NLP
        words1 = set(query1.split())
        words2 = set(query2.split())
        
        if len(words1) == 0 or len(words2) == 0:
            return False
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        # Jaccard similarity > 0.5
        return (intersection / union) > 0.5
    
    def get_usage_statistics(self, user_id: Optional[str] = None, 
                           days: int = 30) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Filter metrics
            metrics = [
                m for m in self.usage_metrics 
                if m.timestamp >= cutoff_date and (user_id is None or m.user_id == user_id)
            ]
            
            # Filter query analytics
            queries = [
                q for q in self.query_analytics
                if q.timestamp >= cutoff_date and (user_id is None or q.user_id == user_id)
            ]
            
            # Calculate statistics
            stats = {
                "period_days": days,
                "total_actions": len(metrics),
                "total_queries": len(queries),
                "unique_users": len(set(m.user_id for m in metrics)),
                "action_breakdown": self._get_action_breakdown(metrics),
                "query_statistics": self._get_query_statistics(queries),
                "language_usage": self._get_language_usage(queries),
                "performance_metrics": self._get_performance_metrics(queries),
                "bilingual_usage": self._get_bilingual_usage(queries),
                "knowledge_gaps_summary": self._get_knowledge_gaps_summary()
            }
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get usage statistics", error=e)
            return {}
    
    def _get_action_breakdown(self, metrics: List[UsageMetrics]) -> Dict[str, int]:
        """Get breakdown of action types"""
        return dict(Counter(m.action_type for m in metrics))
    
    def _get_query_statistics(self, queries: List[QueryAnalytics]) -> Dict[str, Any]:
        """Get query-related statistics"""
        if not queries:
            return {"avg_response_time": 0, "avg_confidence": 0, "avg_sources": 0}
        
        return {
            "avg_response_time": sum(q.response_time_ms for q in queries) / len(queries),
            "avg_confidence": sum(q.confidence_score for q in queries) / len(queries),
            "avg_sources": sum(q.source_count for q in queries) / len(queries),
            "high_confidence_queries": sum(1 for q in queries if q.confidence_score >= 0.8),
            "low_confidence_queries": sum(1 for q in queries if q.confidence_score < 0.6)
        }
    
    def _get_language_usage(self, queries: List[QueryAnalytics]) -> Dict[str, int]:
        """Get language usage breakdown"""
        return dict(Counter(q.query_language for q in queries))
    
    def _get_performance_metrics(self, queries: List[QueryAnalytics]) -> Dict[str, Any]:
        """Get performance-related metrics"""
        if not queries:
            return {}
        
        response_times = [q.response_time_ms for q in queries]
        confidence_scores = [q.confidence_score for q in queries]
        
        return {
            "response_time_p50": sorted(response_times)[len(response_times) // 2],
            "response_time_p95": sorted(response_times)[int(len(response_times) * 0.95)],
            "confidence_p50": sorted(confidence_scores)[len(confidence_scores) // 2],
            "confidence_p95": sorted(confidence_scores)[int(len(confidence_scores) * 0.95)]
        }
    
    def _get_bilingual_usage(self, queries: List[QueryAnalytics]) -> Dict[str, Any]:
        """Get bilingual search usage statistics"""
        bilingual_queries = [q for q in queries if q.bilingual_search_used]
        
        return {
            "total_bilingual_searches": len(bilingual_queries),
            "bilingual_usage_rate": len(bilingual_queries) / len(queries) if queries else 0,
            "avg_bilingual_confidence": (
                sum(q.confidence_score for q in bilingual_queries) / len(bilingual_queries)
                if bilingual_queries else 0
            )
        }
    
    def _get_knowledge_gaps_summary(self) -> Dict[str, Any]:
        """Get summary of identified knowledge gaps"""
        recent_gaps = [
            gap for gap in self.knowledge_gaps
            if gap.last_seen >= datetime.utcnow() - timedelta(days=7)
        ]
        
        # Sort by frequency and impact
        top_gaps = sorted(
            recent_gaps,
            key=lambda g: (g.query_frequency, -g.avg_confidence),
            reverse=True
        )[:10]
        
        return {
            "total_gaps_identified": len(self.knowledge_gaps),
            "recent_gaps": len(recent_gaps),
            "top_gaps": [
                {
                    "query": gap.query_text,
                    "frequency": gap.query_frequency,
                    "avg_confidence": gap.avg_confidence,
                    "languages": gap.languages_involved
                }
                for gap in top_gaps
            ]
        }
    
    def get_document_insights(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get insights about document usage and effectiveness"""
        try:
            # Get document statistics from database
            if user_id:
                documents = db_manager.get_user_documents(user_id)
            else:
                documents = []  # Would need get_all_documents method
            
            # Analyze document effectiveness based on query results
            doc_usage = defaultdict(int)
            doc_confidence = defaultdict(list)
            
            for query in self.query_analytics:
                if query.source_count > 0:
                    # This is a simplified analysis - in reality we'd track which specific documents were used
                    doc_usage["recent_queries"] += 1
                    doc_confidence["overall"].append(query.confidence_score)
            
            insights = {
                "total_documents": len(documents),
                "avg_document_effectiveness": (
                    sum(doc_confidence["overall"]) / len(doc_confidence["overall"])
                    if doc_confidence["overall"] else 0
                ),
                "document_language_distribution": self._get_document_language_distribution(documents),
                "upload_trends": self._get_upload_trends(documents),
                "processing_success_rate": self._get_processing_success_rate(documents)
            }
            
            return insights
            
        except Exception as e:
            logger.error("Failed to get document insights", error=e)
            return {}
    
    def _get_document_language_distribution(self, documents: List[Document]) -> Dict[str, int]:
        """Get distribution of document languages"""
        lang_dist = defaultdict(int)
        
        for doc in documents:
            if hasattr(doc, 'metadata') and doc.metadata:
                detected_lang = doc.metadata.get('detected_language', 'unknown')
                lang_dist[detected_lang] += 1
            else:
                lang_dist['unknown'] += 1
        
        return dict(lang_dist)
    
    def _get_upload_trends(self, documents: List[Document]) -> Dict[str, Any]:
        """Get document upload trends"""
        if not documents:
            return {"daily_uploads": {}, "monthly_uploads": {}}
        
        daily_uploads = defaultdict(int)
        monthly_uploads = defaultdict(int)
        
        for doc in documents:
            date_str = doc.created_at.strftime('%Y-%m-%d')
            month_str = doc.created_at.strftime('%Y-%m')
            
            daily_uploads[date_str] += 1
            monthly_uploads[month_str] += 1
        
        return {
            "daily_uploads": dict(daily_uploads),
            "monthly_uploads": dict(monthly_uploads),
            "total_uploads": len(documents)
        }
    
    def _get_processing_success_rate(self, documents: List[Document]) -> Dict[str, Any]:
        """Get document processing success rate"""
        if not documents:
            return {"success_rate": 0, "by_status": {}}
        
        status_counts = defaultdict(int)
        for doc in documents:
            status_counts[doc.processing_status] += 1
        
        completed = status_counts.get('completed', 0)
        total = len(documents)
        
        return {
            "success_rate": completed / total if total > 0 else 0,
            "by_status": dict(status_counts),
            "total_processed": total
        }
    
    def get_knowledge_gaps(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get identified knowledge gaps for improvement recommendations"""
        try:
            # Sort gaps by impact (frequency * low confidence rate)
            sorted_gaps = sorted(
                self.knowledge_gaps,
                key=lambda g: g.query_frequency * (1 - g.avg_confidence),
                reverse=True
            )
            
            return [
                {
                    "query": gap.query_text,
                    "frequency": gap.query_frequency,
                    "avg_confidence": gap.avg_confidence,
                    "low_confidence_responses": gap.low_confidence_responses,
                    "languages_involved": gap.languages_involved,
                    "impact_score": gap.query_frequency * (1 - gap.avg_confidence),
                    "first_seen": gap.first_seen.isoformat(),
                    "last_seen": gap.last_seen.isoformat(),
                    "suggestions": self._generate_gap_suggestions(gap)
                }
                for gap in sorted_gaps[:limit]
            ]
            
        except Exception as e:
            logger.error("Failed to get knowledge gaps", error=e)
            return []
    
    def _generate_gap_suggestions(self, gap: KnowledgeGap) -> List[str]:
        """Generate suggestions for addressing knowledge gaps"""
        suggestions = []
        
        # Common suggestions based on gap characteristics
        if gap.avg_confidence < 0.3:
            suggestions.append("Consider adding more comprehensive documentation on this topic")
        
        if len(gap.languages_involved) > 1:
            suggestions.append("Ensure bilingual content is available for this topic")
        
        if gap.query_frequency > 5:
            suggestions.append("This is a high-frequency question - consider creating dedicated content")
        
        if not suggestions:
            suggestions.append("Review and enhance existing documentation coverage")
        
        return suggestions
    
    def generate_insights_report(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a comprehensive insights report"""
        try:
            usage_stats = self.get_usage_statistics(user_id)
            doc_insights = self.get_document_insights(user_id)
            knowledge_gaps = self.get_knowledge_gaps()
            
            # Generate key insights
            key_insights = []
            
            # Usage insights
            if usage_stats.get("total_queries", 0) > 0:
                avg_confidence = usage_stats.get("query_statistics", {}).get("avg_confidence", 0)
                if avg_confidence < 0.7:
                    key_insights.append("Average query confidence is below optimal - consider expanding knowledge base")
                
                bilingual_rate = usage_stats.get("bilingual_usage", {}).get("bilingual_usage_rate", 0)
                if bilingual_rate > 0.2:
                    key_insights.append(f"High bilingual usage ({bilingual_rate:.1%}) - multilingual content is valuable")
            
            # Document insights
            if doc_insights.get("total_documents", 0) > 0:
                success_rate = doc_insights.get("processing_success_rate", {}).get("success_rate", 0)
                if success_rate < 0.9:
                    key_insights.append(f"Document processing success rate ({success_rate:.1%}) could be improved")
            
            # Knowledge gap insights
            if len(knowledge_gaps) > 0:
                high_impact_gaps = [gap for gap in knowledge_gaps if gap["impact_score"] > 2.0]
                if high_impact_gaps:
                    key_insights.append(f"Found {len(high_impact_gaps)} high-impact knowledge gaps requiring attention")
            
            return {
                "generated_at": datetime.utcnow().isoformat(),
                "period_analyzed": "Last 30 days",
                "user_scope": "single_user" if user_id else "all_users",
                "key_insights": key_insights,
                "usage_statistics": usage_stats,
                "document_insights": doc_insights,
                "knowledge_gaps": knowledge_gaps[:10],  # Top 10 gaps
                "recommendations": self._generate_recommendations(usage_stats, doc_insights, knowledge_gaps)
            }
            
        except Exception as e:
            logger.error("Failed to generate insights report", error=e)
            return {"error": "Failed to generate report", "generated_at": datetime.utcnow().isoformat()}
    
    def _generate_recommendations(self, usage_stats: Dict[str, Any], 
                                doc_insights: Dict[str, Any], 
                                knowledge_gaps: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations based on analytics"""
        recommendations = []
        
        # Usage-based recommendations
        if usage_stats.get("total_queries", 0) > 0:
            low_conf_rate = (
                usage_stats.get("query_statistics", {}).get("low_confidence_queries", 0) /
                usage_stats.get("total_queries", 1)
            )
            if low_conf_rate > 0.3:
                recommendations.append("High rate of low-confidence responses - expand knowledge base coverage")
        
        # Document-based recommendations
        if doc_insights.get("total_documents", 0) < 5:
            recommendations.append("Upload more documents to improve AI response quality and coverage")
        
        # Knowledge gap recommendations
        if knowledge_gaps:
            top_gap = knowledge_gaps[0]
            recommendations.append(f"Address top knowledge gap: '{top_gap['query'][:50]}...'")
        
        # Bilingual recommendations
        lang_usage = usage_stats.get("language_usage", {})
        if len(lang_usage) > 1:
            recommendations.append("Consider adding more bilingual content to support multiple languages")
        
        if not recommendations:
            recommendations.append("System is performing well - continue monitoring for optimization opportunities")
        
        return recommendations


# Global analytics service instance
analytics_service = AnalyticsService()