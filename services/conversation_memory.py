"""
Conversation Memory Service for Kurachi AI
Handles conversation memory, context management, and intelligent summarization
"""
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import uuid

from langchain_community.llms import Ollama
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from config import config
from models.database import ChatConversation, ChatMessage, db_manager
from utils.logger import get_logger

logger = get_logger("conversation_memory")


@dataclass
class ConversationContext:
    """Enhanced conversation context with memory management"""
    conversation_id: str
    user_id: str
    title: str
    total_messages: int
    context_window_size: int
    summary: Optional[str] = None
    key_topics: List[str] = None
    last_activity: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.key_topics is None:
            self.key_topics = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ContextWindow:
    """Represents a context window for conversation memory"""
    messages: List[ChatMessage]
    summary: Optional[str] = None
    token_count: int = 0
    importance_score: float = 0.0
    created_at: Optional[datetime] = None


class ConversationMemoryService:
    """Service for managing conversation memory and context"""
    
    def __init__(self):
        self.llm = Ollama(
            model=config.ai.llm_model,
            base_url=config.ai.ollama_base_url,
            temperature=0.3  # Lower temperature for more consistent summaries
        )
        
        # Configuration for memory management
        self.max_context_tokens = 4000  # Maximum tokens in context window
        self.max_recent_messages = 20   # Maximum recent messages to keep in full
        self.summary_trigger_threshold = 30  # Trigger summary after N messages
        self.context_overlap_messages = 5   # Messages to overlap between windows
        
    def get_conversation_context(self, conversation_id: str, user_id: Optional[str] = None) -> Optional[ConversationContext]:
        """Get enhanced conversation context with memory management"""
        try:
            # Get conversation metadata
            if user_id:
                conversations = db_manager.get_user_conversations(user_id)
                target_conversation = None
                for conv in conversations:
                    if conv.id == conversation_id:
                        target_conversation = conv
                        break
            else:
                target_conversation = db_manager.get_conversation_by_id(conversation_id)
            
            if not target_conversation:
                logger.warning(f"Conversation {conversation_id} not found for user {user_id}")
                return None
            
            # Get all messages for the conversation
            messages = db_manager.get_conversation_messages(conversation_id)
            
            # Extract key topics from recent messages
            key_topics = self._extract_key_topics(messages[-10:]) if messages else []
            
            # Get or create summary if needed
            summary = None
            if len(messages) > self.summary_trigger_threshold:
                summary = self._get_or_create_summary(conversation_id, messages)
            
            return ConversationContext(
                conversation_id=conversation_id,
                user_id=target_conversation.user_id,
                title=target_conversation.title,
                total_messages=len(messages),
                context_window_size=min(len(messages), self.max_recent_messages),
                summary=summary,
                key_topics=key_topics,
                last_activity=messages[-1].created_at if messages else target_conversation.updated_at,
                metadata=target_conversation.metadata or {}
            )
            
        except Exception as e:
            logger.error("Failed to get conversation context", error=e, extra={
                "conversation_id": conversation_id,
                "user_id": user_id
            })
            return None
    
    def build_context_window(self, conversation_id: str, current_query: str, user_id: Optional[str] = None) -> str:
        """Build optimized context window for the current query"""
        try:
            # Get conversation context
            context = self.get_conversation_context(conversation_id, user_id)
            if not context:
                return current_query
            
            # Get all messages
            all_messages = db_manager.get_conversation_messages(conversation_id)
            
            # Build context parts
            context_parts = []
            
            # Add system prompt
            context_parts.append("You are Kurachi AI, a professional business document assistant.")
            context_parts.append("Maintain conversation context and provide coherent, helpful responses.")
            context_parts.append("")
            
            # Add conversation summary if available
            if context.summary:
                context_parts.append("Conversation Summary:")
                context_parts.append(context.summary)
                context_parts.append("")
            
            # Add key topics for context
            if context.key_topics:
                context_parts.append("Key Topics Discussed:")
                context_parts.append(", ".join(context.key_topics))
                context_parts.append("")
            
            # Add recent messages with intelligent selection
            recent_messages = self._select_relevant_messages(all_messages, current_query)
            
            if recent_messages:
                context_parts.append("Recent Conversation:")
                for msg in recent_messages:
                    role = "Human" if msg.role == "user" else "Assistant"
                    # Truncate very long messages but preserve important content
                    content = self._truncate_message_intelligently(msg.content, 300)
                    context_parts.append(f"{role}: {content}")
                context_parts.append("")
            
            # Add current query
            context_parts.append(f"Current Question: {current_query}")
            context_parts.append("")
            context_parts.append("Please provide a comprehensive response that considers the conversation history:")
            
            full_context = "\n".join(context_parts)
            
            # Ensure context doesn't exceed token limits
            return self._truncate_context_to_limit(full_context)
            
        except Exception as e:
            logger.error("Failed to build context window", error=e, extra={
                "conversation_id": conversation_id,
                "user_id": user_id
            })
            return current_query
    
    def update_conversation_memory(self, conversation_id: str, new_message: ChatMessage) -> bool:
        """Update conversation memory after adding a new message"""
        try:
            # Get current context
            context = self.get_conversation_context(conversation_id)
            if not context:
                return False
            
            # Check if we need to create or update summary
            if context.total_messages > self.summary_trigger_threshold:
                # Get all messages
                all_messages = db_manager.get_conversation_messages(conversation_id)
                
                # Update summary if needed
                self._update_conversation_summary(conversation_id, all_messages)
            
            # Update key topics
            if new_message.role == "user":
                new_topics = self._extract_key_topics([new_message])
                if new_topics:
                    # Update conversation metadata with new topics
                    self._update_conversation_topics(conversation_id, new_topics)
            
            return True
            
        except Exception as e:
            logger.error("Failed to update conversation memory", error=e, extra={
                "conversation_id": conversation_id
            })
            return False
    
    def _get_or_create_summary(self, conversation_id: str, messages: List[ChatMessage]) -> Optional[str]:
        """Get existing summary or create a new one"""
        try:
            # Check if summary exists in conversation metadata
            target_conversation = db_manager.get_conversation_by_id(conversation_id)
            
            if target_conversation and target_conversation.metadata:
                existing_summary = target_conversation.metadata.get("summary")
                summary_version = target_conversation.metadata.get("summary_version", 0)
                
                # Check if summary is up to date (covers most messages)
                if existing_summary and summary_version >= len(messages) - 10:
                    return existing_summary
            
            # Create new summary
            return self._create_conversation_summary(conversation_id, messages)
            
        except Exception as e:
            logger.error("Failed to get or create summary", error=e)
            return None
    
    def _create_conversation_summary(self, conversation_id: str, messages: List[ChatMessage]) -> Optional[str]:
        """Create a comprehensive conversation summary"""
        try:
            if len(messages) < 5:  # Don't summarize very short conversations
                return None
            
            # Select messages to summarize (exclude very recent ones)
            messages_to_summarize = messages[:-5] if len(messages) > 10 else messages[:-2]
            
            # Build summary prompt
            conversation_text = []
            for msg in messages_to_summarize:
                role = "Human" if msg.role == "user" else "Assistant"
                conversation_text.append(f"{role}: {msg.content}")
            
            summary_prompt = f"""
Please create a concise but comprehensive summary of this conversation. Focus on:
1. Main topics discussed
2. Key questions asked by the user
3. Important information provided
4. Any ongoing themes or patterns
5. Context that would be helpful for future responses

Conversation to summarize:
{chr(10).join(conversation_text)}

Summary:"""
            
            # Generate summary using LLM
            summary = self.llm.invoke(summary_prompt)
            
            # Save summary to conversation metadata
            self._save_conversation_summary(conversation_id, summary, len(messages))
            
            logger.info(f"Created conversation summary for {conversation_id}")
            return summary
            
        except Exception as e:
            logger.error("Failed to create conversation summary", error=e)
            return None
    
    def _update_conversation_summary(self, conversation_id: str, messages: List[ChatMessage]) -> bool:
        """Update existing conversation summary with new messages"""
        try:
            # Get existing summary
            existing_summary = self._get_or_create_summary(conversation_id, messages)
            
            if not existing_summary:
                return self._create_conversation_summary(conversation_id, messages) is not None
            
            # Get new messages since last summary
            target_conversation = db_manager.get_conversation_by_id(conversation_id)
            
            if not target_conversation or not target_conversation.metadata:
                return False
            
            last_summary_version = target_conversation.metadata.get("summary_version", 0)
            new_messages = messages[last_summary_version:]
            
            if len(new_messages) < 5:  # Not enough new messages to update summary
                return True
            
            # Create incremental summary
            new_conversation_text = []
            for msg in new_messages:
                role = "Human" if msg.role == "user" else "Assistant"
                new_conversation_text.append(f"{role}: {msg.content}")
            
            update_prompt = f"""
Current conversation summary:
{existing_summary}

New conversation content to integrate:
{chr(10).join(new_conversation_text)}

Please update the summary to include the new information while keeping it concise and comprehensive:"""
            
            updated_summary = self.llm.invoke(update_prompt)
            
            # Save updated summary
            self._save_conversation_summary(conversation_id, updated_summary, len(messages))
            
            logger.info(f"Updated conversation summary for {conversation_id}")
            return True
            
        except Exception as e:
            logger.error("Failed to update conversation summary", error=e)
            return False
    
    def _save_conversation_summary(self, conversation_id: str, summary: str, message_count: int) -> bool:
        """Save conversation summary to database"""
        try:
            # Get existing metadata
            conversation = db_manager.get_conversation_by_id(conversation_id)
            if not conversation:
                return False
            
            # Update metadata with summary
            existing_metadata = conversation.metadata or {}
            existing_metadata.update({
                "summary": summary,
                "summary_version": message_count,
                "summary_created_at": datetime.utcnow().isoformat()
            })
            
            # Save updated metadata
            success = db_manager.update_conversation_metadata(conversation_id, existing_metadata)
            
            if success:
                logger.info(f"Saved summary for conversation {conversation_id}: {summary[:100]}...")
            
            return success
            
        except Exception as e:
            logger.error("Failed to save conversation summary", error=e)
            return False
    
    def _extract_key_topics(self, messages: List[ChatMessage]) -> List[str]:
        """Extract key topics from messages using simple keyword analysis"""
        try:
            if not messages:
                return []
            
            # Combine all user messages for topic extraction
            user_messages = [msg.content for msg in messages if msg.role == "user"]
            
            if not user_messages:
                return []
            
            combined_text = " ".join(user_messages)
            
            # Simple topic extraction prompt
            topic_prompt = f"""
Extract 3-5 key topics or themes from this text. Return only the topics as a comma-separated list:

Text: {combined_text[:1000]}

Key topics:"""
            
            topics_response = self.llm.invoke(topic_prompt)
            
            # Parse topics from response
            topics = [topic.strip() for topic in topics_response.split(",")]
            topics = [topic for topic in topics if topic and len(topic) > 2]
            
            return topics[:5]  # Limit to 5 topics
            
        except Exception as e:
            logger.error("Failed to extract key topics", error=e)
            return []
    
    def _select_relevant_messages(self, messages: List[ChatMessage], current_query: str) -> List[ChatMessage]:
        """Select most relevant messages for context window"""
        try:
            if len(messages) <= self.max_recent_messages:
                return messages
            
            # Always include the most recent messages
            recent_messages = messages[-self.max_recent_messages//2:]
            
            # Select additional relevant messages from earlier in conversation
            earlier_messages = messages[:-self.max_recent_messages//2]
            
            if not earlier_messages:
                return recent_messages
            
            # Simple relevance scoring based on keyword overlap
            query_words = set(current_query.lower().split())
            scored_messages = []
            
            for msg in earlier_messages:
                msg_words = set(msg.content.lower().split())
                overlap_score = len(query_words.intersection(msg_words))
                scored_messages.append((msg, overlap_score))
            
            # Sort by relevance and take top messages
            scored_messages.sort(key=lambda x: x[1], reverse=True)
            relevant_earlier = [msg for msg, score in scored_messages[:self.max_recent_messages//2]]
            
            # Combine and sort by timestamp
            all_selected = relevant_earlier + recent_messages
            all_selected.sort(key=lambda x: x.created_at)
            
            return all_selected
            
        except Exception as e:
            logger.error("Failed to select relevant messages", error=e)
            return messages[-self.max_recent_messages:] if messages else []
    
    def _truncate_message_intelligently(self, content: str, max_length: int) -> str:
        """Intelligently truncate message content while preserving meaning"""
        if len(content) <= max_length:
            return content
        
        # Try to truncate at sentence boundaries
        sentences = content.split('. ')
        truncated = ""
        
        for sentence in sentences:
            if len(truncated + sentence + '. ') <= max_length - 3:
                truncated += sentence + '. '
            else:
                break
        
        if truncated:
            return truncated.rstrip() + "..."
        else:
            # Fallback to simple truncation
            return content[:max_length-3] + "..."
    
    def _truncate_context_to_limit(self, context: str) -> str:
        """Truncate context to stay within token limits"""
        # Rough estimation: 1 token ≈ 4 characters
        max_chars = self.max_context_tokens * 4
        
        if len(context) <= max_chars:
            return context
        
        # Try to truncate at paragraph boundaries
        paragraphs = context.split('\n\n')
        truncated = ""
        
        for paragraph in paragraphs:
            if len(truncated + paragraph + '\n\n') <= max_chars - 100:  # Leave some buffer
                truncated += paragraph + '\n\n'
            else:
                break
        
        if truncated:
            return truncated.rstrip() + "\n\n[Context truncated to fit limits]"
        else:
            # Fallback to simple truncation
            return context[:max_chars-100] + "\n\n[Context truncated to fit limits]"
    
    def _update_conversation_topics(self, conversation_id: str, new_topics: List[str]) -> bool:
        """Update conversation topics in metadata"""
        try:
            # Get existing metadata
            conversation = db_manager.get_conversation_by_id(conversation_id)
            if not conversation:
                return False
            
            # Update metadata with topics
            existing_metadata = conversation.metadata or {}
            existing_topics = existing_metadata.get("key_topics", [])
            
            # Merge new topics with existing ones (avoid duplicates)
            all_topics = list(set(existing_topics + new_topics))
            existing_metadata["key_topics"] = all_topics[:10]  # Limit to 10 topics
            existing_metadata["topics_updated_at"] = datetime.utcnow().isoformat()
            
            # Save updated metadata
            success = db_manager.update_conversation_metadata(conversation_id, existing_metadata)
            
            if success:
                logger.info(f"Updated topics for conversation {conversation_id}: {all_topics}")
            
            return success
            
        except Exception as e:
            logger.error("Failed to update conversation topics", error=e)
            return False
    
    def get_conversation_statistics(self, conversation_id: str) -> Dict[str, Any]:
        """Get detailed statistics about a conversation"""
        try:
            messages = db_manager.get_conversation_messages(conversation_id)
            
            if not messages:
                return {"total_messages": 0}
            
            user_messages = [msg for msg in messages if msg.role == "user"]
            assistant_messages = [msg for msg in messages if msg.role == "assistant"]
            
            # Calculate conversation duration
            start_time = messages[0].created_at
            end_time = messages[-1].created_at
            duration = end_time - start_time
            
            # Calculate average response time (simplified)
            response_times = []
            for i in range(len(messages) - 1):
                if messages[i].role == "user" and messages[i+1].role == "assistant":
                    response_time = messages[i+1].created_at - messages[i].created_at
                    response_times.append(response_time.total_seconds())
            
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            return {
                "total_messages": len(messages),
                "user_messages": len(user_messages),
                "assistant_messages": len(assistant_messages),
                "conversation_duration_minutes": duration.total_seconds() / 60,
                "average_response_time_seconds": avg_response_time,
                "first_message_at": start_time.isoformat(),
                "last_message_at": end_time.isoformat(),
                "messages_per_hour": len(messages) / max(duration.total_seconds() / 3600, 1)
            }
            
        except Exception as e:
            logger.error("Failed to get conversation statistics", error=e)
            return {"error": str(e)}


# Global conversation memory service instance
conversation_memory_service = ConversationMemoryService()