"""
Chat service for Kurachi AI
Handles conversational AI with memory and context management
"""
from typing import Optional, List, Dict, Any, Generator
from datetime import datetime
import uuid
import time
import re

from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

from config import config
from models.database import ChatConversation, ChatMessage, db_manager
from services.document_service import document_service
from services.sota_translation_orchestrator import sota_translation_orchestrator, TranslationQuality
from services.language_detection import language_detection_service
from services.analytics_service import analytics_service
from services.conversation_memory import conversation_memory_service
from services.source_attribution import source_attribution_service
from services.intelligent_response_formatter import intelligent_response_formatter
from services.multilingual_conversation_interface import multilingual_interface
from utils.logger import get_logger, audit_logger

logger = get_logger("chat_service")


class ChatService:
    """Service for handling chat operations with RAG capabilities"""
    
    def __init__(self):
        # Configure Ollama for instant responses with small models
        self.llm = Ollama(
            model=config.ai.llm_model,
            base_url=config.ai.ollama_base_url,
            temperature=config.ai.temperature,
            # Optimizations for instant responses
            timeout=5,  # Short timeout
            num_ctx=2048,  # Smaller context for speed
            num_predict=512,  # Limit output length
        )
        # Configure embeddings for instant processing
        self.embedding_model = OllamaEmbeddings(
            model=config.ai.embedding_model,
            base_url=config.ai.ollama_base_url,
        )
        self.qa_chain = None
        self._init_qa_chain()
    
    def _init_qa_chain(self):
        """Initialize the QA chain with modern LCEL patterns where possible"""
        try:
            if document_service.vector_store:
                # Check if vector store has any documents
                try:
                    # Test if we can query the vector store
                    test_result = document_service.vector_store.similarity_search("test", k=1)
                    
                    retriever = document_service.vector_store.as_retriever(
                        search_kwargs={"k": 5}  # Retrieve top 5 relevant chunks
                    )
                    
                    # Try to use modern LCEL patterns if available
                    try:
                        from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
                        from langchain.prompts import ChatPromptTemplate
                        from langchain.schema.output_parser import StrOutputParser
                        from langchain_community.chat_models import ChatOllama
                        
                        # Modern LCEL implementation
                        template = """Answer based on context: {context}
                        Question: {question}
                        Answer:"""
                        
                        prompt = ChatPromptTemplate.from_template(template)
                        
                        # Create modern LCEL chain
                        def format_docs(docs):
                            return "\n\n".join(doc.page_content for doc in docs)
                        
                        self.qa_chain = (
                            RunnableParallel({
                                "context": retriever | format_docs,
                                "question": RunnablePassthrough()
                            })
                            | prompt
                            | ChatOllama(model=config.ai.llm_model, base_url=config.ai.ollama_base_url)
                            | StrOutputParser()
                        )
                        
                        # Add method to track that we're using modern patterns
                        self.qa_chain._is_modern_lcel = True
                        logger.info("QA chain initialized with modern LCEL patterns")
                        
                    except ImportError:
                        # Fallback to legacy pattern if LCEL not available
                        self.qa_chain = RetrievalQA.from_chain_type(
                            llm=self.llm,
                            chain_type="stuff",
                            retriever=retriever,
                            return_source_documents=True
                        )
                        self.qa_chain._is_modern_lcel = False
                        logger.info("QA chain initialized with legacy patterns (LCEL unavailable)")
                        
                except Exception as e:
                    logger.warning(f"Vector store exists but is empty or invalid: {e}")
                    self.qa_chain = None
            else:
                logger.info("QA chain will be initialized when documents are available")
                self.qa_chain = None
        except Exception as e:
            logger.error("Failed to initialize QA chain", error=e)
            self.qa_chain = None
    
    def _ensure_qa_chain(self):
        """Ensure QA chain is available, reinitialize if needed"""
        if self.qa_chain is None:
            self._init_qa_chain()
        return self.qa_chain is not None
    
    def _has_documents(self) -> bool:
        """Check if there are any documents available in the vector store"""
        try:
            if not document_service.vector_store:
                return False
            # Try a simple query to check if documents exist
            test_results = document_service.vector_store.similarity_search("test", k=1)
            return len(test_results) > 0
        except Exception:
            return False
    
    def create_conversation(self, user_id: str, title: Optional[str] = None) -> Optional[ChatConversation]:
        """Create a new chat conversation"""
        try:
            if not title:
                title = f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
            
            conversation = db_manager.create_conversation(user_id, title)
            if conversation:
                audit_logger.log_user_action(
                    user_id,
                    "create_conversation",
                    conversation.id,
                    success=True,
                    details={"title": title}
                )
                logger.info(f"Created conversation for user {user_id}", extra={
                    "conversation_id": conversation.id,
                    "title": title
                })
            
            return conversation
            
        except Exception as e:
            logger.error("Failed to create conversation", error=e, extra={"user_id": user_id})
            return None
    
    def get_conversation(self, conversation_id: str, user_id: Optional[str] = None) -> Optional[ChatConversation]:
        """Get conversation with permission check"""
        try:
            # Get conversation from database
            conversations = db_manager.get_user_conversations(user_id) if user_id else []
            
            for conv in conversations:
                if conv.id == conversation_id:
                    return conv
            
            return None
            
        except Exception as e:
            logger.error("Failed to get conversation", error=e, extra={
                "conversation_id": conversation_id,
                "user_id": user_id
            })
            return None
    
    def get_conversation_history(self, conversation_id: str, user_id: Optional[str] = None) -> List[ChatMessage]:
        """Get conversation message history"""
        try:
            # Verify user has access to conversation
            if user_id:
                conversation = self.get_conversation(conversation_id, user_id)
                if not conversation:
                    logger.warning(f"User {user_id} attempted to access conversation {conversation_id}")
                    return []
            
            messages = db_manager.get_conversation_messages(conversation_id)
            return messages
            
        except Exception as e:
            logger.error("Failed to get conversation history", error=e, extra={
                "conversation_id": conversation_id,
                "user_id": user_id
            })
            return []
    
    def _build_context_prompt(self, messages: List[ChatMessage], current_query: str, conversation_id: Optional[str] = None) -> str:
        """Build context-aware prompt from conversation history using enhanced memory management"""
        try:
            # Use conversation memory service for enhanced context building
            if conversation_id:
                return conversation_memory_service.build_context_window(conversation_id, current_query)
            
            # Fallback to simple context building if no conversation_id
            max_context_messages = 10
            recent_messages = messages[-max_context_messages:] if len(messages) > max_context_messages else messages
            
            context_parts = []
            
            # Add conversation history
            if recent_messages:
                context_parts.append("Previous conversation:")
                for msg in recent_messages:
                    role = "Human" if msg.role == "user" else "Assistant"
                    context_parts.append(f"{role}: {msg.content}")
                context_parts.append("")
            
            # Add current query
            context_parts.append(f"Current question: {current_query}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error("Failed to build context prompt", error=e)
            return current_query
    
    def send_message(self, conversation_id: str, message: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Send a message and get AI response with multilingual support"""
        try:
            # Verify conversation access
            if user_id:
                conversation = self.get_conversation(conversation_id, user_id)
                if not conversation:
                    logger.warning(f"User {user_id} attempted to send message to conversation {conversation_id}")
                    return None
            
            # Process multilingual query
            multilingual_context = None
            if user_id:
                multilingual_context = multilingual_interface.process_multilingual_query(
                    message, user_id, conversation_id
                )
            
            # Save user message
            user_message = db_manager.add_message(conversation_id, "user", message)
            if not user_message:
                logger.error("Failed to save user message")
                return None
            
            # Get conversation history for context
            history = self.get_conversation_history(conversation_id, user_id)
            
            # Generate AI response with enhanced context and multilingual support
            response_data = self._generate_response(message, history[:-1], conversation_id, multilingual_context)  # Exclude the just-added message
            
            if not response_data:
                logger.error("Failed to generate AI response")
                return None
            
            # Track query analytics
            if response_data.get("metadata"):
                response_time_ms = response_data["metadata"].get("processing_time_ms", 0)
                analytics_service.track_query(
                    query_text=message,
                    response_metadata=response_data["metadata"],
                    response_time_ms=response_time_ms,
                    user_id=user_id or "anonymous",
                    conversation_id=conversation_id
                )
                
                # Track usage
                analytics_service.track_usage(
                    user_id=user_id or "anonymous",
                    action_type="query",
                    duration_ms=response_time_ms,
                    metadata={
                        "query_length": len(message),
                        "response_type": response_data["metadata"].get("response_type"),
                        "source_count": response_data["metadata"].get("source_count", 0),
                        "language": response_data["metadata"].get("query_language")
                    }
                )
            
            # Save AI response
            ai_message = db_manager.add_message(
                conversation_id,
                "assistant",
                response_data["content"],
                metadata=response_data.get("metadata")
            )
            
            if not ai_message:
                logger.error("Failed to save AI response")
                return None
            
            # Update conversation memory with new message
            conversation_memory_service.update_conversation_memory(conversation_id, ai_message)
            
            audit_logger.log_user_action(
                user_id or "system",
                "send_message",
                conversation_id,
                success=True,
                details={"message_length": len(message)}
            )
            
            return {
                "user_message": user_message,
                "ai_message": ai_message,
                "response_data": response_data
            }
            
        except Exception as e:
            logger.error("Failed to send message", error=e, extra={
                "conversation_id": conversation_id,
                "user_id": user_id
            })
            return None
    
    def _generate_response(self, query: str, history: List[ChatMessage], conversation_id: Optional[str] = None, multilingual_context: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """Generate AI response using RAG with multilingual support and analytics tracking"""
        start_time = time.time()
        
        try:
            # Reinitialize QA chain if needed
            if not self.qa_chain and document_service.vector_store:
                self._init_qa_chain()
            
            # Detect query language for bilingual search (returns BCP-47 code string)
            query_language = language_detection_service.detect_language(query)
            logger.info(f"Query language detected: {query_language}")
            
            # Build context-aware query using enhanced memory management
            context_query = self._build_context_prompt(history, query, conversation_id)
            
            if self.qa_chain:
                # Perform bilingual search - search in original language first
                result = self.qa_chain({"query": context_query})
                response_content = result["result"]
                source_documents = result.get("source_documents", [])
                
                # If no good results and query is in Japanese, try English translation
                if len(source_documents) < 2 and query_language == "ja":
                    logger.info("Limited results for Japanese query, attempting English translation")
                    translated_query = sota_translation_orchestrator.translate(
                        query,
                        target_language="en",
                        source_language="ja",
                        quality=TranslationQuality.BUSINESS
                    )
                    
                    if not translated_query.error:
                        english_context_query = self._build_context_prompt(history, translated_query.translated_text)
                        english_result = self.qa_chain({"query": english_context_query})
                        
                        # If English search yields better results, use them
                        if len(english_result.get("source_documents", [])) > len(source_documents):
                            result = english_result
                            response_content = result["result"]
                            source_documents = result.get("source_documents", [])
                            logger.info("Using English translation results for better coverage")
                
                # Similarly, if query is in English, try Japanese if needed
                elif len(source_documents) < 2 and query_language == "en":
                    logger.info("Limited results for English query, attempting Japanese translation")
                    translated_query = sota_translation_orchestrator.translate(
                        query,
                        target_language="ja",
                        source_language="en",
                        quality=TranslationQuality.BUSINESS
                    )
                    
                    if not translated_query.error:
                        japanese_context_query = self._build_context_prompt(history, translated_query.translated_text)
                        japanese_result = self.qa_chain({"query": japanese_context_query})
                        
                        if len(japanese_result.get("source_documents", [])) > len(source_documents):
                            result = japanese_result
                            response_content = result["result"]
                            source_documents = result.get("source_documents", [])
                            logger.info("Using Japanese translation results for better coverage")
                
                # Enhanced source attribution
                attribution = source_attribution_service.enhance_response_with_attribution(
                    response_content, source_documents, query
                )
                
                # Extract source information with enhanced attribution
                sources = []
                for citation in attribution.sources:
                    source_info = {
                        "content": citation.excerpt,
                        "metadata": citation.metadata,
                        "filename": citation.filename,
                        "document_id": citation.document_id,
                        "page_number": citation.page_number,
                        "section": citation.section,
                        "relevance_score": citation.relevance_score,
                        "confidence_score": citation.confidence_score,
                        "citation_type": citation.citation_type,
                        "detected_language": citation.metadata.get("detected_language", "unknown"),
                        "supports_translation": citation.metadata.get("supports_translation", False)
                    }
                    sources.append(source_info)
                
                # Enhanced response formatting with intelligent dual-model approach
                formatted_response_obj = intelligent_response_formatter.format_response_with_streamlit_components(
                    query, response_content, sources
                )
                
                # Enhanced response formatting with detailed source citations
                formatted_response = self._format_response_with_citations(
                    formatted_response_obj.content, attribution, query_language
                )
                
                return {
                    "content": formatted_response,
                    "metadata": {
                        "sources": sources,
                        "source_count": len(sources),
                        "response_type": "rag",
                        "query_language": query_language,
                        "bilingual_search_performed": len(source_documents) >= 2,
                        "confidence": attribution.overall_confidence,
                        "fact_check_status": attribution.fact_check_status,
                        "synthesis_type": attribution.synthesis_type,
                        "uncertainty_indicators": attribution.uncertainty_indicators,
                        "validation_notes": attribution.validation_notes,
                        "processing_time_ms": int((time.time() - start_time) * 1000),
                        "attribution": attribution.to_dict(),
                        "confidence_explanation": source_attribution_service.generate_confidence_explanation(attribution),
                        "formatted_citations": source_attribution_service.format_citations_for_display(attribution.sources),
                        # Intelligent formatting metadata
                        "intelligent_formatting": {
                            "response_format_type": formatted_response_obj.response_type.value,
                            "chart_data": formatted_response_obj.chart_data,
                            "table_data": formatted_response_obj.table_data.to_dict() if formatted_response_obj.table_data is not None else None,
                            "streamlit_components": formatted_response_obj.streamlit_components,
                            "formatting_metadata": formatted_response_obj.metadata
                        }
                    }
                }
            else:
                # Fallback to direct LLM response
                response = self.llm.invoke(context_query)
                
                return {
                    "content": response,
                    "metadata": {
                        "sources": [],
                        "source_count": 0,
                        "response_type": "direct",
                        "query_language": query_language,
                        "confidence": 0.5,
                        "processing_time_ms": int((time.time() - start_time) * 1000)
                    }
                }
                
        except Exception as e:
            logger.error("Failed to generate response", error=e, extra={"query": query[:100]})
            return None
    
    def stream_response(self, conversation_id: str, message: str, user_id: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        """Stream AI response with real token-by-token streaming and enhanced formatting"""
        try:
            # Verify conversation access
            if user_id:
                conversation = self.get_conversation(conversation_id, user_id)
                if not conversation:
                    yield {"error": "Conversation not found or access denied"}
                    return
            
            # Save user message
            user_message = db_manager.add_message(conversation_id, "user", message)
            if not user_message:
                yield {"error": "Failed to save user message"}
                return
            
            yield {"type": "user_message", "data": user_message.to_dict()}
            
            # Get conversation history for context
            history = self.get_conversation_history(conversation_id, user_id)
            
            # Build context-aware query with enhanced memory management
            context_query = conversation_memory_service.build_context_window(conversation_id, message, user_id)
            
            # Stream response with real-time token streaming
            response_content = ""
            sources = []
            
            if self._ensure_qa_chain():
                # Get source documents first for context
                try:
                    if self.qa_chain:
                        result = self.qa_chain({"query": context_query})
                        response_content = result["result"]
                        source_documents = result.get("source_documents", [])
                        
                        # Enhanced source attribution
                        attribution = source_attribution_service.enhance_response_with_attribution(
                            response_content, source_documents, context_query
                        )
                        
                        # Process sources with enhanced attribution
                        sources = []
                        for citation in attribution.sources:
                            source_info = {
                                "content": citation.excerpt,
                                "metadata": citation.metadata,
                                "filename": citation.filename,
                                "document_id": citation.document_id,
                                "page_number": citation.page_number,
                                "section": citation.section,
                                "relevance_score": citation.relevance_score,
                                "confidence_score": citation.confidence_score,
                                "citation_type": citation.citation_type
                            }
                            sources.append(source_info)
                        
                        # Send sources first
                        yield {
                            "type": "sources",
                            "data": {
                                "sources": sources,
                                "source_count": len(sources),
                                "confidence": attribution.overall_confidence,
                                "fact_check_status": attribution.fact_check_status,
                                "synthesis_type": attribution.synthesis_type,
                                "uncertainty_indicators": attribution.uncertainty_indicators
                            }
                        }
                    else:
                        # No QA chain available
                        sources = []
                        response_content = None
                    
                except Exception as e:
                    logger.warning(f"RAG streaming failed, falling back to direct LLM: {e}")
                    # Reset QA chain and fall through to direct LLM
                    self.qa_chain = None
                    sources = []
                    response_content = None
                
                # Process successful RAG response
                if response_content:
                    # Real token-by-token streaming simulation
                    # Enhanced response formatting with intelligent dual-model approach
                    formatted_response_obj = intelligent_response_formatter.format_response_with_streamlit_components(
                        message, response_content, sources
                    )
                    formatted_response = formatted_response_obj.content
                    
                    # Stream word by word for realistic effect
                    words = formatted_response.split()
                    current_text = ""
                    
                    for i, word in enumerate(words):
                        current_text += word + " "
                        
                        # Simulate realistic typing speed
                        time.sleep(0.05)  # 50ms per word - realistic typing speed
                        
                        yield {
                            "type": "response_chunk",
                            "data": {
                                "content": word + " ",
                                "accumulated_content": current_text,
                                "is_final": i == len(words) - 1,
                                "progress": (i + 1) / len(words)
                            }
                        }
                    
                    response_content = formatted_response
                else:
                    # Fall through to direct LLM if RAG failed
                    sources = []
            
            # Direct LLM fallback (either no QA chain or RAG failed)
            if not response_content:
                # Check if we have documents to provide contextual message
                if not self._has_documents():
                    # Special prompt for no documents case
                    no_docs_prompt = f"""
                    The user is asking: "{message}"
                    
                    However, no documents have been uploaded to the system yet. Please provide a helpful response that:
                    1. Acknowledges their question politely
                    2. Explains that no documents are currently available for analysis
                    3. Guides them to upload documents using the Documents page to get document-specific answers
                    4. If appropriate, provides general knowledge about their question
                    
                    Be friendly, professional, and encouraging.
                    """
                    context_to_use = no_docs_prompt
                else:
                    context_to_use = context_query
                
                # Fallback to direct LLM with streaming
                sources = []  # No sources for direct LLM
                try:
                    # Use Ollama's streaming capability if available
                    response_content = ""
                    for chunk in self.llm.stream(context_to_use):
                        response_content += chunk
                        yield {
                            "type": "response_chunk",
                            "data": {
                                "content": chunk,
                                "accumulated_content": response_content,
                                "is_final": False
                            }
                        }
                    
                    # Send final chunk
                    yield {
                        "type": "response_chunk",
                        "data": {
                            "content": "",
                            "accumulated_content": response_content,
                            "is_final": True
                        }
                    }
                    
                except AttributeError:
                    # Fallback for non-streaming LLM
                    response_content = self.llm.invoke(context_to_use)
                    # Enhanced response formatting with intelligent dual-model approach
                    formatted_response_obj = intelligent_response_formatter.format_response_with_streamlit_components(
                        message, response_content, []
                    )
                    formatted_response = formatted_response_obj.content
                    
                    # Stream word by word
                    words = formatted_response.split()
                    current_text = ""
                    
                    for i, word in enumerate(words):
                        current_text += word + " "
                        time.sleep(0.05)
                        
                        yield {
                            "type": "response_chunk",
                            "data": {
                                "content": word + " ",
                                "accumulated_content": current_text,
                                "is_final": i == len(words) - 1
                            }
                        }
                    
                    response_content = formatted_response
            
            # Save AI response with enhanced metadata
            ai_message = db_manager.add_message(
                conversation_id,
                "assistant",
                response_content,
                metadata={
                    "response_type": "rag" if self.qa_chain else "direct",
                    "streamed": True,
                    "sources": sources,
                    "confidence": attribution.overall_confidence if 'attribution' in locals() else 0.5,
                    "fact_check_status": attribution.fact_check_status if 'attribution' in locals() else "not_checked",
                    "synthesis_type": attribution.synthesis_type if 'attribution' in locals() else "unknown",
                    "word_count": len(response_content.split()),
                    "formatting_applied": True,
                    # Intelligent formatting metadata
                    "intelligent_formatting": {
                        "response_format_type": formatted_response_obj.response_type.value if 'formatted_response_obj' in locals() else "plain_text",
                        "chart_data": formatted_response_obj.chart_data if 'formatted_response_obj' in locals() else None,
                        "table_data": formatted_response_obj.table_data.to_dict() if 'formatted_response_obj' in locals() and formatted_response_obj.table_data is not None else None,
                        "streamlit_components": formatted_response_obj.streamlit_components if 'formatted_response_obj' in locals() else [],
                        "formatting_metadata": formatted_response_obj.metadata if 'formatted_response_obj' in locals() else {}
                    }
                }
            )
            
            if ai_message:
                yield {"type": "ai_message", "data": ai_message.to_dict()}
                
                # Update conversation memory with new message
                conversation_memory_service.update_conversation_memory(conversation_id, ai_message)
            
            audit_logger.log_user_action(
                user_id or "system",
                "stream_message",
                conversation_id,
                success=True,
                details={
                    "message_length": len(message),
                    "response_length": len(response_content),
                    "sources_count": len(sources)
                }
            )
            
        except Exception as e:
            logger.error("Failed to stream response", error=e, extra={
                "conversation_id": conversation_id,
                "user_id": user_id
            })
            yield {"error": f"Failed to generate response: {str(e)}"}
    
    def get_user_conversations(self, user_id: str) -> List[ChatConversation]:
        """Get all conversations for a user"""
        try:
            conversations = db_manager.get_user_conversations(user_id)
            return conversations
        except Exception as e:
            logger.error("Failed to get user conversations", error=e, extra={"user_id": user_id})
            return []
    
    def delete_conversation(self, conversation_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a conversation and all its messages"""
        try:
            # Verify user has access to conversation
            if user_id:
                conversation = self.get_conversation(conversation_id, user_id)
                if not conversation:
                    logger.warning(f"User {user_id} attempted to delete conversation {conversation_id}")
                    return False
            
            # Delete conversation and all messages using enhanced database method
            success = db_manager.delete_conversation(conversation_id, user_id)
            
            if success:
                logger.info(f"Conversation deleted", extra={
                    "conversation_id": conversation_id,
                    "user_id": user_id
                })
            
            return success
            
        except Exception as e:
            logger.error("Failed to delete conversation", error=e, extra={
                "conversation_id": conversation_id,
                "user_id": user_id
            })
            return False
    
    def get_conversation_summary(self, conversation_id: str, user_id: Optional[str] = None) -> Optional[str]:
        """Get conversation summary"""
        try:
            # Verify user has access to conversation
            if user_id:
                conversation = self.get_conversation(conversation_id, user_id)
                if not conversation:
                    return None
            
            # Get context which includes summary
            context = conversation_memory_service.get_conversation_context(conversation_id, user_id)
            return context.summary if context else None
            
        except Exception as e:
            logger.error("Failed to get conversation summary", error=e)
            return None
    
    def get_conversation_statistics(self, conversation_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed conversation statistics"""
        try:
            # Verify user has access to conversation
            if user_id:
                conversation = self.get_conversation(conversation_id, user_id)
                if not conversation:
                    return {"error": "Conversation not found or access denied"}
            
            return conversation_memory_service.get_conversation_statistics(conversation_id)
            
        except Exception as e:
            logger.error("Failed to get conversation statistics", error=e)
            return {"error": str(e)}
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages for translation"""
        return sota_translation_orchestrator.get_supported_languages()
    
    def validate_response_accuracy(self, response_content: str, source_documents: List[Any], query: str) -> Dict[str, Any]:
        """Validate response accuracy against source documents"""
        try:
            # Use source attribution service for validation
            attribution = source_attribution_service.enhance_response_with_attribution(
                response_content, source_documents, query
            )
            
            return {
                "overall_confidence": attribution.overall_confidence,
                "fact_check_status": attribution.fact_check_status,
                "validation_notes": attribution.validation_notes,
                "uncertainty_indicators": attribution.uncertainty_indicators,
                "source_count": len(attribution.sources),
                "synthesis_type": attribution.synthesis_type
            }
            
        except Exception as e:
            logger.error("Failed to validate response accuracy", error=e)
            return {
                "overall_confidence": 0.5,
                "fact_check_status": "error",
                "validation_notes": f"Validation failed: {str(e)}",
                "uncertainty_indicators": [],
                "source_count": 0,
                "synthesis_type": "unknown"
            }
    
    def synthesize_multi_document_response(self, query: str, documents: List[Any], max_sources: int = 5) -> Dict[str, Any]:
        """Synthesize comprehensive response from multiple documents"""
        try:
            if not documents:
                return {
                    "content": "I don't have any documents to reference for this query.",
                    "metadata": {
                        "response_type": "no_documents",
                        "confidence": 0.0,
                        "sources": []
                    }
                }
            
            # Limit documents for processing efficiency
            selected_docs = documents[:max_sources]
            
            # Create synthesis prompt
            doc_summaries = []
            for i, doc in enumerate(selected_docs, 1):
                content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                filename = doc.metadata.get("filename", f"Document {i}")
                doc_summaries.append(f"**Source {i} ({filename}):**\n{content_preview}")
            
            synthesis_prompt = f"""
Based on the following source documents, provide a comprehensive answer to the user's question. 
Synthesize information from multiple sources when possible, and clearly indicate when information comes from specific sources.

Question: {query}

Source Documents:
{chr(10).join(doc_summaries)}

Instructions:
- Provide a comprehensive answer that synthesizes information from the sources
- Use specific citations like [Source 1], [Source 2] when referencing information
- If sources conflict, acknowledge the discrepancy
- If information is incomplete, indicate what's missing
- Maintain professional tone and clear structure

Comprehensive Response:"""
            
            # Generate response
            response_content = self.llm.invoke(synthesis_prompt)
            
            # Enhanced attribution
            attribution = source_attribution_service.enhance_response_with_attribution(
                response_content, selected_docs, query
            )
            
            return {
                "content": response_content,
                "metadata": {
                    "response_type": "multi_document_synthesis",
                    "confidence": attribution.overall_confidence,
                    "fact_check_status": attribution.fact_check_status,
                    "synthesis_type": attribution.synthesis_type,
                    "sources": [citation.to_dict() for citation in attribution.sources],
                    "source_count": len(attribution.sources),
                    "uncertainty_indicators": attribution.uncertainty_indicators,
                    "validation_notes": attribution.validation_notes
                }
            }
            
        except Exception as e:
            logger.error("Failed to synthesize multi-document response", error=e)
            return {
                "content": f"I encountered an error while synthesizing information from the documents: {str(e)}",
                "metadata": {
                    "response_type": "error",
                    "confidence": 0.0,
                    "sources": []
                }
            }
    
    def get_response_citations(self, response_metadata: Dict[str, Any]) -> str:
        """Format response citations for display"""
        try:
            sources = response_metadata.get("sources", [])
            if not sources:
                return "No sources available for this response."
            
            # Use source attribution service for formatting
            citations = []
            for source_data in sources:
                if isinstance(source_data, dict):
                    # Convert dict back to SourceCitation for formatting
                    from services.source_attribution import SourceCitation
                    citation = SourceCitation(
                        document_id=source_data.get("document_id", ""),
                        filename=source_data.get("filename", "Unknown"),
                        page_number=source_data.get("page_number"),
                        section=source_data.get("section"),
                        excerpt=source_data.get("content", ""),
                        relevance_score=source_data.get("relevance_score", 0.0),
                        confidence_score=source_data.get("confidence_score", 0.0),
                        citation_type=source_data.get("citation_type", "unknown"),
                        metadata=source_data.get("metadata", {})
                    )
                    citations.append(citation)
            
            return source_attribution_service.format_citations_for_display(citations)
            
        except Exception as e:
            logger.error("Failed to format response citations", error=e)
            return "Error formatting citations."
    
    def get_confidence_explanation(self, response_metadata: Dict[str, Any]) -> str:
        """Get human-readable confidence explanation"""
        try:
            attribution_data = response_metadata.get("attribution")
            if not attribution_data:
                # Fallback for basic confidence info
                confidence = response_metadata.get("confidence", 0.5)
                source_count = response_metadata.get("source_count", 0)
                
                if confidence >= 0.8:
                    level = "High"
                elif confidence >= 0.5:
                    level = "Medium"
                else:
                    level = "Low"
                
                return f"**Confidence: {level} ({confidence:.1%})**\n\nBased on {source_count} source document(s)."
            
            # Use source attribution service for detailed explanation
            from services.source_attribution import ResponseAttribution
            attribution = ResponseAttribution(
                response_content="",  # Not needed for explanation
                overall_confidence=attribution_data.get("overall_confidence", 0.5),
                sources=[],  # Will be populated from metadata
                fact_check_status=attribution_data.get("fact_check_status", "not_checked"),
                synthesis_type=attribution_data.get("synthesis_type", "unknown"),
                uncertainty_indicators=attribution_data.get("uncertainty_indicators", []),
                validation_notes=attribution_data.get("validation_notes")
            )
            
            return source_attribution_service.generate_confidence_explanation(attribution)
            
        except Exception as e:
            logger.error("Failed to generate confidence explanation", error=e)
            return "Unable to generate confidence explanation."


    def _build_enhanced_context_prompt(self, messages: List[ChatMessage], current_query: str) -> str:
        """Build enhanced context-aware prompt with intelligent summarization"""
        try:
            # Configuration for context management
            max_recent_messages = 10
            max_context_tokens = 2000  # Approximate token limit
            
            # Get recent messages
            recent_messages = messages[-max_recent_messages:] if len(messages) > max_recent_messages else messages
            
            # If we have many messages, create a summary of older ones
            summary = ""
            if len(messages) > max_recent_messages:
                older_messages = messages[:-max_recent_messages]
                summary = self._create_conversation_summary(older_messages)
            
            context_parts = []
            
            # Add system prompt for enhanced formatting
            context_parts.append("You are Kurachi AI, a professional business document assistant.")
            context_parts.append("Provide clear, well-formatted responses with:")
            context_parts.append("- Use **bold** for important terms")
            context_parts.append("- Use bullet points for lists")
            context_parts.append("- Use tables for data comparisons")
            context_parts.append("- Use code blocks for technical content")
            context_parts.append("- Provide specific citations when referencing documents")
            context_parts.append("")
            
            # Add conversation summary if available
            if summary:
                context_parts.append("Previous conversation summary:")
                context_parts.append(summary)
                context_parts.append("")
            
            # Add recent conversation history
            if recent_messages:
                context_parts.append("Recent conversation:")
                for msg in recent_messages:
                    role = "Human" if msg.role == "user" else "Assistant"
                    # Truncate very long messages
                    content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
                    context_parts.append(f"{role}: {content}")
                context_parts.append("")
            
            # Add current query
            context_parts.append(f"Current question: {current_query}")
            context_parts.append("\nBased on the conversation history and document context, please provide a comprehensive, well-formatted response:")
            
            full_context = "\n".join(context_parts)
            
            # Estimate token count and truncate if necessary
            if len(full_context) > max_context_tokens * 4:  # Rough estimate (4 chars per token)
                logger.warning("Context too long, truncating")
                truncated_context = full_context[:max_context_tokens * 4]
                return truncated_context + "\n\n" + f"Current question: {current_query}"
            
            return full_context
            
        except Exception as e:
            logger.error("Failed to build enhanced context prompt", error=e)
            return current_query
    
    def _create_conversation_summary(self, messages: List[ChatMessage]) -> str:
        """Create an intelligent summary of conversation history"""
        try:
            if not messages:
                return ""
            
            # Extract key topics and themes
            conversation_text = ""
            for msg in messages:
                role = "User" if msg.role == "user" else "Assistant"
                conversation_text += f"{role}: {msg.content}\n"
            
            # Use LLM to create summary
            summary_prompt = f"""
            Please create a concise summary of this conversation, focusing on:
            - Key topics discussed
            - Important questions asked
            - Main findings or conclusions
            - Any specific document references
            
            Conversation:
            {conversation_text[:2000]}...
            
            Summary (max 200 words):
            """
            
            try:
                summary = self.llm.invoke(summary_prompt)
                return f"Summary: {summary}"
            except Exception as e:
                logger.error("Failed to generate LLM summary", error=e)
                # Fallback: extract key phrases
                return self._extract_key_topics(messages)
                
        except Exception as e:
            logger.error("Failed to create conversation summary", error=e)
            return ""
    
    def _extract_key_topics(self, messages: List[ChatMessage]) -> str:
        """Extract key topics as fallback summary method"""
        try:
            # Simple keyword extraction
            all_text = " ".join([msg.content for msg in messages])
            words = all_text.lower().split()
            
            # Filter common words and find important terms
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
            
            word_freq = {}
            for word in words:
                if len(word) > 3 and word not in stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top keywords
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            keywords = [word for word, freq in top_words if freq > 1]
            
            if keywords:
                return f"Key topics discussed: {', '.join(keywords[:5])}"
            else:
                return "Previous conversation covered various topics."
                
        except Exception as e:
            logger.error("Failed to extract key topics", error=e)
            return "Previous conversation history available."
    
    def get_conversation_context_info(self, conversation_id: str) -> Dict[str, Any]:
        """Get context information about a conversation"""
        try:
            messages = db_manager.get_conversation_messages(conversation_id)
            
            if not messages:
                return {"message_count": 0, "context_length": 0}
            
            total_chars = sum(len(msg.content) for msg in messages)
            user_messages = [msg for msg in messages if msg.role == "user"]
            assistant_messages = [msg for msg in messages if msg.role == "assistant"]
            
            # Calculate average response time (if timestamps available)
            avg_response_time = None
            if len(messages) >= 2:
                response_times = []
                for i in range(1, len(messages)):
                    if messages[i-1].role == "user" and messages[i].role == "assistant":
                        time_diff = (messages[i].created_at - messages[i-1].created_at).total_seconds()
                        response_times.append(time_diff)
                
                if response_times:
                    avg_response_time = sum(response_times) / len(response_times)
            
            return {
                "message_count": len(messages),
                "user_messages": len(user_messages),
                "assistant_messages": len(assistant_messages), 
                "total_characters": total_chars,
                "context_length": total_chars,
                "avg_response_time": avg_response_time,
                "conversation_started": messages[0].created_at if messages else None,
                "last_activity": messages[-1].created_at if messages else None
            }
            
        except Exception as e:
            logger.error("Failed to get conversation context info", error=e)
            return {"message_count": 0, "context_length": 0}
    
    def _process_enhanced_sources(self, source_documents) -> List[Dict[str, Any]]:
        """Process sources with enhanced metadata and confidence scoring"""
        sources = []
        for doc in source_documents:
            # Calculate relevance score (simplified)
            relevance_score = min(1.0, len(doc.page_content) / 1000)  # Higher score for longer content
            
            source = {
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "metadata": doc.metadata,
                "filename": doc.metadata.get("filename", "Unknown"),
                "document_id": doc.metadata.get("document_id"),
                "page_number": doc.metadata.get("page", "N/A"),
                "relevance_score": relevance_score,
                "chunk_type": doc.metadata.get("chunk_type", "text")
            }
            sources.append(source)
        
        # Sort by relevance score
        sources.sort(key=lambda x: x["relevance_score"], reverse=True)
        return sources
    

    
    def _format_response_content(self, content: str, query: str) -> str:
        """Apply intelligent formatting to response content"""
        # Detect if query asks for table/structured data
        table_keywords = ["compare", "list", "table", "summary", "overview", "breakdown"]
        needs_table = any(keyword in query.lower() for keyword in table_keywords)
        
        # Basic formatting improvements
        formatted = content
        
        # Make key terms bold
        formatted = re.sub(r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*?)\b(?=\s*:)', r'**\1**', formatted)
        
        # Format lists better
        lines = formatted.split('\n')
        formatted_lines = []
        
        for line in lines:
            # Convert numbered lists
            if re.match(r'^\d+\.\s', line.strip()):
                formatted_lines.append(line)
            # Convert bullet points
            elif line.strip().startswith('-') or line.strip().startswith('*'):
                formatted_lines.append(line.replace('-', '•').replace('*', '•'))
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _format_response_with_citations(
        self, 
        response: str, 
        attribution, 
        query_language
    ) -> str:
        """Format response with detailed source citations and confidence indicators"""
        try:
            # Start with the main response
            formatted_parts = [response]
            
            # Add confidence indicator if below high threshold
            if attribution.overall_confidence < 0.8:
                confidence_indicator = self._generate_confidence_indicator(attribution.overall_confidence)
                formatted_parts.append(f"\n\n**Confidence Level:** {confidence_indicator}")
            
            # Add uncertainty indicators if present
            if attribution.uncertainty_indicators:
                formatted_parts.append(f"\n\n**Please Note:** {'; '.join(attribution.uncertainty_indicators[:3])}")
            
            # Add source citations
            if attribution.sources:
                citations_text = self._format_inline_citations(attribution.sources)
                formatted_parts.append(f"\n\n**Sources:**\n{citations_text}")
            
            # Add fact-check status if relevant
            if attribution.fact_check_status in ["conflicting", "questionable"]:
                fact_check_note = self._generate_fact_check_note(attribution.fact_check_status)
                formatted_parts.append(f"\n\n**Verification Note:** {fact_check_note}")
            
            # Add synthesis information for multi-document responses
            if len(attribution.sources) > 1:
                synthesis_note = self._generate_synthesis_note(attribution.synthesis_type, len(attribution.sources))
                formatted_parts.append(f"\n\n**Information Synthesis:** {synthesis_note}")
            
            return "".join(formatted_parts)
            
        except Exception as e:
            logger.warning("Failed to format response with citations", error=e)
            return response
    
    def _generate_confidence_indicator(self, confidence: float) -> str:
        """Generate human-readable confidence indicator"""
        if confidence >= 0.8:
            return f"High ({confidence:.0%}) - Well supported by sources"
        elif confidence >= 0.6:
            return f"Medium ({confidence:.0%}) - Moderately supported by sources"
        elif confidence >= 0.4:
            return f"Low ({confidence:.0%}) - Limited source support"
        else:
            return f"Very Low ({confidence:.0%}) - Minimal source support"
    
    def _format_inline_citations(self, citations) -> str:
        """Format citations for inline display"""
        try:
            formatted_citations = []
            
            for i, citation in enumerate(citations[:5], 1):  # Limit to top 5 sources
                citation_parts = [f"[{i}] **{citation.filename}**"]
                
                if citation.page_number:
                    citation_parts.append(f"(Page {citation.page_number})")
                
                if citation.section:
                    citation_parts.append(f"- {citation.section}")
                
                # Add confidence and relevance scores
                citation_parts.append(f"*Confidence: {citation.confidence_score:.0%}, Relevance: {citation.relevance_score:.0%}*")
                
                # Add excerpt if available
                if citation.excerpt and len(citation.excerpt) > 20:
                    excerpt = citation.excerpt[:150] + "..." if len(citation.excerpt) > 150 else citation.excerpt
                    citation_parts.append(f"\n   > {excerpt}")
                
                formatted_citations.append(" ".join(citation_parts))
            
            return "\n".join(formatted_citations)
            
        except Exception as e:
            logger.warning("Failed to format inline citations", error=e)
            return "Citations formatting failed"
    
    def _generate_fact_check_note(self, fact_check_status: str) -> str:
        """Generate fact-check note based on status"""
        if fact_check_status == "conflicting":
            return "Some conflicting information was detected across sources. Please verify important details independently."
        elif fact_check_status == "questionable":
            return "Some information could not be fully verified against available sources. Use with caution."
        elif fact_check_status == "insufficient_sources":
            return "Limited source material available for complete verification."
        else:
            return "Information has been cross-referenced with available sources."
    
    def _generate_synthesis_note(self, synthesis_type: str, source_count: int) -> str:
        """Generate synthesis note for multi-document responses"""
        if synthesis_type.startswith("comprehensive_synthesis"):
            return f"This response synthesizes information from {source_count} different sources to provide a comprehensive answer."
        elif synthesis_type.startswith("multi_source"):
            return f"This response combines information from {source_count} sources."
        elif synthesis_type.startswith("dual_source"):
            return f"This response is based on information from {source_count} sources."
        else:
            return f"This response draws from {source_count} source documents."
    
    def translate_query_if_needed(self, query: str, target_language: str) -> Dict[str, Any]:
        """Translate query if needed for cross-language search using SOTA orchestrator"""
        try:
            query_language = language_detection_service.detect_language(query)
            
            if query_language != target_language:
                translation_result = sota_translation_orchestrator.translate(
                    query,
                    target_language=target_language,
                    source_language=query_language,
                    quality=TranslationQuality.BUSINESS
                )
                
                if not translation_result.error:
                    return {
                        "success": True,
                        "translated_query": translation_result.translated_text,
                        "original_query": query,
                        "source_language": query_language,
                        "target_language": target_language,
                        "confidence": translation_result.confidence
                    }
                else:
                    return {
                        "success": False,
                        "error": translation_result.error,
                        "original_query": query
                    }
            else:
                return {
                    "success": True,
                    "translated_query": query,
                    "original_query": query,
                    "source_language": query_language,
                    "target_language": target_language,
                    "confidence": 1.0,
                    "no_translation_needed": True
                }
                
        except Exception as e:
            logger.error("Failed to translate query", error=e)
            return {
                "success": False,
                "error": str(e),
                "original_query": query
            }
    
    


# Global chat service instance
chat_service = ChatService()