#!/usr/bin/env python3
"""
FastAPI backend for Kurachi AI
Serves React frontend and provides API endpoints
"""
import sys
import os
from pathlib import Path

# Add the parent directory to Python path to import from the main project
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import uuid
from datetime import datetime
import json

# Import from the main project
from config import config
from utils.logger import get_logger
from models.database import db_manager
from services.sota_registry import get_service

# Initialize logger
logger = get_logger("fastapi_backend")

# Create FastAPI app
app = FastAPI(
    title="Kurachi AI API",
    description="Advanced AI Assistant with multilingual capabilities",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class ChatMessage(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    language: str = "en"
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 2048

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    message_id: str
    response_time: float
    metadata: Dict[str, Any] = {}

class Conversation(BaseModel):
    id: str
    title: Optional[str]
    created_at: str
    last_message: Optional[str] = None
    message_count: int = 0

class ConversationCreate(BaseModel):
    title: Optional[str] = None
    user_id: Optional[str] = None

class UserPreferences(BaseModel):
    language: str = "en"
    theme: str = "dark"
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 2048

# Global variables for services
chat_service = None
translation_service = None
document_service = None
analytics_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global chat_service, translation_service, document_service, analytics_service
    
    try:
        logger.info("Initializing Kurachi AI services...")
        
        # Initialize services
        chat_service = get_service('sota_chat')
        translation_service = get_service('sota_translation')
        document_service = get_service('document')
        analytics_service = get_service('analytics')
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        # Continue without services for now

@app.get("/")
async def root():
    """Serve the React frontend"""
    return FileResponse("frontend/build/index.html")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {
            "chat": chat_service is not None,
            "translation": translation_service is not None,
            "document": document_service is not None,
            "analytics": analytics_service is not None,
        }
    }

@app.get("/api/system/status")
async def get_system_status():
    """Get system status and configuration"""
    return {
        "app_name": config.app.app_name,
        "version": config.app.version,
        "debug": config.app.debug,
        "database": {
            "path": config.database.sqlite_path,
            "connected": True  # Add actual connection check
        },
        "ai_models": {
            "llm": config.ai.llm_model,
            "vision": config.ai.vision_model,
            "embedding": config.ai.embedding_model,
        },
        "features": {
            "ocr": config.app.enable_ocr,
            "audio_transcription": config.app.enable_audio_transcription,
            "video_processing": config.app.enable_video_processing,
            "neo4j": config.app.enable_neo4j,
            "spacy": config.app.enable_spacy,
        }
    }

@app.get("/api/system/config")
async def get_system_config():
    """Get system configuration"""
    return {
        "app": config.app.dict(),
        "ai": config.ai.dict(),
        "database": config.database.dict(),
    }

# Chat API endpoints
@app.post("/api/chat/send", response_model=ChatResponse)
async def send_message(message: ChatMessage):
    """Send a message and get AI response"""
    try:
        start_time = datetime.now()
        
        # Create conversation if none exists
        conversation_id = message.conversation_id
        if not conversation_id:
            conversation = await create_conversation_internal()
            conversation_id = conversation["id"]
        
        # Send message to chat service
        if chat_service:
            response_data = await chat_service.send_message_async(
                conversation_id=conversation_id,
                message=message.message,
                user_id=None  # Add user authentication later
            )
            
            response_text = response_data.get("response", "I'm sorry, I couldn't process your request.")
            message_id = str(uuid.uuid4())
            
        else:
            # Fallback response
            response_text = "I'm currently initializing. Please try again in a moment."
            message_id = str(uuid.uuid4())
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Save message to database
        db_manager.add_message(conversation_id, "user", message.message)
        db_manager.add_message(conversation_id, "assistant", response_text)
        
        # Track analytics
        if analytics_service:
            analytics_service.track_event("message_sent", {
                "conversation_id": conversation_id,
                "message_length": len(message.message),
                "response_time": response_time,
                "language": message.language,
            })
        
        return ChatResponse(
            response=response_text,
            conversation_id=conversation_id,
            message_id=message_id,
            response_time=response_time,
            metadata={
                "model": message.model,
                "temperature": message.temperature,
                "max_tokens": message.max_tokens,
                "language": message.language,
            }
        )
        
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/conversations")
async def get_conversations(user_id: Optional[str] = None):
    """Get all conversations"""
    try:
        conversations = db_manager.get_conversations(user_id)
        return {"conversations": conversations}
    except Exception as e:
        logger.error(f"Error getting conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/conversations")
async def create_conversation(conversation: ConversationCreate):
    """Create a new conversation"""
    try:
        return await create_conversation_internal(conversation.title, conversation.user_id)
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def create_conversation_internal(title: Optional[str] = None, user_id: Optional[str] = None):
    """Internal function to create conversation"""
    conversation_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
    conversation_data = {
        "id": conversation_id,
        "title": title or f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "created_at": created_at,
        "user_id": user_id,
        "last_message": None,
        "message_count": 0
    }
    
    # Save to database
    db_manager.create_conversation(conversation_id, title, user_id)
    
    return conversation_data

@app.get("/api/chat/conversations/{conversation_id}/messages")
async def get_messages(conversation_id: str):
    """Get messages for a specific conversation"""
    try:
        messages = db_manager.get_messages(conversation_id)
        return {"messages": messages}
    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/chat/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    try:
        db_manager.delete_conversation(conversation_id)
        return {"message": "Conversation deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/messages/{message_id}/regenerate")
async def regenerate_response(message_id: str):
    """Regenerate AI response for a message"""
    try:
        # Get the original message
        message = db_manager.get_message(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        # Regenerate response
        if chat_service:
            response_data = await chat_service.send_message_async(
                conversation_id=message["conversation_id"],
                message=message["content"],
                user_id=None
            )
            
            new_response = response_data.get("response", "I'm sorry, I couldn't regenerate the response.")
        else:
            new_response = "I'm currently initializing. Please try again in a moment."
        
        # Update message in database
        db_manager.update_message(message_id, new_response)
        
        return ChatResponse(
            response=new_response,
            conversation_id=message["conversation_id"],
            message_id=message_id,
            response_time=0.0,
            metadata={"regenerated": True}
        )
        
    except Exception as e:
        logger.error(f"Error regenerating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Translation API endpoints
@app.post("/api/translation/translate")
async def translate_text(text: str, source_language: str, target_language: str):
    """Translate text between languages"""
    try:
        if translation_service:
            result = translation_service.translate(text, source_language, target_language)
            return {"translated_text": result, "source_language": source_language, "target_language": target_language}
        else:
            return {"translated_text": text, "source_language": source_language, "target_language": target_language, "note": "Translation service not available"}
    except Exception as e:
        logger.error(f"Error translating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/translation/languages")
async def get_supported_languages():
    """Get supported languages"""
    languages = [
        {"code": "en", "name": "English", "flag": "🇺🇸"},
        {"code": "ja", "name": "Japanese", "flag": "🇯🇵"},
        {"code": "es", "name": "Spanish", "flag": "🇪🇸"},
        {"code": "fr", "name": "French", "flag": "🇫🇷"},
        {"code": "de", "name": "German", "flag": "🇩🇪"},
        {"code": "it", "name": "Italian", "flag": "🇮🇹"},
        {"code": "pt", "name": "Portuguese", "flag": "🇵🇹"},
        {"code": "ru", "name": "Russian", "flag": "🇷🇺"},
        {"code": "zh", "name": "Chinese", "flag": "🇨🇳"},
        {"code": "ko", "name": "Korean", "flag": "🇰🇷"},
    ]
    return {"languages": languages}

@app.post("/api/translation/detect")
async def detect_language(text: str):
    """Detect language of text"""
    try:
        # Simple language detection (in production, use a proper library)
        if any(char in text for char in "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"):
            detected_lang = "ja"
        elif any(char in text for char in "áéíóúñü"):
            detected_lang = "es"
        else:
            detected_lang = "en"
        
        return {"detected_language": detected_lang, "confidence": 0.8}
    except Exception as e:
        logger.error(f"Error detecting language: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Document API endpoints
@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    language: str = Form("en"),
    enable_ocr: bool = Form(True),
    enable_transcription: bool = Form(True)
):
    """Upload a document for processing"""
    try:
        # Save file
        file_path = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process document
        if document_service:
            result = document_service.process_document(file_path, {
                "language": language,
                "enable_ocr": enable_ocr,
                "enable_transcription": enable_transcription
            })
        else:
            result = {"status": "uploaded", "message": "Document uploaded successfully"}
        
        return {
            "document_id": str(uuid.uuid4()),
            "filename": file.filename,
            "file_size": len(content),
            "language": language,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def get_documents():
    """Get uploaded documents"""
    try:
        # This would typically query the database
        documents = []
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document"""
    try:
        # This would typically delete from database and file system
        return {"message": "Document deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/search")
async def search_documents(q: str):
    """Search documents"""
    try:
        if document_service:
            results = document_service.search(q)
        else:
            results = []
        
        return {"results": results, "query": q}
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics API endpoints
@app.post("/api/analytics/track")
async def track_event(event: str, data: Dict[str, Any]):
    """Track analytics event"""
    try:
        if analytics_service:
            analytics_service.track_event(event, data)
        
        return {"status": "tracked", "event": event}
    except Exception as e:
        logger.error(f"Error tracking event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics")
async def get_analytics(period: str = "7d"):
    """Get analytics data"""
    try:
        if analytics_service:
            data = analytics_service.get_analytics(period)
        else:
            data = {"period": period, "events": []}
        
        return data
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# User API endpoints
@app.get("/api/user/profile")
async def get_user_profile():
    """Get user profile"""
    try:
        # This would typically get from database with authentication
        profile = {
            "id": str(uuid.uuid4()),
            "name": "User",
            "email": "user@example.com",
            "created_at": datetime.now().isoformat(),
            "preferences": {
                "language": "en",
                "theme": "dark",
                "model": "default",
                "temperature": 0.7,
                "max_tokens": 2048,
            }
        }
        return profile
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/user/preferences")
async def update_user_preferences(preferences: UserPreferences):
    """Update user preferences"""
    try:
        # This would typically save to database
        return {"status": "updated", "preferences": preferences.dict()}
    except Exception as e:
        logger.error(f"Error updating preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/user/stats")
async def get_user_stats():
    """Get user statistics"""
    try:
        stats = {
            "total_conversations": 0,
            "total_messages": 0,
            "total_tokens": 0,
            "favorite_topics": [],
            "usage_trends": []
        }
        return stats
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files (React build)
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """Serve React frontend for all other routes"""
    # Check if the file exists in the build directory
    file_path = Path("frontend/build") / full_path
    
    if file_path.exists() and file_path.is_file():
        return FileResponse(str(file_path))
    else:
        # Serve index.html for client-side routing
        return FileResponse("frontend/build/index.html")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
