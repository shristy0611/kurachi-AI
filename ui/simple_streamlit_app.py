#!/usr/bin/env python3
"""
Simplified Streamlit interface for Kurachi AI
Works without complex dependencies for testing
"""
import sys
import os
sys.path.append('.')

import streamlit as st
from datetime import datetime
import uuid
import time

# Basic imports that work
from config import config
from utils.logger import get_logger
from models.database import db_manager

logger = get_logger("simple_streamlit_app")

# Multilingual text support
UI_TEXT = {
    "en": {
        "app_name": "🧠 Kurachi AI",
        "welcome_title": "Welcome to Kurachi AI",
        "welcome_message": "Your intelligent AI assistant with multilingual capabilities, document analysis, and advanced knowledge processing.",
        "chat_placeholder": "Ask me anything...",
        "upload_files": "Upload Documents",
        "new_chat": "New Chat",
        "theme_light": "☀️ Light",
        "theme_dark": "🌙 Dark",
        "language_en": "🇺🇸 English",
        "language_ja": "🇯🇵 Japanese"
    },
    "ja": {
        "app_name": "🧠 蔵地AI",
        "welcome_title": "蔵地AIへようこそ",
        "welcome_message": "多言語対応、文書分析、高度な知識処理を備えたインテリジェントAIアシスタントです。",
        "chat_placeholder": "何でも聞いてください...",
        "upload_files": "文書アップロード",
        "new_chat": "新しいチャット",
        "theme_light": "☀️ ライト",
        "theme_dark": "🌙 ダーク",
        "language_en": "🇺🇸 英語",
        "language_ja": "🇯🇵 日本語"
    }
}

def get_text(key: str, lang: str = "en") -> str:
    """Get localized text"""
    return UI_TEXT.get(lang, UI_TEXT["en"]).get(key, key)

class SimpleKurachiApp:
    """Simplified Kurachi AI Streamlit application"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Kurachi AI",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Modern CSS
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .stApp {
            font-family: 'Inter', sans-serif;
        }
        
        .main-header {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 700;
            font-size: 2.5rem;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .welcome-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            margin: 2rem 0;
        }
        
        .feature-badge {
            background: rgba(99, 102, 241, 0.1);
            color: #6366f1;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            margin: 0.25rem;
            display: inline-block;
        }
        
        .chat-message {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .user-message {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            margin-left: 2rem;
        }
        
        .assistant-message {
            background: rgba(255, 255, 255, 0.05);
            margin-right: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if "user_id" not in st.session_state:
            st.session_state.user_id = str(uuid.uuid4())
        
        if "language" not in st.session_state:
            st.session_state.language = "en"
        
        if "theme" not in st.session_state:
            st.session_state.theme = "light"
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
    
    def render_sidebar(self):
        """Render the sidebar with controls"""
        with st.sidebar:
            st.title(get_text("app_name", st.session_state.language))
            
            # Language selector
            st.markdown("### 🌐 Language")
            lang_col1, lang_col2 = st.columns(2)
            
            with lang_col1:
                if st.button(get_text("language_en", st.session_state.language), use_container_width=True):
                    st.session_state.language = "en"
                    st.rerun()
            
            with lang_col2:
                if st.button(get_text("language_ja", st.session_state.language), use_container_width=True):
                    st.session_state.language = "ja"
                    st.rerun()
            
            st.divider()
            
            # Theme selector
            st.markdown("### 🎨 Theme")
            theme_col1, theme_col2 = st.columns(2)
            
            with theme_col1:
                if st.button(get_text("theme_light", st.session_state.language), use_container_width=True):
                    st.session_state.theme = "light"
                    st.rerun()
            
            with theme_col2:
                if st.button(get_text("theme_dark", st.session_state.language), use_container_width=True):
                    st.session_state.theme = "dark"
                    st.rerun()
            
            st.divider()
            
            # System info
            st.markdown("### 🔧 System Info")
            st.info(f"User ID: {st.session_state.user_id[:12]}...")
            st.success("✅ Database Connected")
            st.success("✅ Configuration Loaded")
            
            # New chat button
            if st.button(get_text("new_chat", st.session_state.language), use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    
    def render_welcome_screen(self):
        """Render the welcome screen"""
        st.markdown(f'<h1 class="main-header">{get_text("app_name", st.session_state.language)}</h1>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="welcome-card">
            <h2>{get_text("welcome_title", st.session_state.language)}</h2>
            <p>{get_text("welcome_message", st.session_state.language)}</p>
            <div style="margin-top: 1rem;">
                <span class="feature-badge">🌐 Multilingual</span>
                <span class="feature-badge">📄 Document Analysis</span>
                <span class="feature-badge">🧠 AI-Powered</span>
                <span class="feature-badge">⚡ Fast & Secure</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_chat_interface(self):
        """Render the chat interface"""
        # Display messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>AI:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input(get_text("chat_placeholder", st.session_state.language)):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Simulate AI response
            ai_response = self.generate_simple_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            
            st.rerun()
    
    def generate_simple_response(self, prompt: str) -> str:
        """Generate a simple response for testing"""
        # Simple response logic for testing
        if "hello" in prompt.lower() or "hi" in prompt.lower():
            return "Hello! I'm Kurachi AI, your intelligent assistant. How can I help you today?"
        elif "help" in prompt.lower():
            return "I can help you with document analysis, multilingual conversations, and answering questions. Try uploading a document or asking me anything!"
        elif "upload" in prompt.lower() or "document" in prompt.lower():
            return "You can upload documents using the file uploader. I support PDF, DOCX, TXT, and many other formats. I'll analyze the content and can answer questions about it."
        else:
            return f"I understand you said: '{prompt}'. This is a simplified version of Kurachi AI for testing. The full version would provide more sophisticated responses with document analysis and multilingual capabilities."
    
    def run(self):
        """Run the simplified application"""
        try:
            # Render sidebar
            self.render_sidebar()
            
            # Main content
            if not st.session_state.messages:
                self.render_welcome_screen()
            else:
                self.render_chat_interface()
            
            # Footer
            st.markdown("---")
            st.markdown(f"<div style='text-align: center; color: #666;'>Kurachi AI v{config.app.version} | Built with ❤️</div>", unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"Error in Streamlit app: {e}")
            st.error(f"An error occurred: {str(e)}")

def main():
    """Main entry point"""
    app = SimpleKurachiApp()
    app.run()

if __name__ == "__main__":
    main()
