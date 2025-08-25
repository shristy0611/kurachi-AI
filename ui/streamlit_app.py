"""
Enhanced Streamlit interface for Kurachi AI (SOTA Enhanced)
Provides file upload, document management, and improved chat UI
Now using SOTA services for optimal performance
"""
import sys
import os
# Add parent directory to Python path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from datetime import datetime
from typing import Optional, List
import uuid
import time
import json

# Core imports
from datetime import datetime
from typing import Optional, List
import uuid
import time
import json

# Basic configuration import
from config import config
from models.database import db_manager
from utils.logger import get_logger

# Try SOTA imports with fallbacks
try:
    from services.sota_config import sota_config, get_config
    from services.sota_registry import get_service
    from services.sota_observability import trace_operation, record_metric, MetricType
    SOTA_AVAILABLE = True
except ImportError as e:
    print(f"SOTA imports failed, using fallbacks: {e}")
    SOTA_AVAILABLE = False
    # Fallback functions
    def get_service(name):
        if name == 'sota_chat':
            from services.chat_service import ChatService
            return ChatService()
        elif name == 'sota_translation':
            from services.translation_service import translation_service
            return translation_service
        elif name == 'analytics':
            from services.analytics_service import AnalyticsService
            return AnalyticsService()
        return None
    
    def trace_operation(name, **kwargs):
        from contextlib import nullcontext
        return nullcontext()
    
    def record_metric(name, value, metric_type, tags=None):
        pass
        
    class MetricType:
        COUNTER = "counter"
    
    def get_config():
        return config
# Service imports with fallbacks
try:
    from services.document_service import document_service
except ImportError:
    document_service = None

try:
    from services.knowledge_graph import knowledge_graph
except ImportError:
    knowledge_graph = None

try:
    from ui.streamlit_components import streamlit_component_renderer
except ImportError:
    streamlit_component_renderer = None

try:
    from ui.multilingual_components import multilingual_components
except ImportError:
    multilingual_components = None

try:
    from services.multilingual_conversation_interface import multilingual_interface
except ImportError:
    multilingual_interface = None
from utils.logger import get_logger
from services.language_detection import language_detection_service

logger = get_logger("streamlit_app")


# Multilingual text support
UI_TEXT = {
    "en": {
        "app_name": "🧠 Kurachi AI",
        "chat_page": "💬 Chat",
        "documents_page": "📄 Documents", 
        "analytics_page": "📊 Analytics",
        "navigation": "📋 Navigation",
        "session_info": "👤 Session Info",
        "quick_actions": "⚡ Quick Actions",
        "system_status": "🔧 System Status",
        "new_chat": "🔄 New Chat",
        "upload_documents": "📤 Upload Documents",
        "chat_with_documents": "💬 Chat with Your Documents",
        "select_conversation": "Select Conversation",
        "new_chat_btn": "🆕 New Chat",
        "delete_chat": "🗑️ Delete Chat",
        "ask_question": "Ask a question about your documents...",
        "no_conversations": "No conversations yet. Start a new one below!",
        "analyzing_question": "Analyzing your question...",
        "sources": "📚 Sources",
        "confidence": "Confidence",
        "document_management": "📄 Document Management",
        "upload_files": "📤 Upload Documents",
        "drag_drop_text": "Drag & Drop Files Here",
        "or_click_browse": "or click to browse",
        "supported_formats": "Supports: PDF, DOCX, XLSX, PPTX, Images, Audio, Video",
        "processing_files": "🔄 Processing Files",
        "analytics_dashboard": "📊 Analytics Dashboard",
        "knowledge_graph": "🕸️ Knowledge Graph",
        "language_detected": "Language detected",
        "bilingual_search": "🌐 Bilingual Search Active",
        "translation_quality": "Translation Quality"
    },
    "ja": {
        "app_name": "🧠 蔵地AI",
        "chat_page": "💬 チャット",
        "documents_page": "📄 文書",
        "analytics_page": "📊 分析", 
        "navigation": "📋 ナビゲーション",
        "session_info": "👤 セッション情報",
        "quick_actions": "⚡ クイックアクション",
        "system_status": "🔧 システム状態",
        "new_chat": "🔄 新しいチャット",
        "upload_documents": "📤 文書アップロード",
        "chat_with_documents": "💬 文書とチャット",
        "select_conversation": "会話を選択",
        "new_chat_btn": "🆕 新しいチャット",
        "delete_chat": "🗑️ チャット削除",
        "ask_question": "文書について質問してください...",
        "no_conversations": "まだ会話がありません。下記から新しい会話を始めてください！",
        "analyzing_question": "質問を分析中...",
        "sources": "📚 ソース",
        "confidence": "信頼度",
        "document_management": "📄 文書管理",
        "upload_files": "📤 文書アップロード",
        "drag_drop_text": "ファイルをここにドラッグ＆ドロップ",
        "or_click_browse": "またはクリックして参照",
        "supported_formats": "対応: PDF, DOCX, XLSX, PPTX, 画像, 音声, 動画",
        "processing_files": "🔄 ファイル処理中",
        "analytics_dashboard": "📊 分析ダッシュボード",
        "knowledge_graph": "🕸️ 知識グラフ",
        "language_detected": "検出言語",
        "bilingual_search": "🌐 バイリンガル検索有効",
        "translation_quality": "翻訳品質"
    }
}

def get_text(key: str, lang: str = "en") -> str:
    """Get localized text"""
    return UI_TEXT.get(lang, UI_TEXT["en"]).get(key, key)


class KurachiStreamlitApp:
    """Main Streamlit application class with SOTA enhancements"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        self.setup_sota_services()
    
    def setup_sota_services(self):
        """Initialize SOTA services with lazy loading"""
        with trace_operation("ui_service_initialization", component="streamlit_app"):
            # Get SOTA services through the registry
            self.chat_service = get_service('sota_chat')  # Use SOTA chat service
            self.translation_service = get_service('sota_translation')  # Use SOTA translation
            self.analytics_service = get_service('analytics')
            
            # Record UI initialization
            record_metric("ui_initialization", 1, MetricType.COUNTER, {"component": "streamlit"})
    
    def setup_page_config(self):
        """Configure Streamlit page settings with professional styling using SOTA config"""
        # Get SOTA configuration
        config = get_config()
        app_name = getattr(config, 'app_name', 'Kurachi AI') if hasattr(config, 'app_name') else 'Kurachi AI'
        
        st.set_page_config(
            page_title=app_name,
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Get current theme
        theme = getattr(st.session_state, 'theme', 'light')
        
        # Professional theme system with perfect color combinations
        theme_css = self.get_theme_css(theme)
        
        st.markdown(theme_css, unsafe_allow_html=True)
    
    def get_theme_css(self, theme='light'):
        """Get comprehensive CSS for the specified theme"""
        
        # Light mode colors (similar to ChatGPT/Gemini)
        light_colors = {
            # Primary colors
            'primary': '#10a37f',           # Professional green (ChatGPT-style)
            'primary_hover': '#0d8f72',
            'primary_light': '#e6f7f4',
            'accent': '#6366f1',            # Indigo accent
            'accent_hover': '#5b5bd6',
            
            # Background colors
            'bg_primary': '#ffffff',        # Pure white
            'bg_secondary': '#f9fafb',      # Very light gray
            'bg_tertiary': '#f3f4f6',       # Light gray
            'bg_elevated': '#ffffff',       # Elevated surfaces
            'bg_chat_user': '#10a37f',      # User message background
            'bg_chat_assistant': '#f7f7f8', # Assistant message background
            
            # Text colors
            'text_primary': '#0f172a',      # Almost black
            'text_secondary': '#64748b',    # Medium gray
            'text_tertiary': '#94a3b8',     # Light gray
            'text_inverse': '#ffffff',      # White text
            'text_success': '#059669',      # Success green
            'text_error': '#dc2626',        # Error red
            'text_warning': '#d97706',      # Warning orange
            
            # Border and divider colors
            'border_primary': '#e5e7eb',    # Light border
            'border_secondary': '#d1d5db',  # Medium border
            'border_focus': '#10a37f',      # Focus border (primary)
            'divider': '#f3f4f6',           # Subtle divider
            
            # Shadow colors
            'shadow_sm': '0 1px 2px 0 rgb(0 0 0 / 0.05)',
            'shadow_md': '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -1px rgb(0 0 0 / 0.06)',
            'shadow_lg': '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -2px rgb(0 0 0 / 0.05)',
        }
        
        # Dark mode colors (similar to modern dark themes)
        dark_colors = {
            # Primary colors
            'primary': '#00d4aa',           # Bright teal (better visibility in dark)
            'primary_hover': '#00c49a',
            'primary_light': '#0f2e2a',
            'accent': '#818cf8',            # Light indigo accent
            'accent_hover': '#6366f1',
            
            # Background colors
            'bg_primary': '#1f2937',        # Dark gray (main background)
            'bg_secondary': '#111827',      # Darker gray (secondary background)
            'bg_tertiary': '#0f172a',       # Almost black
            'bg_elevated': '#374151',       # Elevated surfaces
            'bg_chat_user': '#00d4aa',      # User message background
            'bg_chat_assistant': '#374151', # Assistant message background
            
            # Text colors
            'text_primary': '#f9fafb',      # Almost white
            'text_secondary': '#d1d5db',    # Light gray
            'text_tertiary': '#9ca3af',     # Medium gray
            'text_inverse': '#1f2937',      # Dark text
            'text_success': '#10b981',      # Success green
            'text_error': '#ef4444',        # Error red
            'text_warning': '#f59e0b',      # Warning orange
            
            # Border and divider colors
            'border_primary': '#374151',    # Dark border
            'border_secondary': '#4b5563',  # Medium dark border
            'border_focus': '#00d4aa',      # Focus border (primary)
            'divider': '#374151',           # Subtle divider
            
            # Shadow colors (more subtle in dark mode)
            'shadow_sm': '0 1px 2px 0 rgb(0 0 0 / 0.3)',
            'shadow_md': '0 4px 6px -1px rgb(0 0 0 / 0.4), 0 2px 4px -1px rgb(0 0 0 / 0.3)',
            'shadow_lg': '0 10px 15px -3px rgb(0 0 0 / 0.5), 0 4px 6px -2px rgb(0 0 0 / 0.4)',
        }
        
        # Select colors based on theme
        colors = dark_colors if theme == 'dark' else light_colors
        
        return f"""
        <style>
        /* Import modern fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        /* Global theme variables */
        :root {{
            /* Primary colors */
            --primary-color: {colors['primary']};
            --primary-hover: {colors['primary_hover']};
            --primary-light: {colors['primary_light']};
            --accent-color: {colors['accent']};
            --accent-hover: {colors['accent_hover']};
            
            /* Background colors */
            --bg-primary: {colors['bg_primary']};
            --bg-secondary: {colors['bg_secondary']};
            --bg-tertiary: {colors['bg_tertiary']};
            --bg-elevated: {colors['bg_elevated']};
            --bg-chat-user: {colors['bg_chat_user']};
            --bg-chat-assistant: {colors['bg_chat_assistant']};
            
            /* Text colors */
            --text-primary: {colors['text_primary']};
            --text-secondary: {colors['text_secondary']};
            --text-tertiary: {colors['text_tertiary']};
            --text-inverse: {colors['text_inverse']};
            --text-success: {colors['text_success']};
            --text-error: {colors['text_error']};
            --text-warning: {colors['text_warning']};
            
            /* Border colors */
            --border-primary: {colors['border_primary']};
            --border-secondary: {colors['border_secondary']};
            --border-focus: {colors['border_focus']};
            --divider: {colors['divider']};
            
            /* Shadows */
            --shadow-sm: {colors['shadow_sm']};
            --shadow-md: {colors['shadow_md']};
            --shadow-lg: {colors['shadow_lg']};
            
            /* Layout */
            --border-radius: 12px;
            --border-radius-sm: 8px;
            --border-radius-lg: 16px;
            --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            --transition-fast: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        /* Global styles */
        .stApp {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--bg-secondary);
            color: var(--text-primary);
            transition: var(--transition);
        }}
        
        /* Main container */
        .main .block-container {{
            background: var(--bg-primary);
            border-radius: var(--border-radius-lg);
            border: 1px solid var(--border-primary);
            box-shadow: var(--shadow-sm);
            margin-top: 1rem;
        }}
        
        /* Sidebar styling */
        .css-1d391kg, .css-1cypcdb {{
            background: var(--bg-elevated) !important;
            border-right: 1px solid var(--border-primary) !important;
        }}
        
        .css-1d391kg .element-container, .css-1cypcdb .element-container {{
            background: transparent;
        }}
        
        /* Chat message containers */
        .stChatMessage {{
            background: transparent !important;
            border: none !important;
            margin-bottom: 1rem;
            border-radius: var(--border-radius);
        }}
        
        /* User messages (right-aligned) */
        .stChatMessage[data-testid="user"] {{
            background: var(--bg-chat-user) !important;
        }}
        
        .stChatMessage[data-testid="user"] .stChatMessageContent {{
            background: var(--bg-chat-user);
            color: var(--text-inverse);
            border-radius: var(--border-radius);
            padding: 1rem;
            margin-left: 2rem;
            box-shadow: var(--shadow-sm);
        }}
        
        /* Assistant messages (left-aligned) */
        .stChatMessage[data-testid="assistant"] .stChatMessageContent {{
            background: var(--bg-chat-assistant);
            color: var(--text-primary);
            border-radius: var(--border-radius);
            padding: 1rem;
            margin-right: 2rem;
            border: 1px solid var(--border-primary);
            box-shadow: var(--shadow-sm);
        }}
        
        /* Chat input */
        .stChatInputContainer {{
            background: var(--bg-elevated);
            border: 1px solid var(--border-primary);
            border-radius: var(--border-radius-lg);
            padding: 0.75rem;
        }}
        
        /* Text inputs */
        .stTextInput > div > div > input, 
        .stTextArea > div > div > textarea {{
            background: var(--bg-primary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-primary) !important;
            border-radius: var(--border-radius) !important;
            transition: var(--transition);
        }}
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {{
            border-color: var(--border-focus) !important;
            box-shadow: 0 0 0 3px {colors['primary']}20 !important;
        }}
        
        /* Buttons */
        .stButton > button {{
            background: var(--primary-color);
            color: var(--text-inverse);
            border: none;
            border-radius: var(--border-radius);
            padding: 0.6rem 1.2rem;
            font-weight: 500;
            font-size: 0.875rem;
            transition: var(--transition-fast);
            box-shadow: var(--shadow-sm);
        }}
        
        .stButton > button:hover {{
            background: var(--primary-hover);
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
        }}
        
        .stButton > button:active {{
            transform: translateY(0);
        }}
        
        /* Secondary buttons */
        .stButton.secondary > button {{
            background: var(--bg-elevated);
            color: var(--text-primary);
            border: 1px solid var(--border-primary);
        }}
        
        .stButton.secondary > button:hover {{
            background: var(--bg-tertiary);
            border-color: var(--border-secondary);
        }}
        
        /* Selectbox */
        .stSelectbox > div > div > div {{
            background: var(--bg-primary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-primary) !important;
            border-radius: var(--border-radius) !important;
        }}
        
        /* Metrics */
        .stMetric {{
            background: var(--bg-elevated);
            border: 1px solid var(--border-primary);
            border-radius: var(--border-radius);
            padding: 1rem;
            text-align: center;
        }}
        
        .stMetric .metric-container {{
            background: transparent;
        }}
        
        .stMetric [data-testid="metric-container"] > div {{
            color: var(--text-primary);
        }}
        
        .stMetric [data-testid="metric-container"] > div:first-child {{
            color: var(--text-secondary);
            font-size: 0.875rem;
        }}
        
        /* Progress bars */
        .stProgress > div > div > div > div {{
            background: linear-gradient(90deg, var(--primary-color), var(--primary-hover));
        }}
        
        /* Expanders */
        .streamlit-expanderHeader {{
            background: var(--bg-elevated) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-primary) !important;
            border-radius: var(--border-radius) !important;
        }}
        
        .streamlit-expanderContent {{
            background: var(--bg-primary) !important;
            border: 1px solid var(--border-primary) !important;
            border-top: none !important;
            border-radius: 0 0 var(--border-radius) var(--border-radius) !important;
        }}
        
        /* File uploader */
        .stFileUploader {{
            border: 2px dashed var(--border-primary);
            border-radius: var(--border-radius);
            background: var(--bg-secondary);
            transition: var(--transition);
        }}
        
        .stFileUploader:hover {{
            border-color: var(--primary-color);
            background: var(--primary-light);
        }}
        
        /* Source citations */
        .source-citation {{
            background: var(--bg-elevated);
            border: 1px solid var(--border-primary);
            border-radius: var(--border-radius-sm);
            padding: 1rem;
            margin: 0.75rem 0;
            font-size: 0.875rem;
            transition: var(--transition);
        }}
        
        .source-citation:hover {{
            box-shadow: var(--shadow-md);
            border-color: var(--border-secondary);
        }}
        
        .source-filename {{
            font-weight: 600;
            color: var(--primary-color);
            margin-bottom: 0.5rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
        }}
        
        .source-content {{
            color: var(--text-secondary);
            line-height: 1.5;
        }}
        
        /* Loading animations */
        .thinking-animation {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 1rem;
            background: var(--bg-chat-assistant);
            border-radius: var(--border-radius);
            border: 1px solid var(--border-primary);
            margin: 1rem 0;
        }}
        
        .thinking-dots {{
            display: flex;
            gap: 6px;
        }}
        
        .thinking-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--primary-color);
            animation: thinking 1.4s ease-in-out infinite both;
        }}
        
        .thinking-dot:nth-child(1) {{ animation-delay: -0.32s; }}
        .thinking-dot:nth-child(2) {{ animation-delay: -0.16s; }}
        
        @keyframes thinking {{
            0%, 80%, 100% {{
                transform: scale(0.8);
                opacity: 0.5;
            }}
            40% {{
                transform: scale(1.2);
                opacity: 1;
            }}
        }}
        
        /* Tables */
        .stDataFrame, .stTable {{
            background: var(--bg-primary);
            border: 1px solid var(--border-primary);
            border-radius: var(--border-radius);
        }}
        
        .stDataFrame thead th, .stTable thead th {{
            background: var(--bg-elevated) !important;
            color: var(--text-primary) !important;
            border-color: var(--border-primary) !important;
        }}
        
        /* Alerts */
        .stAlert {{
            border-radius: var(--border-radius);
            border: 1px solid var(--border-primary);
        }}
        
        .stAlert[data-baseweb="notification"][kind="info"] {{
            background: var(--primary-light);
            border-color: var(--primary-color);
        }}
        
        /* Radio buttons */
        .stRadio > div {{
            background: var(--bg-elevated);
            border-radius: var(--border-radius);
            padding: 0.5rem;
        }}
        
        /* Theme toggle buttons */
        .theme-toggle {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}
        
        .theme-button {{
            flex: 1;
            padding: 0.5rem;
            border: 1px solid var(--border-primary);
            border-radius: var(--border-radius-sm);
            background: var(--bg-elevated);
            color: var(--text-primary);
            cursor: pointer;
            transition: var(--transition-fast);
            text-align: center;
        }}
        
        .theme-button:hover {{
            background: var(--bg-tertiary);
            border-color: var(--primary-color);
        }}
        
        .theme-button.active {{
            background: var(--primary-color);
            color: var(--text-inverse);
            border-color: var(--primary-color);
        }}
        
        /* Responsive design */
        @media (max-width: 768px) {{
            .main .block-container {{
                margin-top: 0.5rem;
                border-radius: var(--border-radius);
            }}
            
            .stChatMessage[data-testid="user"] .stChatMessageContent {{
                margin-left: 1rem;
            }}
            
            .stChatMessage[data-testid="assistant"] .stChatMessageContent {{
                margin-right: 1rem;
            }}
        }}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: var(--bg-secondary);
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: var(--border-secondary);
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: var(--text-tertiary);
        }}
        </style>
        """
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if "user_id" not in st.session_state:
            st.session_state.user_id = str(uuid.uuid4())  # Simple user ID for demo
        
        if "current_conversation_id" not in st.session_state:
            st.session_state.current_conversation_id = None
        
        if "language" not in st.session_state:
            st.session_state.language = "en"
        
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []
        
        # Initialize theme with system preference detection
        if "theme" not in st.session_state:
            # Default to light mode for professional appearance
            st.session_state.theme = "light"
        
        if "show_upload" not in st.session_state:
            st.session_state.show_upload = False
        
        # Theme preferences
        if "theme_auto_switch" not in st.session_state:
            st.session_state.theme_auto_switch = False
    
    def get_theme_indicator(self, theme):
        """Get visual indicator for current theme"""
        if theme == "dark":
            return "🌙", "Dark Mode Active", "Perfect for low-light environments"
        else:
            return "☀️", "Light Mode Active", "Optimized for professional use"
    
    def render_sidebar(self):
        """Render the sidebar with navigation, settings, and enhanced theme toggle"""
        with st.sidebar:
            lang = st.session_state.language
            st.title(get_text("app_name", lang))
            st.caption(f"Version {config.app.version}")
            
            # Language preferences panel
            st.markdown("### 🌐 Language Settings")
            
            # Quick language selector
            current_lang, lang_changed = multilingual_components.render_language_selector(
                st.session_state.user_id, "sidebar_"
            )
            
            if lang_changed:
                st.rerun()
            
            # Language status indicator
            multilingual_components.render_language_status_indicator(st.session_state.user_id)
            
            # Full preferences in expander
            with st.expander("⚙️ Advanced Language Settings"):
                prefs_changed = multilingual_components.render_language_preferences_panel(
                    st.session_state.user_id
                )
                if prefs_changed:
                    st.rerun()
            
            st.divider()
            
            # Enhanced theme toggle with visual feedback
            st.markdown("### 🎨 Theme")
            
            current_theme = st.session_state.theme
            theme_icon, theme_name, theme_desc = self.get_theme_indicator(current_theme)
            
            # Current theme status
            st.markdown(f"""
            <div style="
                background: var(--bg-elevated);
                border: 1px solid var(--border-primary);
                border-radius: var(--border-radius-sm);
                padding: 0.75rem;
                margin-bottom: 1rem;
                text-align: center;
            ">
                <div style="font-size: 1.2rem; margin-bottom: 0.25rem;">{theme_icon}</div>
                <div style="font-weight: 500; color: var(--text-primary);">{theme_name}</div>
                <div style="font-size: 0.75rem; color: var(--text-secondary);">{theme_desc}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Theme toggle buttons
            theme_col1, theme_col2 = st.columns([1, 1])
            
            with theme_col1:
                dark_selected = current_theme == "dark"
                dark_button_style = "background: var(--primary-color); color: var(--text-inverse);" if dark_selected else ""
                if st.button("🌙 Dark", key="dark_theme", use_container_width=True, help="Switch to dark mode"):
                    if current_theme != "dark":
                        st.session_state.theme = "dark"
                        st.rerun()
            
            with theme_col2:
                light_selected = current_theme == "light"
                light_button_style = "background: var(--primary-color); color: var(--text-inverse);" if light_selected else ""
                if st.button("☀️ Light", key="light_theme", use_container_width=True, help="Switch to light mode"):
                    if current_theme != "light":
                        st.session_state.theme = "light"
                        st.rerun()
            
            # Theme features info
            with st.expander("🎨 Theme Features"):
                st.write("**Dark Mode Benefits:**")
                st.write("• Reduced eye strain in low light")
                st.write("• Better battery life on OLED screens")
                st.write("• Professional dark interface")
                st.write("")
                st.write("**Light Mode Benefits:**")
                st.write("• Optimal for bright environments")
                st.write("• Enhanced readability")
                st.write("• Professional business appearance")
            st.divider()
            st.markdown("### 🌐 Language / 言語")
            language_options = {"English": "en", "日本語": "ja"}
            selected_language = st.selectbox(
                "Choose Language",
                options=list(language_options.keys()),
                index=0 if st.session_state.language == "en" else 1,
                label_visibility="collapsed"
            )
            
            # Update language and show current status
            if language_options[selected_language] != st.session_state.language:
                st.session_state.language = language_options[selected_language]
                st.rerun()
            
            # Show translation capabilities
            supported_langs = self.chat_service.get_supported_languages()
            with st.expander("🔧 Translation Settings"):
                st.write("**Supported Languages:**")
                for lang_info in supported_langs:
                    st.write(f"• {lang_info['native_name']} ({lang_info['name']})")
                st.write("\n**Features:**")
                st.write("• Automatic language detection")
                st.write("• Cross-language document search")
                st.write("• Real-time translation")
            
            st.divider()
            
            # Navigation with professional styling
            st.markdown(f"### {get_text('navigation', lang)}")
            page_options = [
                get_text("chat_page", lang),
                get_text("documents_page", lang), 
                get_text("analytics_page", lang),
                get_text("knowledge_graph", lang)
            ]
            page = st.radio(
                "Choose a page:",
                page_options,
                index=0,
                label_visibility="collapsed"
            )
            
            st.divider()
            
            # Enhanced user info section
            st.markdown(f"### {get_text('session_info', lang)}")
            with st.container():
                st.markdown(f"""
                <div class="metric-container">
                    <div style="font-size: 0.9em; color: var(--text-secondary);">User ID</div>
                    <div style="font-family: monospace; font-size: 0.8em;">{st.session_state.user_id[:12]}...</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Enhanced document stats
            doc_stats = document_service.get_document_stats(st.session_state.user_id)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", doc_stats["total_documents"], delta=None)
            with col2:
                if doc_stats["total_size"] > 0:
                    size_mb = doc_stats["total_size"] / (1024 * 1024)
                    st.metric("Size (MB)", f"{size_mb:.1f}")
                else:
                    st.metric("Size (MB)", "0")
            
            # Processing status overview
            if doc_stats["by_status"]:
                completed = doc_stats["by_status"].get("completed", 0)
                total = doc_stats["total_documents"]
                if total > 0:
                    success_rate = (completed / total) * 100
                    st.metric("Success Rate", f"{success_rate:.0f}%")
            
            st.divider()
            
            # Quick actions
            st.markdown(f"### {get_text('quick_actions', lang)}")
            if st.button(get_text("new_chat", lang), use_container_width=True):
                conversation = self.chat_service.create_conversation(
                    st.session_state.user_id,
                    f"Chat {datetime.now().strftime('%m/%d %H:%M')}"
                )
                if conversation:
                    st.session_state.current_conversation_id = conversation.id
                    st.rerun()
            
            if st.button(get_text("upload_documents", lang), use_container_width=True):
                st.session_state.show_upload = True
                st.rerun()
            
            # System status
            st.divider()
            st.markdown(f"### {get_text('system_status', lang)}")
            st.success("🟢 Ollama Connected")
            st.info(f"🧠 Model: {config.ai.llm_model}")
            st.info(f"👁️ Vision: {config.ai.vision_model}")
            
            # Show bilingual capabilities
            st.info(f"🌐 {get_text('bilingual_search', lang)}")
            
            return page
    
    def render_chat_page(self):
        """Render the chat interface"""
        lang = st.session_state.language
        st.header(get_text("chat_with_documents", lang))
        
        # Conversation management
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Get user conversations
            conversations = self.chat_service.get_user_conversations(st.session_state.user_id)
            
            if conversations:
                conversation_options = {}
                for conv in conversations:
                    # Get context info for each conversation
                    context_info = self.chat_service.get_conversation_context_info(conv.id)
                    msg_count = context_info.get("message_count", 0)
                    last_activity = context_info.get("last_activity")
                    
                    if last_activity:
                        time_str = last_activity.strftime('%m/%d %H:%M')
                    else:
                        time_str = conv.created_at.strftime('%m/%d %H:%M')
                    
                    display_name = f"{conv.title} ({msg_count} msgs, {time_str})"
                    conversation_options[display_name] = conv.id
                
                selected_conv = st.selectbox(
                    get_text("select_conversation", lang),
                    options=list(conversation_options.keys()),
                    index=0
                )
                st.session_state.current_conversation_id = conversation_options[selected_conv]
                
                # Show conversation context info with language detection
                if st.session_state.current_conversation_id:
                    context = self.chat_service.get_conversation_context_info(st.session_state.current_conversation_id)
                    if context["message_count"] > 0:
                        # Get recent messages to detect conversation language
                        recent_messages = self.chat_service.get_conversation_history(
                            st.session_state.current_conversation_id, 
                            st.session_state.user_id
                        )
                        if recent_messages:
                            last_user_msg = next((msg for msg in reversed(recent_messages) if msg.role == "user"), None)
                            if last_user_msg:
                                detected_lang = language_detection_service.detect_language(last_user_msg.content)
                                # Map simple codes to flags for display
                                if detected_lang == 'ja':
                                    lang_display = "🇯🇵 Japanese"
                                elif detected_lang == 'en':
                                    lang_display = "🇺🇸 English"
                                elif detected_lang.startswith('zh'):
                                    lang_display = "🇨🇳 Chinese"
                                elif detected_lang == 'ko':
                                    lang_display = "🇰🇷 Korean"
                                else:
                                    lang_display = detected_lang
                                st.caption(f"💬 {context['user_messages']} questions • 🤖 {context['assistant_messages']} responses • 📝 {context['total_characters']:,} chars • {get_text('language_detected', lang)}: {lang_display}")
                            else:
                                st.caption(f"💬 {context['user_messages']} questions • 🤖 {context['assistant_messages']} responses • 📝 {context['total_characters']:,} chars")
            else:
                st.info(get_text("no_conversations", lang))
                st.session_state.current_conversation_id = None
        
        with col2:
            if st.button(get_text("new_chat_btn", lang)):
                conversation = self.chat_service.create_conversation(
                    st.session_state.user_id,
                    f"Chat {datetime.now().strftime('%m/%d %H:%M')}"
                )
                if conversation:
                    st.session_state.current_conversation_id = conversation.id
                    st.rerun()
        
        with col3:
            if st.button(get_text("delete_chat", lang)) and st.session_state.current_conversation_id:
                if self.chat_service.delete_conversation(
                    st.session_state.current_conversation_id,
                    st.session_state.user_id
                ):
                    st.session_state.current_conversation_id = None
                    st.rerun()
        
        st.divider()
        
        # Chat interface
        if not st.session_state.current_conversation_id:
            # Create first conversation
            conversation = self.chat_service.create_conversation(
                st.session_state.user_id,
                f"Chat {datetime.now().strftime('%m/%d %H:%M')}"
            )
            if conversation:
                st.session_state.current_conversation_id = conversation.id
                st.rerun()
            else:
                st.error("Failed to create conversation")
                return
        
        # Display chat history
        messages = self.chat_service.get_conversation_history(
            st.session_state.current_conversation_id,
            st.session_state.user_id
        )
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            for message in messages:
                with st.chat_message(message.role):
                    st.markdown(message.content)
                    
                    # Show intelligent formatting components for assistant messages
                    if message.role == "assistant" and message.metadata:
                        # Render intelligent formatting components
                        intelligent_formatting = message.metadata.get("intelligent_formatting", {})
                        if intelligent_formatting:
                            # Show response type indicator
                            response_format_type = intelligent_formatting.get("response_format_type", "plain_text")
                            formatting_metadata = intelligent_formatting.get("formatting_metadata", {})
                            confidence = formatting_metadata.get("intent_confidence", 0.0)
                            
                            streamlit_component_renderer.render_response_type_indicator(
                                response_format_type, confidence
                            )
                            
                            # Render formatted components
                            streamlit_component_renderer.render_formatted_response(message.metadata)
                        
                        # Show sources for assistant messages
                        sources = message.metadata.get("sources", [])
                        if sources:
                            with st.expander(f"📚 Sources ({len(sources)})"):
                                for i, source in enumerate(sources):
                                    st.text(f"**{source.get('filename', 'Unknown')}**")
                                    st.text(source.get('content', '')[:200] + "...")
                                    if i < len(sources) - 1:
                                        st.divider()
        
        # Chat input
        if prompt := st.chat_input(get_text("ask_question", lang)):
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Professional streaming response
            with st.chat_message("assistant"):
                self._render_streaming_response(prompt)
            
            st.rerun()
    
    def _render_streaming_response(self, prompt: str):
        """Render instant response without thinking delays for small LLM"""
        # Create containers for response parts
        response_container = st.empty()
        sources_container = st.empty()
        
        try:
            # Get instant response without artificial delays
            accumulated_response = ""
            sources_data = None
            
            for chunk in self.chat_service.stream_response(
                st.session_state.current_conversation_id,
                prompt,
                st.session_state.user_id
            ):
                if chunk.get("type") == "response_chunk":
                    data = chunk["data"]
                    accumulated_response = data["accumulated_content"]
                    
                    # Instant response update without delays
                    response_container.markdown(accumulated_response)
                    
                elif chunk.get("type") == "sources":
                    sources_data = chunk["data"]
                
                elif chunk.get("type") == "error":
                    st.error(f"Error: {chunk['error']}")
                    return
            
            # Show final response
            response_container.markdown(accumulated_response)
            
            # Display intelligent formatting components if available
            if sources_data and sources_data.get("intelligent_formatting"):
                intelligent_formatting = sources_data["intelligent_formatting"]
                response_format_type = intelligent_formatting.get("response_format_type", "plain_text")
                formatting_metadata = intelligent_formatting.get("formatting_metadata", {})
                confidence = formatting_metadata.get("intent_confidence", 0.0)
                
                # Show response type indicator
                streamlit_component_renderer.render_response_type_indicator(
                    response_format_type, confidence
                )
                
                # Render formatted components
                streamlit_component_renderer.render_formatted_response(sources_data)
            
            # Display enhanced source citations
            if sources_data and sources_data["sources"]:
                self._render_enhanced_sources(sources_data, sources_container)
                
        except Exception as e:
            st.error(f"Failed to generate response: {str(e)}")
    
    def _render_enhanced_sources(self, sources_data: dict, container):
        """Render professional source citations with confidence scores and language info"""
        sources = sources_data["sources"]
        confidence = sources_data.get("confidence", 0.5)
        lang = st.session_state.language
        
        # Check if bilingual search was performed
        bilingual_search = sources_data.get("bilingual_search_performed", False)
        query_language = sources_data.get("query_language", "unknown")
        
        header_text = f"{get_text('sources', lang)} ({len(sources)}) • {get_text('confidence', lang)}: {confidence:.0%}"
        if bilingual_search:
            header_text += f" • {get_text('bilingual_search', lang)}"
        
        with container.expander(header_text, expanded=False):
            # Show search language information
            if query_language != "unknown":
                lang_display = "🇯🇵 Japanese" if query_language == "ja" else "🇺🇸 English"
                st.info(f"{get_text('language_detected', lang)}: {lang_display}")
                
            for i, source in enumerate(sources):
                # Enhanced source display with language info
                doc_language = source.get('detected_language', 'unknown')
                lang_icon = "🇯🇵" if doc_language == "ja" else "🇺🇸" if doc_language == "en" else "🌐"
                
                st.markdown(f"""
                <div class="source-citation">
                    <div class="source-filename">
                        📄 {source.get('filename', 'Unknown')} {lang_icon}
                        {f"(Page {source.get('page_number', 'N/A')})" if source.get('page_number') != 'N/A' else ""}
                        <span style="float: right; color: #64748b; font-size: 0.8em;">
                            Relevance: {source.get('relevance_score', 0):.0%}
                        </span>
                    </div>
                    <div class="source-content">
                        {source.get('content', '')[:250]}{'...' if len(source.get('content', '')) > 250 else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if i < len(sources) - 1:
                    st.markdown("<br>", unsafe_allow_html=True)

    def render_documents_page(self):
        """Render the modern document management interface with drag & drop"""
        lang = st.session_state.language
        st.header(get_text("document_management", lang))
        
        # Modern file upload section with drag & drop
        st.markdown(f"### {get_text('upload_files', lang)}")
        
        # Create drag & drop zone with custom styling
        st.markdown("""
        <div class="upload-zone">
            <div class="upload-content">
                <div class="upload-icon">📁</div>
                <div class="upload-text">
                    <h3>Drag & Drop Files Here</h3>
                    <p>or click to browse</p>
                </div>
                <div class="upload-formats">
                    Supports: PDF, DOCX, XLSX, PPTX, Images, Audio, Video
                </div>
            </div>
        </div>
        
        <style>
        .upload-zone {
            border: 2px dashed var(--primary-color);
            border-radius: var(--border-radius);
            padding: 2rem;
            text-align: center;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            transition: var(--transition);
            cursor: pointer;
            margin: 1rem 0;
        }
        
        .upload-zone:hover {
            border-color: var(--primary-hover);
            background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }
        
        .upload-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1rem;
        }
        
        .upload-icon {
            font-size: 3rem;
            opacity: 0.7;
        }
        
        .upload-text h3 {
            margin: 0;
            color: var(--text-primary);
            font-weight: 600;
        }
        
        .upload-text p {
            margin: 0;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
        
        .upload-formats {
            color: var(--text-secondary);
            font-size: 0.8rem;
            background: rgba(99, 102, 241, 0.1);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            border: 1px solid rgba(99, 102, 241, 0.2);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # File uploader with enhanced functionality
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=None,
            help=f"Maximum file size: {config.app.max_file_size_mb}MB per file",
            label_visibility="collapsed"
        )
        
        # Enhanced upload processing with better UI
        if uploaded_files:
            self._render_upload_preview(uploaded_files)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                if st.button("🚀 Process Files", use_container_width=True, type="primary"):
                    self._process_uploaded_files(uploaded_files)
        
        st.divider()
        
        # Enhanced document library
        self._render_document_library()
    
    def _render_upload_preview(self, uploaded_files):
        """Render preview of files to be uploaded"""
        st.markdown("### 🔎 File Preview")
        
        total_size = sum(file.size for file in uploaded_files)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Files Selected", len(uploaded_files))
        with col2:
            st.metric("Total Size", f"{total_size / (1024*1024):.1f} MB")
        with col3:
            valid_files = sum(1 for file in uploaded_files if document_service.is_supported_file_type(file.name))
            st.metric("Valid Files", f"{valid_files}/{len(uploaded_files)}")
        
        # File details with enhanced display
        st.markdown("#### 📁 File Details")
        for i, file in enumerate(uploaded_files):
            with st.expander(f"📄 {file.name} ({file.size / 1024:.1f} KB)", expanded=i < 3):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    file_type = file.name.split('.')[-1].upper() if '.' in file.name else 'Unknown'
                    st.text(f"Type: {file_type}")
                    st.text(f"Size: {file.size:,} bytes")
                    
                    # File validation
                    if document_service.is_supported_file_type(file.name):
                        st.success("✓ Supported format")
                    else:
                        st.error("✗ Unsupported format")
                    
                    if file.size > config.app.max_file_size_mb * 1024 * 1024:
                        st.error(f"✗ File too large (max: {config.app.max_file_size_mb}MB)")
                
                with col2:
                    # File type icon
                    icon_map = {
                        'pdf': '📄', 'docx': '📄', 'xlsx': '📈', 
                        'pptx': '📊', 'jpg': '🇿️', 'png': '🇿️',
                        'mp3': '🎧', 'mp4': '🎥'
                    }
                    file_ext = file.name.split('.')[-1].lower() if '.' in file.name else ''
                    icon = icon_map.get(file_ext, '📁')
                    st.markdown(f"<div style='font-size: 3rem; text-align: center;'>{icon}</div>", unsafe_allow_html=True)
    
    def _process_uploaded_files(self, uploaded_files):
        """Process uploaded files with enhanced progress tracking"""
        # Filter valid files
        valid_files = [
            file for file in uploaded_files 
            if document_service.is_supported_file_type(file.name) and 
               file.size <= config.app.max_file_size_mb * 1024 * 1024
        ]
        
        if not valid_files:
            st.error("❌ No valid files to process")
            return
        
        # Enhanced progress tracking
        progress_container = st.container()
        status_container = st.container()
        
        with progress_container:
            st.markdown("### 🔄 Processing Files")
            overall_progress = st.progress(0)
            current_file_text = st.empty()
        
        results = {
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        for i, uploaded_file in enumerate(valid_files):
            current_file_text.text(f"📦 Uploading: {uploaded_file.name}")
            
            try:
                # Save file
                document = document_service.save_uploaded_file(
                    uploaded_file,
                    st.session_state.user_id
                )
                
                if document:
                    current_file_text.text(f"⚙️ Processing: {uploaded_file.name}")
                    
                    # Process document with progress
                    if self._process_document_with_ui_feedback(document, status_container):
                        results['successful'] += 1
                        current_file_text.text(f"✅ Completed: {uploaded_file.name}")
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"Processing failed: {uploaded_file.name}")
                        current_file_text.text(f"❌ Failed: {uploaded_file.name}")
                else:
                    results['failed'] += 1
                    results['errors'].append(f"Upload failed: {uploaded_file.name}")
                    current_file_text.text(f"❌ Upload failed: {uploaded_file.name}")
                
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"Error with {uploaded_file.name}: {str(e)}")
                current_file_text.text(f"❌ Error: {uploaded_file.name}")
            
            # Update progress
            progress = (i + 1) / len(valid_files)
            overall_progress.progress(progress)
            time.sleep(0.5)  # Brief pause for visual feedback
        
        # Show final results
        current_file_text.empty()
        overall_progress.progress(1.0)
        
        self._show_processing_results(results, status_container)
    
    def _process_document_with_ui_feedback(self, document, container):
        """Process document with UI feedback"""
        try:
            with container:
                with st.expander(f"🔍 Processing: {document.original_filename}", expanded=True):
                    steps = [
                        "📝 Extracting content...",
                        "🧠 Analyzing structure...", 
                        "✂️ Chunking text...",
                        "🌐 Creating embeddings...",
                        "💾 Storing in database..."
                    ]
                    
                    step_progress = st.progress(0)
                    step_text = st.empty()
                    
                    for i, step in enumerate(steps):
                        step_text.text(step)
                        step_progress.progress((i + 1) / len(steps))
                        time.sleep(0.3)  # Simulate processing time
                    
                    # Actually process the document
                    success = document_service.process_document(document.id)
                    
                    if success:
                        step_text.text("✅ Processing complete!")
                        return True
                    else:
                        step_text.text("❌ Processing failed!")
                        return False
                        
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            return False
    
    def _show_processing_results(self, results, container):
        """Show final processing results"""
        with container:
            st.markdown("### 🏁 Processing Complete")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if results['successful'] > 0:
                    st.success(f"✅ Successfully processed: {results['successful']} files")
                    st.balloons()
            
            with col2:
                if results['failed'] > 0:
                    st.error(f"❌ Failed to process: {results['failed']} files")
            
            # Show errors if any
            if results['errors']:
                with st.expander("⚠️ View Errors"):
                    for error in results['errors']:
                        st.text(f"• {error}")
            
            # Refresh recommendation
            if results['successful'] > 0:
                st.info("🔄 Page will refresh to show new documents...")
                time.sleep(2)
                st.rerun()
    
    def _render_document_library(self):
        """Render the document library section"""
        st.divider()
        
        # Document list
        st.subheader("Your Documents")
        
        documents = document_service.get_user_documents(st.session_state.user_id)
        
        if not documents:
            st.info("No documents uploaded yet. Upload some documents to get started!")
            return
        
        # Document table with enhanced language information
        for doc in documents:
            with st.expander(f"📄 {doc.original_filename}"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.text(f"**File Type:** {doc.file_type}")
                    st.text(f"**Size:** {doc.file_size / 1024:.1f} KB")
                    st.text(f"**Uploaded:** {doc.created_at.strftime('%Y-%m-%d %H:%M')}")
                    
                    # Show detected language if available
                    if hasattr(doc, 'metadata') and doc.metadata:
                        detected_lang = doc.metadata.get('detected_language')
                        if detected_lang:
                            lang_display = "🇯🇵 Japanese" if detected_lang == "ja" else "🇺🇸 English" if detected_lang == "en" else f"🌐 {detected_lang}"
                            st.text(f"**Language:** {lang_display}")
                
                with col2:
                    status_color = {
                        "completed": "🟢",
                        "processing": "🟡",
                        "pending": "🟡",
                        "failed": "🔴"
                    }
                    st.text(f"**Status:** {status_color.get(doc.processing_status, '⚪')} {doc.processing_status.title()}")
                
                with col3:
                    if st.button(f"🗑️ Delete", key=f"delete_{doc.id}"):
                        if document_service.delete_document(doc.id, st.session_state.user_id):
                            st.success("Document deleted!")
                            st.rerun()
                        else:
                            st.error("Failed to delete document")
                
                if doc.error_message:
                    st.error(f"Error: {doc.error_message}")
    
    def render_analytics_page(self):
        """Render the comprehensive analytics dashboard"""
        lang = st.session_state.language
        st.header(get_text("analytics_dashboard", lang))
        
        # Time period selector
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            time_period = st.selectbox(
                "Analysis Period",
                options=[7, 14, 30, 90],
                format_func=lambda x: f"Last {x} days",
                index=2  # Default to 30 days
            )
        
        with col2:
            if st.button("🔄 Refresh Data"):
                st.rerun()
        
        with col3:
            show_user_only = st.checkbox("My Data Only", value=True)
        
        # Get analytics data
        user_id = st.session_state.user_id if show_user_only else None
        usage_stats = analytics_service.get_usage_statistics(user_id, days=time_period)
        doc_insights = analytics_service.get_document_insights(user_id)
        knowledge_gaps = analytics_service.get_knowledge_gaps(limit=10)
        
        st.divider()
        
        # Key Metrics Overview
        st.subheader("📊 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Queries", 
                usage_stats.get("total_queries", 0),
                help="Number of questions asked to the AI"
            )
        
        with col2:
            avg_confidence = usage_stats.get("query_statistics", {}).get("avg_confidence", 0)
            st.metric(
                "Avg Confidence", 
                f"{avg_confidence:.1%}",
                help="Average confidence score of AI responses"
            )
        
        with col3:
            avg_response_time = usage_stats.get("query_statistics", {}).get("avg_response_time", 0)
            st.metric(
                "Avg Response Time", 
                f"{avg_response_time:.0f}ms",
                help="Average time to generate responses"
            )
        
        with col4:
            bilingual_rate = usage_stats.get("bilingual_usage", {}).get("bilingual_usage_rate", 0)
            st.metric(
                "Bilingual Usage", 
                f"{bilingual_rate:.1%}",
                help="Percentage of queries using bilingual search"
            )
        
        st.divider()
        
        # Charts and Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Language Usage Chart
            st.subheader("🌐 Language Usage")
            lang_usage = usage_stats.get("language_usage", {})
            if lang_usage:
                # Convert language codes to friendly names
                friendly_names = {"en": "English", "ja": "Japanese", "unknown": "Unknown"}
                lang_data = {friendly_names.get(k, k): v for k, v in lang_usage.items()}
                st.bar_chart(lang_data)
            else:
                st.info("No language usage data available")
        
        with col2:
            # Action Breakdown Chart
            st.subheader("⚡ Activity Breakdown")
            action_breakdown = usage_stats.get("action_breakdown", {})
            if action_breakdown:
                st.bar_chart(action_breakdown)
            else:
                st.info("No activity data available")
        
        st.divider()
        
        # Document Analytics
        st.subheader("📄 Document Analytics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_docs = doc_insights.get("total_documents", 0)
            st.metric("Total Documents", total_docs)
        
        with col2:
            doc_effectiveness = doc_insights.get("avg_document_effectiveness", 0)
            st.metric("Document Effectiveness", f"{doc_effectiveness:.1%}")
        
        with col3:
            success_rate = doc_insights.get("processing_success_rate", {}).get("success_rate", 0)
            st.metric("Processing Success", f"{success_rate:.1%}")
        
        # Document Language Distribution
        doc_lang_dist = doc_insights.get("document_language_distribution", {})
        if doc_lang_dist:
            st.subheader("📚 Document Languages")
            friendly_doc_langs = {"en": "English", "ja": "Japanese", "unknown": "Unknown"}
            doc_lang_data = {friendly_doc_langs.get(k, k): v for k, v in doc_lang_dist.items()}
            st.bar_chart(doc_lang_data)
        
        st.divider()
        
        # Knowledge Gaps Analysis
        st.subheader("🔍 Knowledge Gaps Analysis")
        
        if knowledge_gaps:
            st.write("These topics show low confidence responses and may need attention:")
            
            for i, gap in enumerate(knowledge_gaps[:5], 1):
                with st.expander(f"Gap #{i}: {gap['query'][:60]}...", expanded=i <= 2):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Frequency", gap['frequency'])
                    with col2:
                        st.metric("Avg Confidence", f"{gap['avg_confidence']:.1%}")
                    with col3:
                        st.metric("Impact Score", f"{gap['impact_score']:.1f}")
                    
                    st.write("**Languages involved:**", ", ".join(gap['languages_involved']))
                    st.write("**Suggestions:**")
                    for suggestion in gap['suggestions']:
                        st.write(f"• {suggestion}")
        else:
            st.success("🎉 No significant knowledge gaps identified!")
        
        st.divider()
        
        # Performance Insights
        st.subheader("⚡ Performance Insights")
        
        perf_metrics = usage_stats.get("performance_metrics", {})
        if perf_metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Response Time P50", 
                    f"{perf_metrics.get('response_time_p50', 0):.0f}ms",
                    help="50th percentile response time"
                )
            
            with col2:
                st.metric(
                    "Response Time P95", 
                    f"{perf_metrics.get('response_time_p95', 0):.0f}ms",
                    help="95th percentile response time"
                )
            
            with col3:
                st.metric(
                    "Confidence P50", 
                    f"{perf_metrics.get('confidence_p50', 0):.1%}",
                    help="50th percentile confidence score"
                )
            
            with col4:
                st.metric(
                    "Confidence P95", 
                    f"{perf_metrics.get('confidence_p95', 0):.1%}",
                    help="95th percentile confidence score"
                )
        
        st.divider()
        
        # Insights Report
        if st.button("📋 Generate Detailed Insights Report"):
            with st.spinner("Generating comprehensive insights report..."):
                insights_report = analytics_service.generate_insights_report(user_id)
                
                if insights_report.get("error"):
                    st.error(f"Failed to generate report: {insights_report['error']}")
                else:
                    st.subheader("🎯 Key Insights")
                    
                    for insight in insights_report.get("key_insights", []):
                        st.info(f"💡 {insight}")
                    
                    st.subheader("📈 Recommendations")
                    
                    for rec in insights_report.get("recommendations", []):
                        st.success(f"✅ {rec}")
                    
                    # Download report as JSON
                    report_json = json.dumps(insights_report, indent=2, default=str)
                    st.download_button(
                        label="📥 Download Full Report (JSON)",
                        data=report_json,
                        file_name=f"kurachi_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    def render_knowledge_graph_page(self):
        """Render the knowledge graph visualization and interaction page"""
        lang = st.session_state.language
        st.header(get_text("knowledge_graph", lang))
        
        # Knowledge graph overview
        st.subheader("📊 Graph Overview")
        
        graph_stats = knowledge_graph._get_graph_statistics()
        
        if graph_stats.get("total_entities", 0) == 0:
            st.info("📚 No knowledge graph data available yet. Upload and process some documents to build the knowledge graph.")
            return
        
        # Graph statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Entities", 
                graph_stats.get("total_entities", 0),
                help="Number of entities (people, organizations, concepts) in the graph"
            )
        
        with col2:
            st.metric(
                "Total Relationships", 
                graph_stats.get("total_relationships", 0),
                help="Number of relationships between entities"
            )
        
        with col3:
            st.metric(
                "Graph Density", 
                f"{graph_stats.get('graph_density', 0):.3f}",
                help="How densely connected the graph is (0-1)"
            )
        
        with col4:
            st.metric(
                "Connected Components", 
                graph_stats.get("connected_components", 0),
                help="Number of separate connected groups in the graph"
            )
        
        st.divider()
        
        # Entity search and exploration
        st.subheader("🔍 Entity Search")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_query = st.text_input(
                "Search entities:",
                placeholder="Enter name, organization, or concept...",
                help="Search for entities by name or related terms"
            )
        
        with col2:
            max_results = st.selectbox(
                "Max Results",
                options=[10, 20, 50, 100],
                index=1
            )
        
        if search_query:
            search_results = knowledge_graph.search_entities(search_query, limit=max_results)
            
            if search_results:
                st.write(f"Found {len(search_results)} entities:")
                
                for i, entity in enumerate(search_results):
                    with st.expander(f"📎 {entity['text']} ({entity['label']})", expanded=i < 3):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Confidence", f"{entity['confidence']:.2f}")
                        with col2:
                            st.metric("Importance", f"{entity.get('importance_score', 0):.3f}")
                        with col3:
                            st.metric("Relevance", f"{entity.get('relevance_score', 0):.2f}")
                        
                        if entity.get('context'):
                            st.write("**Context:**")
                            st.text(entity['context'][:200] + "..." if len(entity['context']) > 200 else entity['context'])
                        
                        # Show entity neighbors
                        if st.button(f"🕸️ Explore Connections", key=f"explore_{entity['id']}"):
                            neighbors = knowledge_graph.get_entity_neighbors(entity['id'])
                            
                            if neighbors['entities']:
                                st.write(f"**Connected Entities ({neighbors['neighbor_count']}):**")
                                
                                for neighbor in neighbors['entities'][:5]:  # Show top 5
                                    st.write(f"• {neighbor['text']} ({neighbor['label']})")
                                
                                if len(neighbors['entities']) > 5:
                                    st.write(f"... and {len(neighbors['entities']) - 5} more")
                            else:
                                st.write("No connected entities found.")
            else:
                st.info("🔍 No entities found matching your search.")
        
        st.divider()
        
        # Entity type distribution
        st.subheader("📊 Entity Types")
        
        entity_types = graph_stats.get("entity_type_distribution", {})
        if entity_types:
            # Create a more readable display
            friendly_types = {
                "PERSON": "People",
                "ORG": "Organizations", 
                "COMPANY_JP": "Japanese Companies",
                "COMPANY_EN": "Companies",
                "DEPARTMENT_JP": "Japanese Departments",
                "DEPARTMENT_EN": "Departments",
                "PROJECT_JP": "Japanese Projects",
                "PROJECT_EN": "Projects",
                "GPE": "Locations",
                "DATE": "Dates",
                "MONEY": "Monetary Values",
                "EMAIL": "Email Addresses",
                "PHONE_JP": "Japanese Phone Numbers",
                "URL": "URLs"
            }
            
            display_types = {friendly_types.get(k, k): v for k, v in entity_types.items()}
            st.bar_chart(display_types)
        else:
            st.info("No entity type data available")
        
        # Relationship type distribution
        st.subheader("🔗 Relationship Types")
        
        rel_types = graph_stats.get("relationship_type_distribution", {})
        if rel_types:
            friendly_rels = {
                "ASSOCIATED_WITH": "Associated With",
                "WORKS_AT": "Works At",
                "MEMBER_OF": "Member Of",
                "MANAGES": "Manages",
                "PARTICIPATES_IN": "Participates In",
                "LOCATED_IN": "Located In",
                "CO_OCCURS": "Co-occurs",
                "RELATED_TO": "Related To",
                "MENTIONED_WITH": "Mentioned With"
            }
            
            display_rels = {friendly_rels.get(k, k): v for k, v in rel_types.items()}
            st.bar_chart(display_rels)
        else:
            st.info("No relationship data available")
        
        st.divider()
        
        # Export options
        st.subheader("📥 Export Knowledge Graph")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Export as JSON"):
                graph_data = knowledge_graph.export_graph("json")
                
                if not graph_data.get("error"):
                    graph_json = json.dumps(graph_data, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="📥 Download Knowledge Graph",
                        data=graph_json,
                        file_name=f"knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.error(f"Export failed: {graph_data['error']}")
        
        with col2:
            st.info("🕰️ More export formats coming soon...")
        
        with col3:
            if st.button("🔄 Refresh Graph Data"):
                st.rerun()
    
    def run(self):
        """Run the Streamlit application"""
        try:
            # Render sidebar and get selected page
            page = self.render_sidebar()
            lang = st.session_state.language
            
            # Render selected page based on localized names
            if page == get_text("chat_page", lang):
                self.render_chat_page()
            elif page == get_text("documents_page", lang):
                self.render_documents_page()
            elif page == get_text("analytics_page", lang):
                self.render_analytics_page()
            elif page == get_text("knowledge_graph", lang):
                self.render_knowledge_graph_page()
            
        except Exception as e:
            logger.error("Error in Streamlit app", error=e)
            st.error(f"An error occurred: {str(e)}")
            
            if config.app.debug:
                st.exception(e)


def main():
    """Main entry point for the Streamlit app"""
    app = KurachiStreamlitApp()
    app.run()


if __name__ == "__main__":
    main()