import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

// Chat API endpoints
export const chatAPI = {
  // Send a message and get AI response
  sendMessage: async (message, conversationId = null, options = {}) => {
    const payload = {
      message,
      conversation_id: conversationId,
      language: options.language || 'en',
      model: options.model || 'default',
      temperature: options.temperature || 0.7,
      max_tokens: options.maxTokens || 2048,
    };
    
    const response = await api.post('/api/chat/send', payload);
    return response.data;
  },

  // Get conversation history
  getConversations: async (userId = null) => {
    const params = userId ? { user_id: userId } : {};
    const response = await api.get('/api/chat/conversations', { params });
    return response.data;
  },

  // Get messages for a specific conversation
  getMessages: async (conversationId) => {
    const response = await api.get(`/api/chat/conversations/${conversationId}/messages`);
    return response.data;
  },

  // Create a new conversation
  createConversation: async (title = null, userId = null) => {
    const payload = { title, user_id: userId };
    const response = await api.post('/api/chat/conversations', payload);
    return response.data;
  },

  // Delete a conversation
  deleteConversation: async (conversationId) => {
    const response = await api.delete(`/api/chat/conversations/${conversationId}`);
    return response.data;
  },

  // Regenerate AI response
  regenerateResponse: async (messageId) => {
    const response = await api.post(`/api/chat/messages/${messageId}/regenerate`);
    return response.data;
  },
};

// Document API endpoints
export const documentAPI = {
  // Upload a document
  uploadDocument: async (file, options = {}) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('language', options.language || 'en');
    formData.append('enable_ocr', options.enableOcr || true);
    formData.append('enable_transcription', options.enableTranscription || true);

    const response = await api.post('/api/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: options.onProgress,
    });
    return response.data;
  },

  // Get uploaded documents
  getDocuments: async () => {
    const response = await api.get('/api/documents');
    return response.data;
  },

  // Delete a document
  deleteDocument: async (documentId) => {
    const response = await api.delete(`/api/documents/${documentId}`);
    return response.data;
  },

  // Search documents
  searchDocuments: async (query) => {
    const response = await api.get('/api/documents/search', {
      params: { q: query },
    });
    return response.data;
  },
};

// Translation API endpoints
export const translationAPI = {
  // Translate text
  translateText: async (text, sourceLang, targetLang) => {
    const payload = {
      text,
      source_language: sourceLang,
      target_language: targetLang,
    };
    const response = await api.post('/api/translation/translate', payload);
    return response.data;
  },

  // Get supported languages
  getSupportedLanguages: async () => {
    const response = await api.get('/api/translation/languages');
    return response.data;
  },

  // Detect language
  detectLanguage: async (text) => {
    const payload = { text };
    const response = await api.post('/api/translation/detect', payload);
    return response.data;
  },
};

// User API endpoints
export const userAPI = {
  // Get user profile
  getProfile: async () => {
    const response = await api.get('/api/user/profile');
    return response.data;
  },

  // Update user preferences
  updatePreferences: async (preferences) => {
    const response = await api.put('/api/user/preferences', preferences);
    return response.data;
  },

  // Get user statistics
  getStats: async () => {
    const response = await api.get('/api/user/stats');
    return response.data;
  },
};

// Analytics API endpoints
export const analyticsAPI = {
  // Track user interaction
  trackEvent: async (event, data = {}) => {
    const payload = {
      event,
      data,
      timestamp: new Date().toISOString(),
    };
    const response = await api.post('/api/analytics/track', payload);
    return response.data;
  },

  // Get analytics data
  getAnalytics: async (period = '7d') => {
    const response = await api.get('/api/analytics', {
      params: { period },
    });
    return response.data;
  },
};

// System API endpoints
export const systemAPI = {
  // Get system status
  getStatus: async () => {
    const response = await api.get('/api/system/status');
    return response.data;
  },

  // Get system configuration
  getConfig: async () => {
    const response = await api.get('/api/system/config');
    return response.data;
  },

  // Health check
  healthCheck: async () => {
    const response = await api.get('/api/system/health');
    return response.data;
  },
};

export default api;
