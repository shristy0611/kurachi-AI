import { create } from 'zustand';
import { chatAPI, analyticsAPI } from '../services/api';

const useChatStore = create((set, get) => ({
  // State
  messages: [],
  conversations: [],
  currentConversationId: null,
  isLoading: false,
  isTyping: false,
  error: null,
  userPreferences: {
    language: 'en',
    theme: 'dark',
    model: 'default',
    temperature: 0.7,
    maxTokens: 2048,
  },

  // Actions
  setLoading: (loading) => set({ isLoading: loading }),
  setTyping: (typing) => set({ isTyping: typing }),
  setError: (error) => set({ error }),
  clearError: () => set({ error: null }),

  // User preferences
  updatePreferences: (preferences) => {
    set((state) => ({
      userPreferences: { ...state.userPreferences, ...preferences }
    }));
    // Save to localStorage
    localStorage.setItem('userPreferences', JSON.stringify(get().userPreferences));
  },

  loadPreferences: () => {
    const saved = localStorage.getItem('userPreferences');
    if (saved) {
      const preferences = JSON.parse(saved);
      set({ userPreferences: preferences });
    }
  },

  // Conversations
  setCurrentConversation: (conversationId) => {
    set({ currentConversationId: conversationId });
  },

  loadConversations: async () => {
    try {
      set({ isLoading: true });
      const response = await chatAPI.getConversations();
      set({ conversations: response.conversations || [] });
    } catch (error) {
      set({ error: error.message });
    } finally {
      set({ isLoading: false });
    }
  },

  createConversation: async (title = null) => {
    try {
      set({ isLoading: true });
      const response = await chatAPI.createConversation(title);
      const newConversation = response.conversation;
      
      set((state) => ({
        conversations: [newConversation, ...state.conversations],
        currentConversationId: newConversation.id,
      }));

      // Track analytics
      analyticsAPI.trackEvent('conversation_created', {
        conversation_id: newConversation.id,
        title: newConversation.title,
      });

      return newConversation;
    } catch (error) {
      set({ error: error.message });
      throw error;
    } finally {
      set({ isLoading: false });
    }
  },

  deleteConversation: async (conversationId) => {
    try {
      await chatAPI.deleteConversation(conversationId);
      set((state) => ({
        conversations: state.conversations.filter(c => c.id !== conversationId),
        currentConversationId: state.currentConversationId === conversationId ? null : state.currentConversationId,
      }));

      // Track analytics
      analyticsAPI.trackEvent('conversation_deleted', {
        conversation_id: conversationId,
      });
    } catch (error) {
      set({ error: error.message });
      throw error;
    }
  },

  // Messages
  loadMessages: async (conversationId) => {
    try {
      set({ isLoading: true });
      const response = await chatAPI.getMessages(conversationId);
      set({ 
        messages: response.messages || [],
        currentConversationId: conversationId,
      });
    } catch (error) {
      set({ error: error.message });
    } finally {
      set({ isLoading: false });
    }
  },

  addMessage: (message) => {
    set((state) => ({
      messages: [...state.messages, message],
    }));
  },

  updateMessage: (messageId, updates) => {
    set((state) => ({
      messages: state.messages.map(msg =>
        msg.id === messageId ? { ...msg, ...updates } : msg
      ),
    }));
  },

  sendMessage: async (content, options = {}) => {
    const { userPreferences } = get();
    const conversationId = get().currentConversationId;

    // Create user message
    const userMessage = {
      id: Date.now(),
      type: 'user',
      content,
      timestamp: new Date().toISOString(),
      conversation_id: conversationId,
    };

    // Add user message to state
    get().addMessage(userMessage);

    // Create AI message placeholder
    const aiMessageId = Date.now() + 1;
    const aiMessage = {
      id: aiMessageId,
      type: 'ai',
      content: '',
      timestamp: new Date().toISOString(),
      conversation_id: conversationId,
      isTyping: true,
    };

    // Add AI message placeholder
    get().addMessage(aiMessage);
    set({ isTyping: true });

    try {
      // Send message to backend
      const response = await chatAPI.sendMessage(content, conversationId, {
        language: userPreferences.language,
        model: userPreferences.model,
        temperature: userPreferences.temperature,
        maxTokens: userPreferences.maxTokens,
        ...options,
      });

      // Update AI message with response
      get().updateMessage(aiMessageId, {
        content: response.response,
        isTyping: false,
        metadata: response.metadata || {},
      });

      // Track analytics
      analyticsAPI.trackEvent('message_sent', {
        conversation_id: conversationId,
        message_length: content.length,
        response_time: response.response_time,
        tokens_used: response.metadata?.tokens_used,
      });

      return response;
    } catch (error) {
      // Update AI message with error
      get().updateMessage(aiMessageId, {
        content: 'Sorry, I encountered an error. Please try again.',
        isTyping: false,
        error: true,
      });
      set({ error: error.message });
      throw error;
    } finally {
      set({ isTyping: false });
    }
  },

  regenerateResponse: async (messageId) => {
    try {
      set({ isLoading: true });
      const response = await chatAPI.regenerateResponse(messageId);
      
      // Update the message with new response
      get().updateMessage(messageId, {
        content: response.response,
        timestamp: new Date().toISOString(),
        metadata: response.metadata || {},
      });

      // Track analytics
      analyticsAPI.trackEvent('response_regenerated', {
        message_id: messageId,
        conversation_id: get().currentConversationId,
      });

      return response;
    } catch (error) {
      set({ error: error.message });
      throw error;
    } finally {
      set({ isLoading: false });
    }
  },

  clearMessages: () => {
    set({ messages: [] });
  },

  // Voice recording
  startVoiceRecording: () => {
    // Voice recording logic would go here
    analyticsAPI.trackEvent('voice_recording_started');
  },

  stopVoiceRecording: () => {
    // Voice recording logic would go here
    analyticsAPI.trackEvent('voice_recording_stopped');
  },

  // File upload
  uploadFile: async (file) => {
    try {
      set({ isLoading: true });
      // File upload logic would go here
      analyticsAPI.trackEvent('file_uploaded', {
        file_name: file.name,
        file_size: file.size,
        file_type: file.type,
      });
    } catch (error) {
      set({ error: error.message });
      throw error;
    } finally {
      set({ isLoading: false });
    }
  },

  // Reset store
  reset: () => {
    set({
      messages: [],
      conversations: [],
      currentConversationId: null,
      isLoading: false,
      isTyping: false,
      error: null,
    });
  },
}));

export default useChatStore;
