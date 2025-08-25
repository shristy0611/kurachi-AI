import React, { useState, useRef, useEffect } from 'react';
import { Send, Mic, MicOff, Plus, Settings, Moon, Sun, Copy, RotateCcw, ThumbsUp, ThumbsDown, Paperclip, Code, Image, FileText, Zap, Sparkles, Brain, MessageSquare, History, User, Bot, Command, Cpu } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { toast } from 'react-hot-toast';
import useChatStore from '../store/chatStore';
import { chatAPI, documentAPI, translationAPI } from '../services/api';

const PremiumAIChatbot = () => {
  const {
    messages,
    conversations,
    currentConversationId,
    isLoading,
    isTyping,
    error,
    userPreferences,
    setCurrentConversation,
    loadConversations,
    createConversation,
    deleteConversation,
    loadMessages,
    sendMessage,
    regenerateResponse,
    clearMessages,
    updatePreferences,
    loadPreferences,
  } = useChatStore();

  const [isDarkMode, setIsDarkMode] = useState(userPreferences.theme === 'dark');
  const [isRecording, setIsRecording] = useState(false);
  const [inputText, setInputText] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [supportedLanguages, setSupportedLanguages] = useState([]);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Load preferences and conversations on mount
  useEffect(() => {
    loadPreferences();
    loadConversations();
    loadSupportedLanguages();
  }, []);

  // Update theme when userPreferences changes
  useEffect(() => {
    setIsDarkMode(userPreferences.theme === 'dark');
    if (userPreferences.theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [userPreferences.theme]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const loadSupportedLanguages = async () => {
    try {
      const response = await translationAPI.getSupportedLanguages();
      setSupportedLanguages(response.languages || []);
    } catch (error) {
      console.error('Failed to load languages:', error);
    }
  };

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;

    try {
      // Create conversation if none exists
      let conversationId = currentConversationId;
      if (!conversationId) {
        const conversation = await createConversation();
        conversationId = conversation.id;
      }

      // Send message
      await sendMessage(inputText);
      setInputText('');
    } catch (error) {
      toast.error('Failed to send message. Please try again.');
      console.error('Send message error:', error);
    }
  };

  const handleNewConversation = async () => {
    try {
      await createConversation();
      clearMessages();
      toast.success('New conversation started!');
    } catch (error) {
      toast.error('Failed to create new conversation.');
    }
  };

  const handleConversationSelect = async (conversationId) => {
    try {
      await loadMessages(conversationId);
      setCurrentConversation(conversationId);
    } catch (error) {
      toast.error('Failed to load conversation.');
    }
  };

  const handleDeleteConversation = async (conversationId) => {
    try {
      await deleteConversation(conversationId);
      toast.success('Conversation deleted!');
    } catch (error) {
      toast.error('Failed to delete conversation.');
    }
  };

  const handleRegenerateResponse = async (messageId) => {
    try {
      await regenerateResponse(messageId);
      toast.success('Response regenerated!');
    } catch (error) {
      toast.error('Failed to regenerate response.');
    }
  };

  const copyMessage = async (content) => {
    try {
      await navigator.clipboard.writeText(content);
      toast.success('Message copied to clipboard!');
    } catch (error) {
      toast.error('Failed to copy message.');
    }
  };

  const handleVoiceToggle = () => {
    setIsRecording(!isRecording);
    if (!isRecording) {
      toast.success('Voice recording started!');
    } else {
      toast.success('Voice recording stopped!');
    }
  };

  const handleThemeToggle = () => {
    const newTheme = isDarkMode ? 'light' : 'dark';
    updatePreferences({ theme: newTheme });
    toast.success(`${newTheme.charAt(0).toUpperCase() + newTheme.slice(1)} theme activated!`);
  };

  const handleLanguageChange = (language) => {
    updatePreferences({ language });
    toast.success(`Language changed to ${language.toUpperCase()}`);
  };

  const MessageComponent = ({ message, isLast }) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`group mb-8 ${message.type === 'user' ? 'flex justify-end' : 'flex justify-start'}`}
    >
      <div className={`max-w-4xl flex ${message.type === 'user' ? 'flex-row-reverse' : 'flex-row'} gap-6`}>
        {/* Enhanced Avatar */}
        <motion.div
          whileHover={{ scale: 1.05 }}
          className={`flex-shrink-0 w-12 h-12 rounded-2xl ${
            message.type === 'user' 
              ? 'bg-gradient-to-br from-violet-500 via-purple-500 to-blue-500 shadow-lg shadow-violet-500/25' 
              : 'bg-gradient-to-br from-emerald-400 via-teal-500 to-cyan-400 shadow-lg shadow-emerald-500/25'
          } flex items-center justify-center ring-2 ring-white/20 backdrop-blur-xl relative overflow-hidden transition-all duration-300`}
        >
          <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent"></div>
          {message.type === 'user' ? 
            <User className="w-6 h-6 text-white relative z-10" /> : 
            <Cpu className="w-6 h-6 text-white relative z-10" />
          }
        </motion.div>

        {/* Enhanced Message Content */}
        <div className={`flex flex-col ${message.type === 'user' ? 'items-end' : 'items-start'}`}>
          <motion.div
            whileHover={{ y: -2 }}
            className={`px-8 py-6 rounded-3xl backdrop-blur-xl border-2 shadow-2xl transition-all duration-500 hover:shadow-3xl relative overflow-hidden ${
              message.type === 'user'
                ? isDarkMode 
                  ? 'bg-gradient-to-br from-violet-600/80 to-purple-600/80 border-violet-400/30 text-white shadow-violet-500/20' 
                  : 'bg-gradient-to-br from-violet-500 to-purple-500 border-violet-300/40 text-white shadow-violet-500/30'
                : isDarkMode
                  ? 'bg-slate-800/90 border-slate-600/30 text-slate-50 shadow-slate-900/50'
                  : 'bg-white/95 border-slate-200/50 text-slate-800 shadow-slate-900/10'
            }`}
          >
            {/* Subtle animated background pattern */}
            <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-50"></div>
            
            <div className="prose prose-sm max-w-none relative z-10">
              <p className="leading-relaxed font-medium text-base m-0">{message.content}</p>
            </div>
            
            {/* Enhanced Message Actions */}
            {message.type === 'ai' && (
              <motion.div
                initial={{ opacity: 0 }}
                whileHover={{ opacity: 1 }}
                className="flex items-center gap-3 mt-6 pt-4 border-t border-slate-600/20 transition-all duration-300"
              >
                <motion.button 
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => copyMessage(message.content)}
                  className="p-2 rounded-xl hover:bg-white/10 transition-all duration-200"
                  title="Copy message"
                >
                  <Copy className="w-4 h-4 text-slate-300 hover:text-white transition-colors" />
                </motion.button>
                <motion.button 
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => handleRegenerateResponse(message.id)}
                  className="p-2 rounded-xl hover:bg-white/10 transition-all duration-200"
                  title="Regenerate response"
                >
                  <RotateCcw className="w-4 h-4 text-slate-300 hover:text-white transition-colors" />
                </motion.button>
                <motion.button 
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  className="p-2 rounded-xl hover:bg-green-500/20 transition-all duration-200" 
                  title="Good response"
                >
                  <ThumbsUp className="w-4 h-4 text-slate-300 hover:text-green-400 transition-colors" />
                </motion.button>
                <motion.button 
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  className="p-2 rounded-xl hover:bg-red-500/20 transition-all duration-200" 
                  title="Poor response"
                >
                  <ThumbsDown className="w-4 h-4 text-slate-300 hover:text-red-400 transition-colors" />
                </motion.button>
              </motion.div>
            )}
          </motion.div>
          
          <div className={`text-sm mt-3 font-medium ${isDarkMode ? 'text-slate-400' : 'text-slate-600'} ${message.type === 'user' ? 'text-right' : 'text-left'}`}>
            {new Date(message.timestamp).toLocaleTimeString()}
          </div>
        </div>
      </div>
    </motion.div>
  );

  const TypingIndicator = () => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex justify-start mb-8"
    >
      <div className="max-w-4xl flex flex-row gap-6">
        <div className="flex-shrink-0 w-12 h-12 rounded-2xl bg-gradient-to-br from-emerald-400 via-teal-500 to-cyan-400 shadow-lg shadow-emerald-500/25 flex items-center justify-center ring-2 ring-white/20 backdrop-blur-xl relative overflow-hidden animate-pulse">
          <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent"></div>
          <Cpu className="w-6 h-6 text-white relative z-10" />
        </div>
        <div className={`px-8 py-6 rounded-3xl backdrop-blur-xl border-2 shadow-2xl ${
          isDarkMode ? 'bg-slate-800/90 border-slate-600/30' : 'bg-white/95 border-slate-200/50'
        }`}>
          <div className="flex space-x-2">
            <div className="w-3 h-3 bg-gradient-to-r from-emerald-400 to-teal-400 rounded-full animate-bounce"></div>
            <div className="w-3 h-3 bg-gradient-to-r from-teal-400 to-cyan-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
            <div className="w-3 h-3 bg-gradient-to-r from-cyan-400 to-blue-400 rounded-full animate-bounce" style={{animationDelay: '0.4s'}}></div>
          </div>
        </div>
      </div>
    </motion.div>
  );

  return (
    <div className={`h-screen flex ${isDarkMode ? 'dark bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950' : 'bg-gradient-to-br from-slate-50 via-white to-slate-100'} transition-all duration-500 relative overflow-hidden`}>
      {/* Animated background patterns */}
      <div className="absolute inset-0 opacity-30">
        <div className="absolute top-0 left-0 w-96 h-96 bg-gradient-to-br from-violet-500/20 to-transparent rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-0 right-0 w-96 h-96 bg-gradient-to-br from-emerald-500/20 to-transparent rounded-full blur-3xl animate-pulse" style={{animationDelay: '2s'}}></div>
      </div>

      {/* Enhanced Sidebar */}
      <motion.div
        initial={{ x: -300 }}
        animate={{ x: 0 }}
        className={`${sidebarOpen ? 'w-84' : 'w-20'} ${isDarkMode ? 'bg-slate-900/60' : 'bg-white/60'} backdrop-blur-2xl border-r-2 ${isDarkMode ? 'border-slate-700/50' : 'border-slate-200/50'} transition-all duration-500 flex flex-col relative z-10 shadow-2xl`}
      >
        {/* Enhanced Sidebar Header */}
        <div className={`p-6 border-b-2 ${isDarkMode ? 'border-slate-700/50' : 'border-slate-200/50'}`}>
          <div className="flex items-center justify-between">
            {sidebarOpen && (
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-gradient-to-br from-violet-500 via-purple-500 to-blue-500 rounded-2xl flex items-center justify-center shadow-lg shadow-violet-500/25 relative overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent"></div>
                  <Brain className="w-6 h-6 text-white relative z-10" />
                </div>
                <div>
                  <h1 className="font-bold text-xl bg-gradient-to-r from-violet-400 via-purple-400 to-blue-400 bg-clip-text text-transparent">
                    Kurachi AI
                  </h1>
                  <p className={`text-sm font-medium ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>Advanced Intelligence</p>
                </div>
              </div>
            )}
            <motion.button 
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className={`p-3 rounded-2xl ${isDarkMode ? 'hover:bg-slate-800/50' : 'hover:bg-slate-100/50'} transition-all duration-300`}
            >
              <Command className="w-6 h-6" />
            </motion.button>
          </div>
        </div>

        {/* Enhanced New Chat Button */}
        <div className="p-6">
          <motion.button
            whileHover={{ scale: 1.05, y: -2 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleNewConversation}
            className="w-full flex items-center justify-center gap-4 px-6 py-4 bg-gradient-to-r from-violet-600 via-purple-600 to-blue-600 hover:from-violet-700 hover:via-purple-700 hover:to-blue-700 rounded-2xl font-semibold text-white transition-all duration-300 shadow-lg shadow-violet-500/25 hover:shadow-xl hover:shadow-violet-500/40 relative overflow-hidden group"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-white/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
            <Plus className="w-6 h-6 relative z-10" />
            {sidebarOpen && <span className="relative z-10">New Conversation</span>}
          </motion.button>
        </div>

        {/* Enhanced Conversations */}
        {sidebarOpen && (
          <div className="flex-1 overflow-y-auto px-6">
            <div className="space-y-3">
              <AnimatePresence>
                {conversations.map(conv => (
                  <motion.div
                    key={conv.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    whileHover={{ y: -2 }}
                    onClick={() => handleConversationSelect(conv.id)}
                    className={`p-4 rounded-2xl cursor-pointer transition-all duration-300 hover:${isDarkMode ? 'bg-slate-800/40' : 'bg-slate-100/40'} group backdrop-blur-sm border border-transparent hover:${isDarkMode ? 'border-slate-600/30' : 'border-slate-200/30'} ${currentConversationId === conv.id ? (isDarkMode ? 'bg-slate-800/60' : 'bg-slate-100/60') : ''}`}
                  >
                    <div className="flex items-center justify-between">
                      <h3 className={`font-semibold text-base truncate ${isDarkMode ? 'text-slate-200' : 'text-slate-800'}`}>{conv.title || 'New Conversation'}</h3>
                      <span className={`text-xs font-medium ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>{new Date(conv.created_at).toLocaleDateString()}</span>
                    </div>
                    <p className={`text-sm truncate mt-2 ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>{conv.last_message || 'Start a new conversation'}</p>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </div>
        )}

        {/* Enhanced Settings */}
        <div className={`p-6 border-t-2 ${isDarkMode ? 'border-slate-700/50' : 'border-slate-200/50'}`}>
          <div className="flex items-center justify-between">
            <motion.button 
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleThemeToggle}
              className={`p-4 rounded-2xl ${isDarkMode ? 'hover:bg-slate-800/50' : 'hover:bg-slate-100/50'} transition-all duration-300 relative overflow-hidden group`}
              title="Toggle theme"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-yellow-400/10 to-orange-400/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              {isDarkMode ? <Sun className="w-6 h-6 text-slate-300 group-hover:text-yellow-400 relative z-10 transition-colors" /> : <Moon className="w-6 h-6 text-slate-600 group-hover:text-slate-800 relative z-10 transition-colors" />}
            </motion.button>
            <motion.button 
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
              className={`p-4 rounded-2xl ${isDarkMode ? 'hover:bg-slate-800/50' : 'hover:bg-slate-100/50'} transition-all duration-300`}
            >
              <Settings className={`w-6 h-6 ${isDarkMode ? 'text-slate-300 hover:text-white' : 'text-slate-600 hover:text-slate-800'} transition-colors`} />
            </motion.button>
            <motion.button 
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
              className={`p-4 rounded-2xl ${isDarkMode ? 'hover:bg-slate-800/50' : 'hover:bg-slate-100/50'} transition-all duration-300`}
            >
              <History className={`w-6 h-6 ${isDarkMode ? 'text-slate-300 hover:text-white' : 'text-slate-600 hover:text-slate-800'} transition-colors`} />
            </motion.button>
          </div>
        </div>
      </motion.div>

      {/* Enhanced Main Chat Area */}
      <div className="flex-1 flex flex-col relative z-10">
        {/* Enhanced Chat Header */}
        <div className={`px-8 py-6 border-b-2 ${isDarkMode ? 'border-slate-700/50 bg-slate-900/40' : 'border-slate-200/50 bg-white/40'} backdrop-blur-2xl shadow-lg`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-gradient-to-br from-emerald-400 via-teal-500 to-cyan-400 shadow-lg shadow-emerald-500/25 rounded-2xl flex items-center justify-center ring-2 ring-white/20 relative overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent"></div>
                  <Cpu className="w-6 h-6 text-white relative z-10" />
                </div>
                <div>
                  <h2 className={`font-bold text-xl ${isDarkMode ? 'text-slate-100' : 'text-slate-900'}`}>Kurachi Assistant</h2>
                  <p className={`text-sm font-medium ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>AI-Powered • Ready to assist</p>
                </div>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <span className="inline-flex items-center px-4 py-2 rounded-full text-sm font-semibold bg-gradient-to-r from-emerald-100 to-teal-100 text-emerald-800 dark:from-emerald-900/30 dark:to-teal-900/30 dark:text-emerald-300 border border-emerald-200/50 dark:border-emerald-700/30 shadow-lg">
                <span className="w-2 h-2 bg-emerald-500 rounded-full mr-3 animate-pulse shadow-lg shadow-emerald-500/50"></span>
                Online
              </span>
            </div>
          </div>
        </div>

        {/* Enhanced Messages Area */}
        <div className="flex-1 overflow-y-auto px-8 py-8 space-y-2">
          <AnimatePresence>
            {messages.map((message, index) => (
              <MessageComponent 
                key={message.id} 
                message={message} 
                isLast={index === messages.length - 1} 
              />
            ))}
          </AnimatePresence>
          {isTyping && <TypingIndicator />}
          <div ref={messagesEndRef} />
        </div>

        {/* Enhanced Input Area */}
        <div className={`px-8 py-6 border-t-2 ${isDarkMode ? 'border-slate-700/50 bg-slate-900/40' : 'border-slate-200/50 bg-white/40'} backdrop-blur-2xl shadow-2xl`}>
          <div className="max-w-5xl mx-auto">
            <div className={`flex items-end space-x-6 p-6 rounded-3xl backdrop-blur-xl border-2 ${isDarkMode ? 'bg-slate-800/60 border-slate-600/30 shadow-slate-900/50' : 'bg-white/80 border-slate-200/40 shadow-slate-900/10'} shadow-2xl hover:shadow-3xl transition-all duration-300 relative overflow-hidden`}>
              {/* Subtle glow effect */}
              <div className="absolute inset-0 bg-gradient-to-r from-violet-500/5 via-transparent to-emerald-500/5 opacity-50"></div>
              
              {/* Enhanced Quick Actions */}
              <div className="flex space-x-3 relative z-10">
                <motion.button 
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  className={`p-3 rounded-2xl ${isDarkMode ? 'hover:bg-slate-700/50' : 'hover:bg-slate-100/50'} transition-all duration-300 group`} 
                  title="Attach file"
                >
                  <Paperclip className={`w-6 h-6 ${isDarkMode ? 'text-slate-400 group-hover:text-white' : 'text-slate-600 group-hover:text-slate-900'} transition-all duration-300 group-hover:rotate-12`} />
                </motion.button>
                <motion.button 
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  className={`p-3 rounded-2xl ${isDarkMode ? 'hover:bg-slate-700/50' : 'hover:bg-slate-100/50'} transition-all duration-300 group`} 
                  title="Insert code"
                >
                  <Code className={`w-6 h-6 ${isDarkMode ? 'text-slate-400 group-hover:text-white' : 'text-slate-600 group-hover:text-slate-900'} transition-all duration-300`} />
                </motion.button>
                <motion.button 
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  className={`p-3 rounded-2xl ${isDarkMode ? 'hover:bg-slate-700/50' : 'hover:bg-slate-100/50'} transition-all duration-300 group`} 
                  title="Upload image"
                >
                  <Image className={`w-6 h-6 ${isDarkMode ? 'text-slate-400 group-hover:text-white' : 'text-slate-600 group-hover:text-slate-900'} transition-all duration-300`} />
                </motion.button>
              </div>

              {/* Enhanced Text Input */}
              <div className="flex-1 relative z-10">
                <textarea
                  ref={inputRef}
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSendMessage();
                    }
                  }}
                  placeholder="Ask me anything... (Enter to send, Shift+Enter for new line)"
                  className={`w-full resize-none border-0 bg-transparent focus:ring-0 focus:outline-none ${isDarkMode ? 'text-slate-100 placeholder-slate-400' : 'text-slate-900 placeholder-slate-500'} text-base font-medium leading-relaxed`}
                  rows={1}
                  style={{ minHeight: '32px', maxHeight: '160px' }}
                />
              </div>

              {/* Enhanced Voice & Send */}
              <div className="flex items-center space-x-3 relative z-10">
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleVoiceToggle}
                  className={`p-4 rounded-2xl transition-all duration-300 ${
                    isRecording 
                      ? 'bg-gradient-to-r from-red-500 to-pink-500 text-white shadow-lg shadow-red-500/40 animate-pulse ring-4 ring-red-500/30' 
                      : `${isDarkMode ? 'hover:bg-slate-700/50' : 'hover:bg-slate-100/50'} ${isDarkMode ? 'text-slate-400 hover:text-white' : 'text-slate-600 hover:text-slate-900'}`
                  }`}
                  title={isRecording ? 'Stop recording' : 'Start voice input'}
                >
                  {isRecording ? <MicOff className="w-6 h-6" /> : <Mic className="w-6 h-6" />}
                </motion.button>
                
                <motion.button
                  whileHover={{ scale: 1.1, y: -2 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleSendMessage}
                  disabled={!inputText.trim() || isLoading}
                  className="p-4 bg-gradient-to-r from-violet-600 via-purple-600 to-blue-600 hover:from-violet-700 hover:via-purple-700 hover:to-blue-700 disabled:from-slate-600 disabled:to-slate-600 rounded-2xl transition-all duration-300 transform hover:-translate-y-1 disabled:scale-100 disabled:translate-y-0 shadow-lg shadow-violet-500/25 hover:shadow-xl hover:shadow-violet-500/40 disabled:shadow-none relative overflow-hidden group"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-white/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                  <Send className="w-6 h-6 text-white relative z-10" />
                </motion.button>
              </div>
            </div>

            {/* Enhanced Quick Suggestions */}
            <div className="flex flex-wrap gap-3 mt-6 justify-center">
              {[
                { icon: "✨", text: "Write code for me", gradient: "from-violet-500 to-purple-500" },
                { icon: "🔍", text: "Analyze this data", gradient: "from-emerald-500 to-teal-500" },
                { icon: "📝", text: "Help me write", gradient: "from-blue-500 to-cyan-500" },
                { icon: "🎨", text: "Generate ideas", gradient: "from-pink-500 to-rose-500" }
              ].map((suggestion, index) => (
                <motion.button
                  key={index}
                  whileHover={{ scale: 1.05, y: -2 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setInputText(suggestion.text)}
                  className={`px-5 py-3 text-sm font-semibold rounded-2xl backdrop-blur-sm border-2 transition-all duration-300 relative overflow-hidden group ${
                    isDarkMode 
                      ? 'bg-slate-800/50 border-slate-600/30 hover:bg-slate-700/50 text-slate-200 hover:text-white' 
                      : 'bg-white/60 border-slate-200/40 hover:bg-white/80 text-slate-800 hover:text-slate-900'
                  } shadow-lg hover:shadow-xl`}
                >
                  <div className={`absolute inset-0 bg-gradient-to-r ${suggestion.gradient} opacity-0 group-hover:opacity-10 transition-opacity duration-300`}></div>
                  <span className="relative z-10 flex items-center gap-2">
                    <span>{suggestion.icon}</span>
                    <span>{suggestion.text}</span>
                  </span>
                </motion.button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PremiumAIChatbot;
