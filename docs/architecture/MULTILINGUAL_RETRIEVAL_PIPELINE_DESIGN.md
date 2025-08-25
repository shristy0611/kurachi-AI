# Multilingual Retrieval + Translation Pipeline Design
## For Task 5.4: Create multilingual conversation interface

🔥 **Building on the intelligent translation system (5.3) to create seamless cross-language Q&A**

## Architecture Overview

```
User Query (EN/JA) → Language Detection → Query Translation (if needed) → 
Document Retrieval → Result Translation → Response Assembly → Cache Update
```

## Core Pipeline Components

### 1. **Multilingual Query Processor**
```python
class MultilingualQueryProcessor:
    def process_query(self, query: str, user_preferences: Dict) -> Dict:
        # Detect query language
        query_lang = translation_service.detect_language(query)
        
        # Get user's preferred response language
        response_lang = user_preferences.get('answer_language', 'auto')
        
        # Determine search strategy
        search_strategy = self._determine_search_strategy(query_lang, response_lang)
        
        return {
            'original_query': query,
            'query_language': query_lang,
            'response_language': response_lang,
            'search_strategy': search_strategy
        }
```

### 2. **Cross-Language Document Retrieval**
```python
class CrossLanguageRetriever:
    def retrieve_documents(self, query_info: Dict) -> List[Document]:
        results = []
        
        # Strategy 1: Direct search in query language
        direct_results = self._search_in_language(
            query_info['original_query'], 
            query_info['query_language']
        )
        results.extend(direct_results)
        
        # Strategy 2: Translate query and search other language
        if len(direct_results) < self.min_results_threshold:
            translated_query = translation_service.translate(
                query_info['original_query'],
                target_language=self._get_alternate_language(query_info['query_language']),
                context_type=TranslationContext.GENERAL,
                quality=TranslationQuality.BUSINESS
            )
            
            cross_lang_results = self._search_in_language(
                translated_query['translated_text'],
                translated_query['target_language']
            )
            results.extend(cross_lang_results)
        
        return self._rank_and_deduplicate(results)
```

### 3. **Intelligent Response Assembly**
```python
class MultilingualResponseAssembler:
    def assemble_response(self, query_info: Dict, documents: List[Document]) -> Dict:
        # Group documents by language
        docs_by_lang = self._group_documents_by_language(documents)
        
        # Generate response in user's preferred language
        response_parts = []
        citations = []
        
        for lang, docs in docs_by_lang.items():
            if lang != query_info['response_language']:
                # Translate document snippets
                translated_docs = translation_service.translate_document_content(
                    docs, 
                    target_language=query_info['response_language'],
                    context_type=self._infer_context_from_query(query_info['original_query'])
                )
                docs = translated_docs
            
            # Extract relevant snippets and build citations
            for doc in docs:
                snippet = self._extract_relevant_snippet(doc, query_info['original_query'])
                citation = self._build_multilingual_citation(doc, snippet, query_info)
                citations.append(citation)
        
        # Generate final response
        response = self._generate_response_with_citations(
            query_info['original_query'], 
            citations, 
            query_info['response_language']
        )
        
        return {
            'response': response,
            'citations': citations,
            'languages_searched': list(docs_by_lang.keys()),
            'translation_stats': self._get_translation_stats()
        }
```

## Key Features to Implement

### 🌐 **Language Preference Management**
```python
class UserLanguagePreferences:
    def __init__(self):
        self.preferences_schema = {
            'ui_language': 'en',           # Interface language
            'answer_language': 'auto',     # AI response language (auto/en/ja)
            'fallback_languages': ['ja', 'en'],  # Search fallback order
            'show_original_text': True,    # Show original snippets
            'translation_quality_threshold': 0.75,  # Min quality for auto-translation
            'cultural_adaptation': True    # Adapt cultural references
        }
    
    def get_effective_language(self, query_lang: str, user_pref: str) -> str:
        if user_pref == 'auto':
            return query_lang  # Respond in same language as query
        return user_pref
```

### 🔍 **Cross-Language Search Strategy**
```python
class CrossLanguageSearchStrategy:
    def determine_strategy(self, query: str, query_lang: str) -> Dict:
        strategies = []
        
        # Always search in query language first
        strategies.append({
            'type': 'direct',
            'language': query_lang,
            'query': query,
            'weight': 1.0
        })
        
        # Add cross-language search if needed
        if self._should_search_cross_language(query, query_lang):
            alt_lang = 'ja' if query_lang == 'en' else 'en'
            translated_query = translation_service.translate(
                query, 
                target_language=Language(alt_lang),
                context_type=self._infer_query_context(query)
            )
            
            strategies.append({
                'type': 'translated',
                'language': alt_lang,
                'query': translated_query['translated_text'],
                'weight': 0.8,  # Slightly lower weight for translated queries
                'translation_confidence': translated_query['confidence']
            })
        
        return {'strategies': strategies}
```

### 📝 **Multilingual Citation Format**
```python
class MultilingualCitationBuilder:
    def build_citation(self, document: Document, snippet: str, query_info: Dict) -> Dict:
        citation = {
            'document_name': document.metadata.get('filename', 'Unknown'),
            'page': document.metadata.get('page', 1),
            'original_language': document.metadata.get('detected_language', 'unknown'),
            'snippet_language': query_info['response_language'],
            'original_snippet': snippet if not document.metadata.get('is_translated') else document.metadata.get('original_content', ''),
            'translated_snippet': snippet if document.metadata.get('is_translated') else None,
            'translation_confidence': document.metadata.get('translation_confidence', 1.0),
            'quality_indicator': self._get_quality_indicator(document.metadata.get('translation_confidence', 1.0))
        }
        
        return citation
    
    def format_citation(self, citation: Dict, user_prefs: Dict) -> str:
        template = self._get_citation_template(user_prefs['ui_language'])
        
        if user_prefs['show_original_text'] and citation['translated_snippet']:
            return template['with_original'].format(**citation)
        else:
            return template['translated_only'].format(**citation)
```

## Integration Points

### 🔗 **Chat Service Integration**
```python
# Enhanced chat_service.py
class ChatService:
    def __init__(self):
        self.multilingual_processor = MultilingualQueryProcessor()
        self.cross_lang_retriever = CrossLanguageRetriever()
        self.response_assembler = MultilingualResponseAssembler()
    
    async def process_multilingual_query(self, query: str, user_id: str) -> Dict:
        # Get user language preferences
        user_prefs = self._get_user_preferences(user_id)
        
        # Process query with language detection
        query_info = self.multilingual_processor.process_query(query, user_prefs)
        
        # Retrieve documents across languages
        documents = self.cross_lang_retriever.retrieve_documents(query_info)
        
        # Assemble multilingual response
        response = self.response_assembler.assemble_response(query_info, documents)
        
        # Cache and log
        self._cache_multilingual_response(query_info, response)
        self._log_multilingual_metrics(query_info, response)
        
        return response
```

### 🎨 **UI Components for Language Selection**
```python
# Streamlit UI components
def render_language_selector():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ui_lang = st.selectbox(
            "Interface Language",
            options=['en', 'ja'],
            format_func=lambda x: {'en': '🇺🇸 English', 'ja': '🇯🇵 日本語'}[x]
        )
    
    with col2:
        answer_lang = st.selectbox(
            "AI Response Language",
            options=['auto', 'en', 'ja'],
            format_func=lambda x: {
                'auto': '🤖 Auto-detect',
                'en': '🇺🇸 English',
                'ja': '🇯🇵 日本語'
            }[x]
        )
    
    with col3:
        show_original = st.checkbox("Show original text", value=True)
    
    return {
        'ui_language': ui_lang,
        'answer_language': answer_lang,
        'show_original_text': show_original
    }

def render_multilingual_response(response: Dict, user_prefs: Dict):
    # Main response
    st.markdown(response['response'])
    
    # Language info
    if len(response['languages_searched']) > 1:
        st.info(f"🌐 Searched in: {', '.join(response['languages_searched'])}")
    
    # Citations with translation info
    with st.expander("📚 Sources"):
        for citation in response['citations']:
            render_multilingual_citation(citation, user_prefs)

def render_multilingual_citation(citation: Dict, user_prefs: Dict):
    # Quality indicator
    quality_emoji = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}
    quality = citation.get('quality_indicator', 'medium')
    
    st.markdown(f"**{quality_emoji[quality]} {citation['document_name']}** (Page {citation['page']})")
    
    # Original text (if available and requested)
    if user_prefs['show_original_text'] and citation['original_snippet']:
        with st.container():
            st.markdown(f"**Original ({citation['original_language']}):**")
            st.markdown(f"_{citation['original_snippet']}_")
    
    # Translated text
    if citation['translated_snippet']:
        st.markdown(f"**Translation ({citation['snippet_language']}):**")
        st.markdown(citation['translated_snippet'])
        
        if citation['translation_confidence'] < 0.8:
            st.warning(f"⚠️ Translation confidence: {citation['translation_confidence']:.1%}")
```

## Performance Optimizations

### ⚡ **Caching Strategy**
```python
class MultilingualCacheManager:
    def __init__(self):
        self.query_cache = {}  # Query translation cache
        self.response_cache = {}  # Full response cache
        self.snippet_cache = {}  # Document snippet cache
    
    def get_cached_response(self, query: str, user_prefs: Dict) -> Optional[Dict]:
        cache_key = self._generate_cache_key(query, user_prefs)
        return self.response_cache.get(cache_key)
    
    def cache_response(self, query: str, user_prefs: Dict, response: Dict):
        cache_key = self._generate_cache_key(query, user_prefs)
        self.response_cache[cache_key] = {
            'response': response,
            'timestamp': time.time(),
            'hit_count': 0
        }
```

### 📊 **Metrics and Monitoring**
```python
class MultilingualMetrics:
    def track_query(self, query_info: Dict, response: Dict):
        metrics = {
            'query_language': query_info['query_language'],
            'response_language': query_info['response_language'],
            'languages_searched': response['languages_searched'],
            'translation_count': len([c for c in response['citations'] if c.get('translated_snippet')]),
            'avg_translation_confidence': self._calc_avg_confidence(response['citations']),
            'cache_hits': response.get('cache_hits', 0),
            'response_time': response.get('response_time', 0)
        }
        
        self._log_metrics(metrics)
        self._update_dashboards(metrics)
```

## Implementation Roadmap

### Phase 1: Core Pipeline (Week 1)
- [ ] Implement `MultilingualQueryProcessor`
- [ ] Build `CrossLanguageRetriever` 
- [ ] Create basic `MultilingualResponseAssembler`
- [ ] Add language preference management

### Phase 2: UI Integration (Week 2)
- [ ] Add language selector components
- [ ] Implement multilingual citation rendering
- [ ] Create preference persistence
- [ ] Add cultural adaptation features

### Phase 3: Optimization (Week 3)
- [ ] Implement multilingual caching
- [ ] Add performance metrics
- [ ] Create monitoring dashboards
- [ ] Optimize cross-language search ranking

### Phase 4: Testing & Polish (Week 4)
- [ ] Comprehensive multilingual testing
- [ ] User experience optimization
- [ ] Error handling and fallbacks
- [ ] Documentation and examples

## Success Metrics

### 🎯 **Functional Goals**
- ✅ EN query → JP docs → EN response with JP snippets
- ✅ JP query → EN docs → JP response with EN snippets  
- ✅ Mixed-language document search and retrieval
- ✅ Cultural adaptation and context preservation
- ✅ User preference persistence and management

### 📈 **Performance Goals**
- ⚡ <2s response time for cross-language queries
- 🎯 >85% translation confidence for business content
- 💾 >70% cache hit rate for repeated queries
- 🔍 >90% relevant results in cross-language search

### 🚀 **User Experience Goals**
- 🌐 Seamless language switching
- 📱 Mobile-friendly multilingual interface
- 🎨 Clear translation quality indicators
- 📚 Rich multilingual citations with original text

---

**Ready to implement?** This pipeline will give you enterprise-grade multilingual Q&A that leverages your translation cache, preserves terminology, and provides transparent quality indicators. The modular design means you can implement incrementally while maintaining the existing functionality.

Want me to start with the `MultilingualQueryProcessor` implementation? 🚀