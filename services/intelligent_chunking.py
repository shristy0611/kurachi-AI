"""
Intelligent document chunking service with semantic-aware splitting
Replaces basic text splitting with context-preserving, structure-aware chunking
Designed for Python 3.11 with full LangChain support
"""
import re
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path

from langchain.schema import Document as LangChainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama

from config import config
from services.markdown_converter import sota_markdown_chunking_service
from services.language_detection import DocumentLanguage, language_detection_service
from services.language_processing_pipelines import language_pipeline_factory
from utils.logger import get_logger

logger = get_logger("intelligent_chunking")


class ChunkType(Enum):
    """Types of document chunks"""
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST = "list"
    TABLE = "table"
    CODE = "code"
    QUOTE = "quote"
    METADATA = "metadata"
    MIXED = "mixed"


@dataclass
class ChunkMetadata:
    """Enhanced metadata for document chunks"""
    chunk_id: str
    chunk_type: ChunkType
    source_document: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    heading_level: Optional[int] = None
    table_info: Optional[Dict[str, Any]] = None
    list_info: Optional[Dict[str, Any]] = None
    position_in_document: Optional[int] = None
    semantic_context: Optional[str] = None
    related_chunks: Optional[List[str]] = None
    confidence_score: float = 1.0
    word_count: int = 0
    character_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type.value,
            "source_document": self.source_document,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "heading_level": self.heading_level,
            "table_info": self.table_info,
            "list_info": self.list_info,
            "position_in_document": self.position_in_document,
            "semantic_context": self.semantic_context,
            "related_chunks": self.related_chunks or [],
            "confidence_score": self.confidence_score,
            "word_count": self.word_count,
            "character_count": self.character_count
        }
        return {k: v for k, v in data.items() if v is not None}


@dataclass
class IntelligentChunk:
    """Enhanced document chunk with semantic information"""
    content: str
    metadata: ChunkMetadata
    
    def to_langchain_document(self) -> LangChainDocument:
        """Convert to LangChain document format"""
        return LangChainDocument(
            page_content=self.content,
            metadata=self.metadata.to_dict()
        )


class DocumentStructureAnalyzer:
    """Analyzes document structure to identify semantic boundaries"""
    
    def __init__(self):
        self.llm = Ollama(
            model=config.ai.llm_model,
            base_url=config.ai.ollama_base_url,
            temperature=0.1  # Low temperature for consistent analysis
        )
    
    def analyze_document_structure(self, content: str, source_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document structure using LLM"""
        try:
            # Create a structured prompt for document analysis
            prompt = f"""Analyze this document content and identify its structure. Focus on:

1. Document sections and headings
2. Paragraph boundaries
3. Lists and enumerations
4. Tables and structured data
5. Code blocks
6. Quotes and citations

Document content (first 2000 characters):
{content[:2000]}

Please provide a JSON response with the following structure:
{{
    "document_type": "type of document",
    "main_sections": [
        {{
            "title": "section title",
            "start_position": 0,
            "end_position": 100,
            "level": 1,
            "type": "heading/paragraph/list/table/code"
        }}
    ],
    "structural_elements": {{
        "headings": number,
        "paragraphs": number,
        "lists": number,
        "tables": number,
        "code_blocks": number
    }},
    "semantic_boundaries": [
        {{
            "position": 150,
            "reason": "topic change",
            "confidence": 0.8
        }}
    ]
}}

Respond only with valid JSON."""

            response = self.llm.invoke(prompt)
            
            try:
                analysis = json.loads(response)
                return analysis
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM structure analysis, using fallback")
                return self._fallback_structure_analysis(content)
                
        except Exception as e:
            logger.error(f"Document structure analysis failed: {e}")
            return self._fallback_structure_analysis(content)
    
    def _fallback_structure_analysis(self, content: str) -> Dict[str, Any]:
        """Fallback structure analysis using regex patterns"""
        analysis = {
            "document_type": "text",
            "main_sections": [],
            "structural_elements": {
                "headings": 0,
                "paragraphs": 0,
                "lists": 0,
                "tables": 0,
                "code_blocks": 0
            },
            "semantic_boundaries": []
        }
        
        # Find headings (markdown style)
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        headings = list(re.finditer(heading_pattern, content, re.MULTILINE))
        analysis["structural_elements"]["headings"] = len(headings)
        
        for match in headings:
            level = len(match.group(1))
            title = match.group(2)
            analysis["main_sections"].append({
                "title": title,
                "start_position": match.start(),
                "end_position": match.end(),
                "level": level,
                "type": "heading"
            })
        
        # Find paragraphs (double newline separated)
        paragraphs = content.split('\n\n')
        analysis["structural_elements"]["paragraphs"] = len([p for p in paragraphs if p.strip()])
        
        # Find lists
        list_pattern = r'^[\s]*[-*+]\s+|^\s*\d+\.\s+'
        list_items = re.findall(list_pattern, content, re.MULTILINE)
        analysis["structural_elements"]["lists"] = len(list_items)
        
        # Find tables (simple detection)
        table_pattern = r'\|.*\|'
        table_rows = re.findall(table_pattern, content)
        analysis["structural_elements"]["tables"] = len(table_rows)
        
        # Find code blocks
        code_pattern = r'```[\s\S]*?```|`[^`]+`'
        code_blocks = re.findall(code_pattern, content)
        analysis["structural_elements"]["code_blocks"] = len(code_blocks)
        
        return analysis


class SemanticChunker:
    """Semantic-aware document chunker"""
    
    def __init__(self):
        self.structure_analyzer = DocumentStructureAnalyzer()
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.ai.chunk_size,
            chunk_overlap=config.ai.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.language_aware_splitters = {}  # Cache for language-specific splitters
    
    def chunk_document(self, document: LangChainDocument) -> List[IntelligentChunk]:
        """Create intelligent chunks from a document with language-aware processing"""
        try:
            content = document.page_content
            source_metadata = document.metadata
            
            # Get language-specific text splitter
            language_splitter = self._get_language_aware_splitter(document)
            
            # Analyze document structure
            structure_analysis = self.structure_analyzer.analyze_document_structure(
                content, source_metadata
            )
            
            # Create chunks based on structure with language awareness
            chunks = self._create_structure_aware_chunks(
                content, structure_analysis, source_metadata, language_splitter
            )
            
            # Post-process chunks for optimal size
            optimized_chunks = self._optimize_chunk_sizes(chunks, language_splitter)
            
            # Add semantic context and relationships
            enhanced_chunks = self._add_semantic_context(optimized_chunks)
            
            logger.info(f"Created {len(enhanced_chunks)} intelligent chunks from document with language-aware processing")
            return enhanced_chunks
            
        except Exception as e:
            logger.error(f"Intelligent chunking failed, using fallback: {e}")
            return self._fallback_chunking(document)
    
    def _get_language_aware_splitter(self, document: LangChainDocument) -> RecursiveCharacterTextSplitter:
        """Get language-specific text splitter for the document"""
        try:
            # Check if document already has language metadata
            detected_language = document.metadata.get('detected_language', 'unknown')
            is_mixed_language = document.metadata.get('is_mixed_language', False)
            
            # Create cache key
            cache_key = f"{detected_language}_{is_mixed_language}"
            
            # Return cached splitter if available
            if cache_key in self.language_aware_splitters:
                return self.language_aware_splitters[cache_key]
            
            # Get appropriate pipeline and text splitter
            if detected_language == 'ja':
                pipeline = language_pipeline_factory.get_pipeline(DocumentLanguage.JAPANESE)
                splitter = pipeline.get_text_splitter()
            elif detected_language == 'en':
                pipeline = language_pipeline_factory.get_pipeline(DocumentLanguage.ENGLISH)
                splitter = pipeline.get_text_splitter()
            elif is_mixed_language:
                pipeline = language_pipeline_factory.get_pipeline(DocumentLanguage.MIXED, is_mixed=True)
                splitter = pipeline.get_text_splitter()
            else:
                # Fallback to default splitter
                splitter = self.fallback_splitter
            
            # Cache the splitter
            self.language_aware_splitters[cache_key] = splitter
            
            logger.debug(f"Using language-aware splitter for {detected_language} (mixed: {is_mixed_language})")
            return splitter
            
        except Exception as e:
            logger.warning(f"Failed to get language-aware splitter: {e}, using fallback")
            return self.fallback_splitter
    
    def _create_structure_aware_chunks(self, content: str, structure: Dict[str, Any], 
                                     source_metadata: Dict[str, Any], 
                                     language_splitter: RecursiveCharacterTextSplitter) -> List[IntelligentChunk]:
        """Create chunks based on document structure"""
        chunks = []
        
        # Handle different document types
        if structure.get("structural_elements", {}).get("tables", 0) > 0:
            chunks.extend(self._extract_table_chunks(content, source_metadata))
        
        if structure.get("structural_elements", {}).get("lists", 0) > 0:
            chunks.extend(self._extract_list_chunks(content, source_metadata))
        
        if structure.get("structural_elements", {}).get("headings", 0) > 0:
            chunks.extend(self._extract_section_chunks(content, structure, source_metadata))
        
        if structure.get("structural_elements", {}).get("code_blocks", 0) > 0:
            chunks.extend(self._extract_code_chunks(content, source_metadata))
        
        # Extract remaining content as paragraph chunks
        remaining_content = self._get_remaining_content(content, chunks)
        if remaining_content.strip():
            chunks.extend(self._extract_paragraph_chunks(remaining_content, source_metadata, language_splitter))
        
        return chunks
    
    def _extract_table_chunks(self, content: str, source_metadata: Dict[str, Any]) -> List[IntelligentChunk]:
        """Extract table chunks keeping tabular data together"""
        chunks = []
        
        # Pattern for markdown tables
        table_pattern = r'(\|.*\|(?:\n\|.*\|)*)'
        tables = re.finditer(table_pattern, content, re.MULTILINE)
        
        for i, match in enumerate(tables):
            table_content = match.group(1)
            
            # Analyze table structure
            rows = table_content.split('\n')
            header_row = rows[0] if rows else ""
            data_rows = rows[2:] if len(rows) > 2 else []  # Skip separator row
            
            table_info = {
                "table_index": i,
                "rows": len(data_rows),
                "columns": len(header_row.split('|')) - 2 if header_row else 0,
                "has_header": bool(header_row),
                "start_position": match.start(),
                "end_position": match.end()
            }
            
            chunk_id = self._generate_chunk_id(table_content, source_metadata.get("source", ""))
            
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                chunk_type=ChunkType.TABLE,
                source_document=source_metadata.get("source", ""),
                page_number=source_metadata.get("page_number"),
                table_info=table_info,
                position_in_document=match.start(),
                word_count=len(table_content.split()),
                character_count=len(table_content)
            )
            
            chunks.append(IntelligentChunk(
                content=table_content,
                metadata=metadata
            ))
        
        return chunks
    
    def _extract_list_chunks(self, content: str, source_metadata: Dict[str, Any]) -> List[IntelligentChunk]:
        """Extract list chunks preserving list structure"""
        chunks = []
        
        # Pattern for lists (both ordered and unordered)
        list_pattern = r'((?:^[\s]*[-*+]\s+.+(?:\n|$))+|(?:^\s*\d+\.\s+.+(?:\n|$))+)'
        lists = re.finditer(list_pattern, content, re.MULTILINE)
        
        for i, match in enumerate(lists):
            list_content = match.group(1).strip()
            
            # Analyze list structure
            items = re.findall(r'^[\s]*[-*+]\s+(.+)|^\s*\d+\.\s+(.+)', list_content, re.MULTILINE)
            list_items = [item[0] or item[1] for item in items]
            
            list_info = {
                "list_index": i,
                "item_count": len(list_items),
                "list_type": "ordered" if re.search(r'^\s*\d+\.', list_content, re.MULTILINE) else "unordered",
                "start_position": match.start(),
                "end_position": match.end()
            }
            
            chunk_id = self._generate_chunk_id(list_content, source_metadata.get("source", ""))
            
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                chunk_type=ChunkType.LIST,
                source_document=source_metadata.get("source", ""),
                page_number=source_metadata.get("page_number"),
                list_info=list_info,
                position_in_document=match.start(),
                word_count=len(list_content.split()),
                character_count=len(list_content)
            )
            
            chunks.append(IntelligentChunk(
                content=list_content,
                metadata=metadata
            ))
        
        return chunks
    
    def _extract_section_chunks(self, content: str, structure: Dict[str, Any], 
                              source_metadata: Dict[str, Any]) -> List[IntelligentChunk]:
        """Extract section chunks based on headings"""
        chunks = []
        sections = structure.get("main_sections", [])
        
        for i, section in enumerate(sections):
            if section.get("type") != "heading":
                continue
            
            start_pos = section["start_position"]
            # Find end position (next section or end of document)
            end_pos = sections[i + 1]["start_position"] if i + 1 < len(sections) else len(content)
            
            section_content = content[start_pos:end_pos].strip()
            
            chunk_id = self._generate_chunk_id(section_content, source_metadata.get("source", ""))
            
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                chunk_type=ChunkType.HEADING,
                source_document=source_metadata.get("source", ""),
                page_number=source_metadata.get("page_number"),
                section_title=section.get("title"),
                heading_level=section.get("level"),
                position_in_document=start_pos,
                word_count=len(section_content.split()),
                character_count=len(section_content)
            )
            
            chunks.append(IntelligentChunk(
                content=section_content,
                metadata=metadata
            ))
        
        return chunks
    
    def _extract_code_chunks(self, content: str, source_metadata: Dict[str, Any]) -> List[IntelligentChunk]:
        """Extract code chunks preserving code structure"""
        chunks = []
        
        # Pattern for code blocks
        code_pattern = r'```(\w+)?\n?([\s\S]*?)```|`([^`]+)`'
        code_blocks = re.finditer(code_pattern, content)
        
        for i, match in enumerate(code_blocks):
            language = match.group(1) or "text"
            code_content = match.group(2) or match.group(3) or ""
            
            chunk_id = self._generate_chunk_id(code_content, source_metadata.get("source", ""))
            
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                chunk_type=ChunkType.CODE,
                source_document=source_metadata.get("source", ""),
                page_number=source_metadata.get("page_number"),
                position_in_document=match.start(),
                semantic_context=f"Code block in {language}",
                word_count=len(code_content.split()),
                character_count=len(code_content)
            )
            
            chunks.append(IntelligentChunk(
                content=code_content,
                metadata=metadata
            ))
        
        return chunks
    
    def _extract_paragraph_chunks(self, content: str, source_metadata: Dict[str, Any], 
                                language_splitter: RecursiveCharacterTextSplitter) -> List[IntelligentChunk]:
        """Extract paragraph chunks with semantic boundaries"""
        chunks = []
        
        # Split by double newlines (paragraph boundaries)
        paragraphs = content.split('\n\n')
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if paragraph is too long and needs splitting
            if len(paragraph) > config.ai.chunk_size:
                # Use language-aware splitter for long paragraphs
                sub_chunks = language_splitter.split_text(paragraph)
                for j, sub_chunk in enumerate(sub_chunks):
                    chunk_id = self._generate_chunk_id(sub_chunk, source_metadata.get("source", ""))
                    
                    metadata = ChunkMetadata(
                        chunk_id=chunk_id,
                        chunk_type=ChunkType.PARAGRAPH,
                        source_document=source_metadata.get("source", ""),
                        page_number=source_metadata.get("page_number"),
                        position_in_document=i,
                        semantic_context=f"Paragraph {i+1}, part {j+1}",
                        word_count=len(sub_chunk.split()),
                        character_count=len(sub_chunk)
                    )
                    
                    chunks.append(IntelligentChunk(
                        content=sub_chunk,
                        metadata=metadata
                    ))
            else:
                chunk_id = self._generate_chunk_id(paragraph, source_metadata.get("source", ""))
                
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    chunk_type=ChunkType.PARAGRAPH,
                    source_document=source_metadata.get("source", ""),
                    page_number=source_metadata.get("page_number"),
                    position_in_document=i,
                    semantic_context=f"Paragraph {i+1}",
                    word_count=len(paragraph.split()),
                    character_count=len(paragraph)
                )
                
                chunks.append(IntelligentChunk(
                    content=paragraph,
                    metadata=metadata
                ))
        
        return chunks
    
    def _get_remaining_content(self, content: str, extracted_chunks: List[IntelligentChunk]) -> str:
        """Get content that hasn't been extracted into chunks yet"""
        # This is a simplified implementation
        # In practice, you'd track which parts of the content have been extracted
        extracted_positions = []
        for chunk in extracted_chunks:
            if chunk.metadata.position_in_document is not None:
                start = chunk.metadata.position_in_document
                end = start + chunk.metadata.character_count
                extracted_positions.append((start, end))
        
        # Sort positions and find gaps
        extracted_positions.sort()
        remaining_parts = []
        last_end = 0
        
        for start, end in extracted_positions:
            if start > last_end:
                remaining_parts.append(content[last_end:start])
            last_end = max(last_end, end)
        
        # Add remaining content after last extraction
        if last_end < len(content):
            remaining_parts.append(content[last_end:])
        
        return '\n\n'.join(part.strip() for part in remaining_parts if part.strip())
    
    def _optimize_chunk_sizes(self, chunks: List[IntelligentChunk], 
                            language_splitter: RecursiveCharacterTextSplitter) -> List[IntelligentChunk]:
        """Optimize chunk sizes for better processing"""
        optimized_chunks = []
        
        for chunk in chunks:
            # If chunk is too small, try to merge with adjacent chunks
            if chunk.metadata.character_count < config.ai.chunk_size // 4:
                # For now, keep small chunks as-is
                # In a more sophisticated implementation, you'd merge related small chunks
                optimized_chunks.append(chunk)
            elif chunk.metadata.character_count > config.ai.chunk_size * 2:
                # Split very large chunks
                split_chunks = self._split_large_chunk(chunk, language_splitter)
                optimized_chunks.extend(split_chunks)
            else:
                optimized_chunks.append(chunk)
        
        return optimized_chunks
    
    def _split_large_chunk(self, chunk: IntelligentChunk, 
                         language_splitter: RecursiveCharacterTextSplitter = None) -> List[IntelligentChunk]:
        """Split a large chunk into smaller ones"""
        # Use language-aware splitter for large chunks
        splitter = language_splitter if language_splitter else self.fallback_splitter
        sub_texts = splitter.split_text(chunk.content)
        
        split_chunks = []
        for i, sub_text in enumerate(sub_texts):
            chunk_id = self._generate_chunk_id(sub_text, chunk.metadata.source_document)
            
            # Copy metadata and update for sub-chunk
            new_metadata = ChunkMetadata(
                chunk_id=chunk_id,
                chunk_type=chunk.metadata.chunk_type,
                source_document=chunk.metadata.source_document,
                page_number=chunk.metadata.page_number,
                section_title=chunk.metadata.section_title,
                heading_level=chunk.metadata.heading_level,
                position_in_document=chunk.metadata.position_in_document,
                semantic_context=f"{chunk.metadata.semantic_context} (part {i+1})" if chunk.metadata.semantic_context else f"Part {i+1}",
                word_count=len(sub_text.split()),
                character_count=len(sub_text)
            )
            
            split_chunks.append(IntelligentChunk(
                content=sub_text,
                metadata=new_metadata
            ))
        
        return split_chunks
    
    def _add_semantic_context(self, chunks: List[IntelligentChunk]) -> List[IntelligentChunk]:
        """Add semantic context and relationships between chunks"""
        # Add relationships between adjacent chunks
        for i, chunk in enumerate(chunks):
            related_chunks = []
            
            # Add previous and next chunk IDs
            if i > 0:
                related_chunks.append(chunks[i-1].metadata.chunk_id)
            if i < len(chunks) - 1:
                related_chunks.append(chunks[i+1].metadata.chunk_id)
            
            # Add chunks from the same section
            for other_chunk in chunks:
                if (other_chunk != chunk and 
                    other_chunk.metadata.section_title == chunk.metadata.section_title and
                    chunk.metadata.section_title is not None):
                    related_chunks.append(other_chunk.metadata.chunk_id)
            
            chunk.metadata.related_chunks = list(set(related_chunks))
        
        return chunks
    
    def _generate_chunk_id(self, content: str, source: str) -> str:
        """Generate unique chunk ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        source_hash = hashlib.md5(source.encode()).hexdigest()[:4]
        return f"chunk_{source_hash}_{content_hash}"
    
    def _fallback_chunking(self, document: LangChainDocument) -> List[IntelligentChunk]:
        """Fallback to basic chunking if intelligent chunking fails"""
        logger.warning("Using fallback chunking method")
        
        # Use language-aware splitter even for fallback
        language_splitter = self._get_language_aware_splitter(document)
        chunks = language_splitter.split_documents([document])
        intelligent_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_chunk_id(chunk.page_content, document.metadata.get("source", ""))
            
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                chunk_type=ChunkType.MIXED,
                source_document=document.metadata.get("source", ""),
                page_number=document.metadata.get("page_number"),
                position_in_document=i,
                semantic_context="Fallback chunk",
                word_count=len(chunk.page_content.split()),
                character_count=len(chunk.page_content)
            )
            
            intelligent_chunks.append(IntelligentChunk(
                content=chunk.page_content,
                metadata=metadata
            ))
        
        return intelligent_chunks


class IntelligentChunkingService:
    """SOTA Intelligent document chunking with Markdown conversion for optimal RAG performance"""
    
    def __init__(self):
        self.semantic_chunker = SemanticChunker()
        self.use_sota_markdown = True  # Enable SOTA Markdown conversion
        logger.info("Initialized SOTA intelligent chunking service with Markdown conversion")
    
    def chunk_documents(self, documents: List[LangChainDocument]) -> List[LangChainDocument]:
        """Process multiple documents with SOTA Markdown conversion and intelligent chunking"""
        try:
            if self.use_sota_markdown and documents:
                logger.info("🚀 Using SOTA Markdown conversion for optimal RAG performance")
                
                # Step 1: Convert to structured Markdown
                markdown_result = sota_markdown_chunking_service.process_documents_to_markdown(documents)
                
                if markdown_result.success:
                    logger.info(f"✅ Successfully converted to Markdown ({len(markdown_result.markdown_content)} chars)")
                    
                    # Create Markdown document
                    original_metadata = documents[0].metadata if documents else {}
                    markdown_doc = sota_markdown_chunking_service.create_markdown_document(
                        markdown_result.markdown_content, original_metadata
                    )
                    
                    # Step 2: Apply intelligent chunking to Markdown
                    intelligent_chunks = self.semantic_chunker.chunk_document(markdown_doc)
                    langchain_chunks = [chunk.to_langchain_document() for chunk in intelligent_chunks]
                    
                    # Add conversion metadata to chunks
                    conversion_stats = sota_markdown_chunking_service.get_conversion_statistics(markdown_result)
                    for chunk in langchain_chunks:
                        chunk.metadata.update({
                            "sota_markdown_conversion": True,
                            "conversion_quality": conversion_stats.get("conversion_quality", {}),
                            "rag_optimization_score": conversion_stats.get("rag_optimization_score", 0)
                        })
                    
                    logger.info(f"🎯 SOTA processing complete: {len(langchain_chunks)} optimized chunks created")
                    return langchain_chunks
                
                else:
                    logger.warning(f"Markdown conversion failed: {markdown_result.error_message}")
                    logger.info("Falling back to standard intelligent chunking")
            
            # Fallback to standard intelligent chunking
            return self._standard_chunking(documents)
            
        except Exception as e:
            logger.error(f"SOTA chunking failed: {e}")
            logger.info("Falling back to standard intelligent chunking")
            return self._standard_chunking(documents)
    
    def _standard_chunking(self, documents: List[LangChainDocument]) -> List[LangChainDocument]:
        """Standard intelligent chunking without Markdown conversion"""
        all_chunks = []
        
        for document in documents:
            try:
                intelligent_chunks = self.semantic_chunker.chunk_document(document)
                langchain_chunks = [chunk.to_langchain_document() for chunk in intelligent_chunks]
                all_chunks.extend(langchain_chunks)
                
                logger.info(f"Processed document with {len(intelligent_chunks)} intelligent chunks")
                
            except Exception as e:
                logger.error(f"Failed to process document {document.metadata.get('source', 'unknown')}: {e}")
                # Fallback to basic chunking
                fallback_chunks = self.semantic_chunker._fallback_chunking(document)
                langchain_chunks = [chunk.to_langchain_document() for chunk in fallback_chunks]
                all_chunks.extend(langchain_chunks)
        
        return all_chunks
    
    def get_chunk_statistics(self, chunks: List[LangChainDocument]) -> Dict[str, Any]:
        """Get statistics about the chunking results"""
        if not chunks:
            return {}
        
        chunk_types = {}
        total_words = 0
        total_chars = 0
        
        for chunk in chunks:
            chunk_type = chunk.metadata.get("chunk_type", "unknown")
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            total_words += chunk.metadata.get("word_count", 0)
            total_chars += chunk.metadata.get("character_count", 0)
        
        return {
            "total_chunks": len(chunks),
            "chunk_types": chunk_types,
            "total_words": total_words,
            "total_characters": total_chars,
            "average_words_per_chunk": total_words / len(chunks) if chunks else 0,
            "average_chars_per_chunk": total_chars / len(chunks) if chunks else 0
        }


# Global intelligent chunking service instance
intelligent_chunking_service = IntelligentChunkingService()