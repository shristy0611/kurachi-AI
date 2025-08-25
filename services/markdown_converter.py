"""
SOTA Markdown Converter Service
Converts extracted document content to structured Markdown format for optimal RAG performance
LLMs perform significantly better with Markdown-structured content
"""
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from langchain.schema import Document as LangChainDocument
from langchain_community.llms import Ollama

from config import config
from utils.logger import get_logger

# LLM usage optimization constants
LLM_TIMEOUT = 20  # seconds
MAX_PAGES_FOR_LLM = 6  # Only use LLM for complex documents
MAX_CHARS_FOR_LLM = 8000  # Character limit for LLM processing
MIN_COMPLEXITY_SCORE = 0.6  # Minimum complexity to justify LLM usage

logger = get_logger("markdown_converter")


@dataclass
class MarkdownConversionResult:
    """Result of markdown conversion"""
    success: bool
    markdown_content: str
    original_structure: Dict[str, Any]
    conversion_metadata: Dict[str, Any]
    error_message: Optional[str] = None


def should_use_llm_structurize(documents: List[LangChainDocument], combined_content: str) -> bool:
    """Determine if LLM structurization is worth the cost"""
    # Check document count
    if len(documents) > MAX_PAGES_FOR_LLM:
        logger.info(f"Skipping LLM: too many pages ({len(documents)} > {MAX_PAGES_FOR_LLM})")
        return False
    
    # Check content length
    if len(combined_content) > MAX_CHARS_FOR_LLM:
        logger.info(f"Skipping LLM: content too long ({len(combined_content)} > {MAX_CHARS_FOR_LLM})")
        return False
    
    # Check document metadata for complexity indicators
    complexity_score = 0.0
    total_docs = len(documents)
    
    for doc in documents:
        meta = doc.metadata
        # Indicators that suggest LLM would be beneficial
        if meta.get("is_scanned", False):
            complexity_score += 0.3
        if meta.get("has_tables", False):
            complexity_score += 0.2
        if meta.get("layout_confidence", 1.0) < 0.6:
            complexity_score += 0.2
        if meta.get("extraction_method") == "ocr":
            complexity_score += 0.2
        if meta.get("document_type") in ["presentation", "spreadsheet"]:
            complexity_score += 0.1
    
    avg_complexity = complexity_score / total_docs if total_docs > 0 else 0.0
    
    should_use = avg_complexity >= MIN_COMPLEXITY_SCORE
    logger.info(f"LLM usage decision: complexity={avg_complexity:.2f}, threshold={MIN_COMPLEXITY_SCORE}, use_llm={should_use}")
    
    return should_use


class StructureToMarkdownConverter:
    """Converts document structures to clean Markdown format"""
    
    def __init__(self):
        self.llm = Ollama(
            model=config.ai.llm_model,
            base_url=config.ai.ollama_base_url,
            temperature=0.1,  # Low temperature for consistent formatting
            timeout=LLM_TIMEOUT  # Add timeout
        )
    
    def convert_to_markdown(self, documents: List[LangChainDocument]) -> MarkdownConversionResult:
        """Convert extracted documents to structured Markdown format with smart LLM usage"""
        try:
            logger.info(f"Converting {len(documents)} documents to Markdown format")
            
            # Combine all document content
            combined_content = self._combine_documents(documents)
            
            # Decide whether to use LLM based on complexity and size
            use_llm = should_use_llm_structurize(documents, combined_content)
            
            if use_llm:
                try:
                    # Analyze structure using LLM with timeout
                    logger.info("Using LLM for document structure analysis")
                    structure_analysis = self._analyze_document_structure_with_timeout(combined_content)
                    conversion_method = "llm_structured"
                    
                except Exception as llm_error:
                    logger.warning(f"LLM structure analysis failed: {llm_error}, falling back to simple conversion")
                    structure_analysis = self._simple_structure_analysis(combined_content)
                    conversion_method = "simple_fallback"
            else:
                # Use simple structure analysis
                logger.info("Using simple structure analysis (LLM not needed)")
                structure_analysis = self._simple_structure_analysis(combined_content)
                conversion_method = "simple_optimized"
            
            # Convert to Markdown
            markdown_content = self._convert_to_structured_markdown(
                combined_content, structure_analysis
            )
            
            # Validate and clean Markdown
            cleaned_markdown = self._clean_and_validate_markdown(markdown_content)
            
            conversion_metadata = {
                "original_documents": len(documents),
                "conversion_method": conversion_method,
                "used_llm": use_llm,
                "structure_elements": structure_analysis.get("structural_elements", {}),
                "markdown_length": len(cleaned_markdown),
                "conversion_quality": self._assess_conversion_quality(cleaned_markdown)
            }
            
            logger.info(f"Successfully converted to Markdown ({len(cleaned_markdown)} characters) using {conversion_method}")
            
            return MarkdownConversionResult(
                success=True,
                markdown_content=cleaned_markdown,
                original_structure=structure_analysis,
                conversion_metadata=conversion_metadata
            )
            
        except Exception as e:
            logger.error(f"Markdown conversion failed: {e}")
            return MarkdownConversionResult(
                success=False,
                markdown_content="",
                original_structure={},
                conversion_metadata={},
                error_message=str(e)
            )
    
    def _combine_documents(self, documents: List[LangChainDocument]) -> str:
        """Combine multiple documents into a single content string"""
        combined_parts = []
        
        for i, doc in enumerate(documents):
            content = doc.page_content.strip()
            if not content:
                continue
            
            # Add document separator if multiple documents
            if len(documents) > 1:
                page_num = doc.metadata.get("page_number", i + 1)
                combined_parts.append(f"\n<!-- Page {page_num} -->\n")
            
            combined_parts.append(content)
        
        return "\n\n".join(combined_parts)
    
    def _analyze_document_structure_with_timeout(self, content: str) -> Dict[str, Any]:
        """Analyze document structure using LLM with timeout handling"""
        try:
            # Truncate content if too long
            analysis_content = content[:MAX_CHARS_FOR_LLM] if len(content) > MAX_CHARS_FOR_LLM else content
            
            prompt = f"""Analyze this document and identify its structure for Markdown conversion.
Focus on: headings, sections, lists, tables, and logical flow.

Document content (first {len(analysis_content)} characters):
{analysis_content}

Respond with JSON only:
{{
    "document_type": "report/article/manual/presentation",
    "main_sections": [
        {{"title": "section name", "level": 1, "type": "heading"}},
        {{"title": "subsection", "level": 2, "type": "heading"}}
    ],
    "structural_elements": {{
        "headings": 3,
        "paragraphs": 8,
        "lists": 2,
        "tables": 1
    }},
    "conversion_strategy": "preserve_structure/flatten/reorganize"
}}"""

            response = self.llm.invoke(prompt)
            
            try:
                import json
                analysis = json.loads(response)
                logger.info("LLM structure analysis completed successfully")
                return analysis
            except json.JSONDecodeError:
                logger.warning("LLM returned invalid JSON, using fallback")
                return self._simple_structure_analysis(content)
                
        except Exception as e:
            logger.error(f"LLM structure analysis failed: {e}")
            raise
    
    def _simple_structure_analysis(self, content: str) -> Dict[str, Any]:
        """Simple regex-based structure analysis as fallback"""
        analysis = {
            "document_type": "text",
            "main_sections": [],
            "structural_elements": {
                "headings": 0,
                "paragraphs": 0,
                "lists": 0,
                "tables": 0
            },
            "conversion_strategy": "preserve_structure"
        }
        
        # Count structural elements using regex
        headings = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        analysis["structural_elements"]["headings"] = len(headings)
        
        # Add headings to main_sections
        for match in headings:
            level = len(match[0])
            title = match[1]
            analysis["main_sections"].append({
                "title": title,
                "level": level,
                "type": "heading"
            })
        
        # Count paragraphs (double newline separated)
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        analysis["structural_elements"]["paragraphs"] = len(paragraphs)
        
        # Count lists
        list_items = re.findall(r'^[\s]*[-*+]\s+|^\s*\d+\.\s+', content, re.MULTILINE)
        analysis["structural_elements"]["lists"] = len(list_items)
        
        # Count tables
        table_rows = re.findall(r'\|.*\|', content)
        analysis["structural_elements"]["tables"] = len(table_rows)
        
        logger.info(f"Simple structure analysis: {analysis['structural_elements']}")
        return analysis
    
    def _analyze_document_structure(self, content: str) -> Dict[str, Any]:
        """Analyze document structure for Markdown conversion"""
        try:
            # Check if content is too short for LLM analysis
            if len(content.strip()) < 100:
                logger.info("Content too short for LLM analysis, using fallback")
                return self._fallback_structure_analysis(content)
            
            prompt = f"""Analyze this document content and identify its structure for conversion to clean Markdown format.

Document content (first 3000 characters):
{content[:3000]}

Identify:
1. Document hierarchy (headings, sections, subsections)
2. Tables and their structure
3. Lists (ordered/unordered)
4. Code blocks or technical content
5. Quotes or callouts
6. Overall document type and purpose

Provide a JSON response with this structure:
{{
    "document_type": "type of document (report, manual, article, etc.)",
    "title": "main document title if identifiable",
    "structural_elements": {{
        "headings": number,
        "tables": number,
        "lists": number,
        "code_blocks": number,
        "quotes": number
    }},
    "hierarchy": [
        {{
            "level": 1,
            "title": "Main Section",
            "type": "heading",
            "content_preview": "brief preview"
        }}
    ],
    "formatting_notes": "special formatting requirements or observations"
}}

Respond only with valid JSON."""

            # Set a timeout for LLM calls
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("LLM call timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout
            
            try:
                response = self.llm.invoke(prompt)
                signal.alarm(0)  # Cancel timeout
                
                try:
                    analysis = json.loads(response)
                    return analysis
                except json.JSONDecodeError:
                    logger.warning("Failed to parse LLM structure analysis, using fallback")
                    return self._fallback_structure_analysis(content)
                    
            except TimeoutError:
                logger.warning("LLM structure analysis timed out, using fallback")
                return self._fallback_structure_analysis(content)
            finally:
                signal.alarm(0)  # Ensure timeout is cancelled
                
        except Exception as e:
            logger.warning(f"Structure analysis failed ({e}), using fallback")
            return self._fallback_structure_analysis(content)
    
    def _fallback_structure_analysis(self, content: str) -> Dict[str, Any]:
        """Fallback structure analysis using pattern matching"""
        analysis = {
            "document_type": "text_document",
            "title": "Document",
            "structural_elements": {
                "headings": 0,
                "tables": 0,
                "lists": 0,
                "code_blocks": 0,
                "quotes": 0
            },
            "hierarchy": [],
            "formatting_notes": "Fallback analysis using pattern matching"
        }
        
        # Count structural elements
        analysis["structural_elements"]["headings"] = len(re.findall(r'^#{1,6}\s+', content, re.MULTILINE))
        analysis["structural_elements"]["tables"] = len(re.findall(r'\|.*\|', content))
        analysis["structural_elements"]["lists"] = len(re.findall(r'^[\s]*[-*+]\s+|^\s*\d+\.\s+', content, re.MULTILINE))
        analysis["structural_elements"]["code_blocks"] = len(re.findall(r'```[\s\S]*?```', content))
        analysis["structural_elements"]["quotes"] = len(re.findall(r'^>\s+', content, re.MULTILINE))
        
        return analysis
    
    def _convert_to_structured_markdown(self, content: str, structure: Dict[str, Any]) -> str:
        """Convert content to structured Markdown using LLM"""
        try:
            # Check if content is too short or if we should skip LLM
            if len(content.strip()) < 100:
                logger.info("Content too short for LLM conversion, using fallback")
                return self._fallback_markdown_conversion(content)
            
            document_type = structure.get("document_type", "document")
            title = structure.get("title", "Document")
            
            prompt = f"""Convert this document content to clean, well-structured Markdown format.

Document Type: {document_type}
Title: {title}

Original Content:
{content}

Requirements:
1. Create proper Markdown hierarchy with # ## ### headings
2. Convert tables to Markdown table format with | separators
3. Format lists as proper Markdown lists (- or 1.)
4. Preserve code blocks with ``` fencing
5. Use > for quotes and callouts
6. Maintain logical document flow and structure
7. Ensure all content is preserved but properly formatted
8. Add a clear document title at the top
9. Use proper spacing between sections

Output only the clean Markdown content, no explanations."""

            # Set timeout for LLM conversion
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("LLM conversion timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  # 60 second timeout for conversion
            
            try:
                markdown_response = self.llm.invoke(prompt)
                signal.alarm(0)  # Cancel timeout
                return markdown_response.strip()
            except TimeoutError:
                logger.warning("LLM markdown conversion timed out, using fallback")
                return self._fallback_markdown_conversion(content)
            finally:
                signal.alarm(0)  # Ensure timeout is cancelled
            
        except Exception as e:
            logger.warning(f"LLM markdown conversion failed ({e}), using fallback")
            return self._fallback_markdown_conversion(content)
    
    def _fallback_markdown_conversion(self, content: str) -> str:
        """Fallback Markdown conversion using pattern-based approach"""
        logger.info("Using fallback Markdown conversion")
        
        # Basic Markdown conversion
        lines = content.split('\n')
        markdown_lines = []
        
        in_table = False
        table_headers_added = False
        
        for line in lines:
            line = line.strip()
            if not line:
                markdown_lines.append("")
                in_table = False
                table_headers_added = False
                continue
            
            # Convert tables
            if '|' in line and len(line.split('|')) > 2:
                if not in_table:
                    in_table = True
                    table_headers_added = False
                
                # Clean up table row
                cells = [cell.strip() for cell in line.split('|')]
                if cells[0] == '':
                    cells = cells[1:]
                if cells[-1] == '':
                    cells = cells[:-1]
                
                markdown_lines.append('| ' + ' | '.join(cells) + ' |')
                
                # Add separator after header
                if not table_headers_added:
                    separator = '| ' + ' | '.join(['---'] * len(cells)) + ' |'
                    markdown_lines.append(separator)
                    table_headers_added = True
            
            # Convert lists
            elif re.match(r'^[\s]*[-*+]\s+', line):
                markdown_lines.append(re.sub(r'^[\s]*[-*+]\s+', '- ', line))
                in_table = False
            
            elif re.match(r'^\s*\d+\.\s+', line):
                markdown_lines.append(re.sub(r'^\s*\d+\.\s+', '1. ', line))
                in_table = False
            
            # Convert headings (if not already Markdown)
            elif not line.startswith('#') and len(line) < 100 and line.isupper():
                markdown_lines.append(f"## {line.title()}")
                in_table = False
            
            # Regular content
            else:
                markdown_lines.append(line)
                in_table = False
        
        # Add title if not present
        markdown_content = '\n'.join(markdown_lines)
        if not markdown_content.startswith('#'):
            markdown_content = "# Document\n\n" + markdown_content
        
        return markdown_content
    
    def _clean_and_validate_markdown(self, markdown: str) -> str:
        """Clean and validate the generated Markdown"""
        lines = markdown.split('\n')
        cleaned_lines = []
        
        prev_line_empty = False
        
        for line in lines:
            # Remove excessive whitespace
            line = line.rstrip()
            
            # Handle empty lines (max 2 consecutive)
            if not line:
                if not prev_line_empty:
                    cleaned_lines.append("")
                    prev_line_empty = True
                continue
            else:
                prev_line_empty = False
            
            # Fix heading spacing
            if line.startswith('#'):
                # Ensure space after #
                line = re.sub(r'^(#+)([^\s])', r'\1 \2', line)
                # Ensure proper heading levels (max 6)
                if line.startswith('#######'):
                    line = '######' + line[7:]
            
            # Fix table formatting
            elif '|' in line:
                # Ensure proper table spacing
                line = re.sub(r'\|\s*([^|]+)\s*', r'| \1 ', line)
                line = line.rstrip() + ' |' if not line.endswith('|') else line
            
            # Fix list formatting
            elif re.match(r'^[\s]*[-*+]\s+', line):
                line = re.sub(r'^[\s]*[-*+]\s+', '- ', line)
            
            elif re.match(r'^\s*\d+\.\s+', line):
                line = re.sub(r'^\s*(\d+)\.\s+', r'\1. ', line)
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _assess_conversion_quality(self, markdown: str) -> Dict[str, Any]:
        """Assess the quality of Markdown conversion"""
        quality = {
            "has_title": markdown.strip().startswith('#'),
            "heading_count": len(re.findall(r'^#{1,6}\s+', markdown, re.MULTILINE)),
            "table_count": len(re.findall(r'^\|.*\|$', markdown, re.MULTILINE)),
            "list_count": len(re.findall(r'^[-*+]\s+|^\d+\.\s+', markdown, re.MULTILINE)),
            "code_block_count": len(re.findall(r'```[\s\S]*?```', markdown)),
            "proper_spacing": not re.search(r'\n{4,}', markdown),  # No excessive spacing
            "valid_tables": self._validate_markdown_tables(markdown),
            "overall_score": 0.0
        }
        
        # Calculate overall score
        score_factors = [
            quality["has_title"],
            quality["heading_count"] > 0,
            quality["proper_spacing"],
            quality["valid_tables"],
            len(markdown) > 100  # Reasonable content length
        ]
        
        quality["overall_score"] = sum(score_factors) / len(score_factors) * 100
        
        return quality
    
    def _validate_markdown_tables(self, markdown: str) -> bool:
        """Validate that Markdown tables are properly formatted"""
        table_lines = re.findall(r'^\|.*\|$', markdown, re.MULTILINE)
        
        if not table_lines:
            return True  # No tables to validate
        
        # Check if tables have consistent column counts
        for i in range(0, len(table_lines), 2):  # Check header-separator pairs
            if i + 1 < len(table_lines):
                header_cols = len(table_lines[i].split('|')) - 2
                separator_cols = len(table_lines[i + 1].split('|')) - 2
                
                if header_cols != separator_cols or header_cols < 1:
                    return False
        
        return True


class SOTAMarkdownChunkingService:
    """SOTA Markdown-based chunking service for optimal RAG performance"""
    
    def __init__(self):
        self.markdown_converter = StructureToMarkdownConverter()
        logger.info("Initialized SOTA Markdown chunking service")
    
    def process_documents_to_markdown(self, documents: List[LangChainDocument]) -> MarkdownConversionResult:
        """Convert documents to Markdown format for optimal chunking"""
        return self.markdown_converter.convert_to_markdown(documents)
    
    def create_markdown_document(self, markdown_content: str, original_metadata: Dict[str, Any]) -> LangChainDocument:
        """Create a LangChain document with Markdown content"""
        # Enhance metadata for Markdown format
        enhanced_metadata = original_metadata.copy()
        enhanced_metadata.update({
            "content_format": "markdown",
            "conversion_method": "sota_markdown",
            "optimized_for_rag": True,
            "structure_preserved": True
        })
        
        return LangChainDocument(
            page_content=markdown_content,
            metadata=enhanced_metadata
        )
    
    def get_conversion_statistics(self, result: MarkdownConversionResult) -> Dict[str, Any]:
        """Get detailed statistics about the Markdown conversion"""
        if not result.success:
            return {"conversion_failed": True, "error": result.error_message}
        
        stats = {
            "conversion_success": True,
            "markdown_length": len(result.markdown_content),
            "original_documents": result.conversion_metadata.get("original_documents", 0),
            "structure_elements": result.conversion_metadata.get("structure_elements", {}),
            "conversion_quality": result.conversion_metadata.get("conversion_quality", {}),
            "rag_optimization_score": self._calculate_rag_score(result.markdown_content)
        }
        
        return stats
    
    def _calculate_rag_score(self, markdown: str) -> float:
        """Calculate RAG optimization score for Markdown content"""
        score_factors = []
        
        # Structure factors
        score_factors.append(1.0 if markdown.startswith('#') else 0.0)  # Has title
        score_factors.append(min(len(re.findall(r'^#{1,6}\s+', markdown, re.MULTILINE)) / 5, 1.0))  # Heading density
        score_factors.append(1.0 if re.search(r'^\|.*\|$', markdown, re.MULTILINE) else 0.0)  # Has tables
        score_factors.append(1.0 if re.search(r'^[-*+]\s+|^\d+\.\s+', markdown, re.MULTILINE) else 0.0)  # Has lists
        
        # Content quality factors
        score_factors.append(min(len(markdown) / 1000, 1.0))  # Content length
        score_factors.append(1.0 if not re.search(r'\n{4,}', markdown) else 0.0)  # Proper spacing
        score_factors.append(1.0 if len(markdown.split()) > 50 else 0.0)  # Sufficient content
        
        return sum(score_factors) / len(score_factors) * 100


# Global SOTA Markdown chunking service
sota_markdown_chunking_service = SOTAMarkdownChunkingService()