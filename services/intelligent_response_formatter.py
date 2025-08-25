"""
Intelligent Response Formatter for Kurachi AI
Implements dual-model approach using llava:7b for query intent analysis 
and qwen3:4b for formatted response generation with Streamlit components
"""
import json
import re
import time
import requests
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_community.llms import Ollama

from config import config
from utils.logger import get_logger

logger = get_logger("intelligent_response_formatter")


# Unified rule-first intent classification (used by both main and fallback paths)
_INTENT_RULES = [
    (r'\b(table|tabular|tabulate|columns?|grid|matrix|in\s+tabular\s+format|table\s+of)\b', 'table'),
    (r'\b(graph|chart|plot|visuali[sz]e|bar\s+chart|line\s+chart|pie\s+chart|trend\s+chart)\b', 'chart'),
    (r'\b(compare|vs\.?|versus|difference|side\s+by\s+side)\b', 'comparison'),
    (r'\b(how\s*to|steps?\b|install|setup|set\s*up|walk(?:\s*me)?\s*through|guide|procedure)\b', 'process_flow'),
    (r'\b(summary|overview|brief|tl;dr|executive\s+summary)\b', 'summary'),
    (r'\b(code|snippet|syntax|example)\b', 'code_block'),
    (r'\b(list|bullets?|enumerate)\b', 'list'),
]


def robust_json_parse(s: str) -> dict:
    """Robust JSON parser that handles LLM output issues"""
    # 1) Quick path - try direct parsing
    try:
        return json.loads(s)
    except Exception:
        pass
    
    # 2) Extract the biggest {...} block
    m = re.search(r'\{.*\}', s, re.DOTALL)
    if m:
        candidate = m.group(0)
        # Normalize common bad escapes: single backslashes before non-escapables
        candidate = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', candidate)
        # Remove trailing commas in objects/arrays
        candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
        try:
            return json.loads(candidate)
        except Exception:
            pass
    
    # 3) Last resort: try json5 if available
    try:
        import json5
        return json5.loads(s)
    except Exception:
        pass
    
    return {}


def rule_first_guess(query: str) -> Optional[str]:
    """Fast rule-based intent classification"""
    query_lower = query.lower()
    for pattern, label in _INTENT_RULES:
        if re.search(pattern, query_lower):
            return label
    return None


def strip_think(s: str) -> str:
    """Remove chain-of-thought markers from output"""
    return re.sub(r'<think>.*?</think>\s*', '', s, flags=re.DOTALL)


def map_intent_to_response_type(query: str, intent_type: str) -> str:
    """Smart intent mapping with tie-breaker logic"""
    query_lower = query.lower()
    
    # Prefer table when comparison + table keywords both present
    if intent_type == "comparison":
        if re.search(r'\b(table|tabular|side\s+by\s+side)\b', query_lower):
            return "table"
    
    return intent_type


def has_numeric_series(data: Dict[str, Any]) -> bool:
    """Check if data contains valid numeric series suitable for charting"""
    try:
        if not data:
            return False
            
        values = data.get('values', [])
        labels = data.get('labels', [])
        
        # Must have matching labels and values
        if len(labels) != len(values) or len(values) < 2:
            return False
        
        # Values must be numeric and not None/zero-only
        numeric_values = []
        for v in values:
            if v is None:
                return False
            try:
                num_val = float(v)
                numeric_values.append(num_val)
            except (ValueError, TypeError):
                return False
        
        # Must have at least 2 non-zero values or meaningful variation
        non_zero_count = sum(1 for v in numeric_values if v != 0)
        return non_zero_count >= 2
        
    except Exception:
        return False


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    """Generate clean markdown table with proper formatting"""
    if not headers or not rows:
        return ""
    
    # Headers with bold formatting
    head = "| " + " | ".join(f"**{h}**" for h in headers) + " |"
    # Separator row
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    # Data rows
    body = "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)
    
    return head + "\n" + sep + "\n" + body


class ResponseType(Enum):
    """Types of response formatting"""
    TABLE = "table"
    CHART = "chart"
    LIST = "list"
    CODE_BLOCK = "code_block"
    STRUCTURED_TEXT = "structured_text"
    COMPARISON = "comparison"
    PROCESS_FLOW = "process_flow"
    SUMMARY = "summary"
    PLAIN_TEXT = "plain_text"


class ChartType(Enum):
    """Types of charts for data visualization"""
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    PIE_CHART = "pie_chart"
    SCATTER_PLOT = "scatter_plot"
    HISTOGRAM = "histogram"
    AREA_CHART = "area_chart"


@dataclass
class QueryIntent:
    """Query intent analysis result"""
    response_type: ResponseType
    chart_type: Optional[ChartType] = None
    data_structure_needed: bool = False
    comparison_requested: bool = False
    numerical_data_expected: bool = False
    process_flow_requested: bool = False
    confidence: float = 0.0
    reasoning: str = ""
    formatting_hints: List[str] = None
    
    def __post_init__(self):
        if self.formatting_hints is None:
            self.formatting_hints = []


@dataclass
class FormattedResponse:
    """Formatted response with metadata"""
    content: str
    response_type: ResponseType
    chart_data: Optional[Dict[str, Any]] = None
    table_data: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = None
    streamlit_components: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.streamlit_components is None:
            self.streamlit_components = []


class IntelligentResponseFormatter:
    """Main service for intelligent response formatting with dual-model approach"""
    
    def __init__(self):
        # Initialize both models
        self.vision_model = Ollama(
            model=config.ai.vision_model,  # llava:7b
            base_url=config.ai.ollama_base_url,
            temperature=0.1  # Low temperature for consistent analysis
        )
        
        self.text_model = Ollama(
            model=config.ai.llm_model,  # qwen3:4b
            base_url=config.ai.ollama_base_url,
            temperature=0.3  # Slightly higher for creative formatting
        )
        
        # Response type templates
        self.response_templates = self._initialize_templates()
        
    def _initialize_templates(self) -> Dict[ResponseType, str]:
        """Initialize response formatting templates"""
        return {
            ResponseType.TABLE: """
Format this information as a well-structured table with clear headers and organized data.
Use markdown table format with proper alignment.
Include relevant columns and ensure data is properly categorized.
""",
            ResponseType.CHART: """
Format this information for chart visualization.
Identify the key data points, categories, and values.
Structure the response to highlight numerical relationships and trends.
""",
            ResponseType.LIST: """
Format this information as a clear, organized list.
Use bullet points or numbered lists as appropriate.
Group related items together and maintain logical hierarchy.
""",
            ResponseType.CODE_BLOCK: """
Format this information as code or technical content.
Use appropriate syntax highlighting and code block formatting.
Include comments and explanations where helpful.
""",
            ResponseType.STRUCTURED_TEXT: """
Format this information with clear structure using headings, subheadings, and sections.
Use markdown formatting to create a well-organized document.
Include emphasis and formatting to highlight key points.
""",
            ResponseType.COMPARISON: """
Format this information as a comparison between different items or options.
Use tables, side-by-side formatting, or structured lists to highlight differences.
Include pros/cons or feature comparisons where relevant.
""",
            ResponseType.PROCESS_FLOW: """
Format this information as a step-by-step process or workflow.
Use numbered steps, clear transitions, and logical flow.
Include any decision points or branching paths.
""",
            ResponseType.SUMMARY: """
Format this information as a concise summary with key points highlighted.
Use bullet points for main takeaways and brief explanations.
Structure with clear sections for different aspects.
"""
        }
    
    def analyze_query_intent(self, query: str, context: str = "") -> QueryIntent:
        """Analyze query intent with unified rule-first approach and smart mapping"""
        try:
            # 1) Rule-first fast path - catches obvious patterns instantly
            rule_guess = rule_first_guess(query)
            if rule_guess:
                # Apply smart intent mapping (handles comparison + table tie-breaker)
                final_type = map_intent_to_response_type(query, rule_guess)
                
                return QueryIntent(
                    response_type=ResponseType(final_type),
                    chart_type=ChartType.BAR_CHART if final_type == "chart" else None,
                    data_structure_needed=final_type in ["table", "chart"],
                    comparison_requested=rule_guess == "comparison",  # Keep original for metadata
                    numerical_data_expected=final_type in ["table", "chart"],
                    process_flow_requested=final_type == "process_flow",
                    confidence=0.9,
                    reasoning=f"Matched keyword rule for {rule_guess}" + (f" → mapped to {final_type}" if final_type != rule_guess else ""),
                    formatting_hints=[f"Rule-based classification: {final_type}"]
                )
            
            # 2) Fallback to LLM with strict JSON schema
            intent_data = self._call_llava_intent_analysis(query, context)
            
            if intent_data:
                # Map string values to enums with smart mapping
                llm_type = intent_data.get("type", "structured_text")
                final_type = map_intent_to_response_type(query, llm_type)
                response_type = ResponseType(final_type)
                
                chart_type = None
                if response_type == ResponseType.CHART:
                    chart_type = ChartType.BAR_CHART  # Default chart type
                
                return QueryIntent(
                    response_type=response_type,
                    chart_type=chart_type,
                    data_structure_needed=response_type in [ResponseType.TABLE, ResponseType.CHART],
                    comparison_requested=llm_type == "comparison",  # Keep original for metadata
                    numerical_data_expected=response_type in [ResponseType.TABLE, ResponseType.CHART],
                    process_flow_requested=response_type == ResponseType.PROCESS_FLOW,
                    confidence=float(intent_data.get("confidence", 0.7)),
                    reasoning=intent_data.get("reason", "LLM classification") + (f" → mapped to {final_type}" if final_type != llm_type else ""),
                    formatting_hints=[]
                )
            
            # 3) Final fallback - use unified logic
            return self._unified_fallback_analysis(query)
                
        except Exception as e:
            logger.error(f"Failed to analyze query intent: {e}")
            return self._unified_fallback_analysis(query)
    
    def _call_llava_intent_analysis(self, query: str, context: str = "") -> Optional[Dict[str, Any]]:
        """Call llava with strict JSON schema for intent analysis"""
        try:
            # Minimal, unambiguous prompt
            intent_prompt = f"""You classify the user query into ONE of: ["table","chart","list","comparison","process_flow","summary","code_block","structured_text"].

Return STRICT JSON ONLY:
{{"type":"<one>","confidence":0.0-1.0,"reason":"<short>"}}

Rules:
- "table", "tabular", "grid", "columns" => type=table
- "chart","graph","plot","bar","line","pie","visualize" => type=chart
- "compare","vs","difference","side by side" => type=comparison
- "how to","steps","install","setup","walk me through" => type=process_flow
- "summary","overview","brief","tl;dr","executive summary" => type=summary
- "code","snippet","syntax","example" => type=code_block
- "list","bullet","bulleted","enumerate" => type=list
- Else => type=structured_text

Query: "{query}"
Context: {context}"""
            
            # Use Ollama API directly with format="json" for strict JSON
            try:
                payload = {
                    "model": config.ai.vision_model,
                    "prompt": intent_prompt,
                    "options": {"temperature": 0, "num_ctx": 2048},
                    "format": "json"
                }
                
                response = requests.post(
                    f"{config.ai.ollama_base_url}/api/generate",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    raw_response = response.json().get("response", "")
                    return robust_json_parse(raw_response)
                    
            except Exception as e:
                logger.warning(f"Direct Ollama API call failed: {e}")
                # Fallback to LangChain
                response = self.vision_model.invoke(intent_prompt)
                return robust_json_parse(response)
                
        except Exception as e:
            logger.warning(f"LLaVA intent analysis failed: {e}")
            return None
    

    
    def _unified_fallback_analysis(self, query: str) -> QueryIntent:
        """Unified fallback using the same rule-first logic"""
        # Use the same rule_first_guess logic for consistency
        rule_guess = rule_first_guess(query)
        
        if rule_guess:
            # Apply smart intent mapping
            final_type = map_intent_to_response_type(query, rule_guess)
            
            return QueryIntent(
                response_type=ResponseType(final_type),
                chart_type=ChartType.BAR_CHART if final_type == "chart" else None,
                data_structure_needed=final_type in ["table", "chart"],
                comparison_requested=rule_guess == "comparison",
                numerical_data_expected=final_type in ["table", "chart"],
                process_flow_requested=final_type == "process_flow",
                confidence=0.7,  # Slightly lower confidence for fallback
                reasoning=f"Fallback matched keyword rule for {rule_guess}" + (f" → mapped to {final_type}" if final_type != rule_guess else ""),
                formatting_hints=[f"Fallback rule-based classification: {final_type}"]
            )
        else:
            # Ultimate fallback
            return QueryIntent(
                response_type=ResponseType.STRUCTURED_TEXT,
                confidence=0.5,
                reasoning="Default structured text format (no rules matched)",
                formatting_hints=["Ultimate fallback"]
            )
    
    def generate_formatted_response(
        self, 
        query: str, 
        raw_response: str, 
        intent: QueryIntent,
        sources: List[Dict[str, Any]] = None
    ) -> FormattedResponse:
        """Generate formatted response using qwen3:4b with enhanced prompts"""
        try:
            # Get the appropriate template
            template = self.response_templates.get(intent.response_type, "")
            
            # Create enhanced formatting prompt
            formatting_prompt = f"""
You are an expert at formatting information for business users. 

Original Query: "{query}"
Raw Information: {raw_response}
Requested Format: {intent.response_type.value}
Formatting Instructions: {template}

Additional Context:
- Response Type: {intent.response_type.value}
- Chart Type Needed: {intent.chart_type.value if intent.chart_type else "None"}
- Data Structure Needed: {intent.data_structure_needed}
- Comparison Requested: {intent.comparison_requested}
- Numerical Data Expected: {intent.numerical_data_expected}
- Process Flow Requested: {intent.process_flow_requested}
- Formatting Hints: {', '.join(intent.formatting_hints)}

Please format the information according to the requested type. Use proper markdown formatting:
- **Bold** for headings and important terms
- Tables with | separators and proper alignment
- Numbered lists for processes (1. 2. 3.)
- Bullet points for features/items (• or -)
- Code blocks with ```language``` for technical content
- Clear section headers with ## or ###

Make the response professional, clear, and easy to scan. Focus on the user's specific request.

Formatted Response:
"""
            
            # Generate formatted response using text model
            formatted_content = self.text_model.invoke(formatting_prompt)
            
            # Strip chain-of-thought markers
            formatted_content = strip_think(formatted_content)
            
            # Post-process the response
            processed_content = self._post_process_response(formatted_content, intent)
            
            # Extract data for charts/tables if needed
            chart_data = None
            table_data = None
            streamlit_components = []
            
            if intent.response_type == ResponseType.TABLE:
                table_data = self._extract_table_data(processed_content)
                if table_data is not None:
                    streamlit_components.append({
                        "type": "dataframe",
                        "data": table_data.to_dict(),
                        "use_container_width": True
                    })
            
            elif intent.response_type == ResponseType.CHART and intent.chart_type:
                chart_data = self._extract_chart_data(processed_content, intent.chart_type)
                if chart_data and has_numeric_series(chart_data):
                    streamlit_components.append({
                        "type": "chart",
                        "chart_type": intent.chart_type.value,
                        "data": chart_data
                    })
                else:
                    # Fallback to list if chart data is insufficient
                    logger.info("Chart data insufficient, falling back to list format")
                    intent.response_type = ResponseType.LIST
            
            return FormattedResponse(
                content=processed_content,
                response_type=intent.response_type,
                chart_data=chart_data,
                table_data=table_data,
                streamlit_components=streamlit_components,
                metadata={
                    "intent_confidence": intent.confidence,
                    "intent_reasoning": intent.reasoning,
                    "formatting_applied": True,
                    "processing_time": time.time(),
                    "sources_count": len(sources) if sources else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to generate formatted response: {e}")
            # Return basic formatted response as fallback
            return FormattedResponse(
                content=raw_response,
                response_type=ResponseType.PLAIN_TEXT,
                metadata={"formatting_failed": True, "error": str(e)}
            )
    
    def _post_process_response(self, content: str, intent: QueryIntent) -> str:
        """Post-process response content for better formatting"""
        try:
            # Clean up common formatting issues
            content = re.sub(r'\n{3,}', '\n\n', content)  # Remove excessive line breaks
            content = re.sub(r'^\s+', '', content, flags=re.MULTILINE)  # Remove leading whitespace
            
            # Enhance markdown formatting based on response type
            if intent.response_type == ResponseType.LIST:
                content = self._enhance_list_formatting(content)
            elif intent.response_type == ResponseType.TABLE:
                content = self._enhance_table_formatting(content)
            elif intent.response_type == ResponseType.CODE_BLOCK:
                content = self._enhance_code_formatting(content)
            elif intent.response_type == ResponseType.STRUCTURED_TEXT:
                content = self._enhance_structured_text_formatting(content)
            
            return content.strip()
            
        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")
            return content
    
    def _enhance_list_formatting(self, content: str) -> str:
        """Enhance list formatting with proper bullets and indentation"""
        lines = content.split('\n')
        enhanced_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                # Check if it looks like a list item
                if re.match(r'^\w+[:\-]', line) or line.startswith(('The ', 'A ', 'An ')):
                    line = f"• {line}"
            enhanced_lines.append(line)
        
        return '\n'.join(enhanced_lines)
    
    def _enhance_table_formatting(self, content: str) -> str:
        """Enhance table formatting with proper markdown tables"""
        # Look for table-like content and format it properly
        lines = content.split('\n')
        enhanced_lines = []
        in_table = False
        
        for line in lines:
            # Detect potential table rows
            if '|' in line and line.count('|') >= 2:
                if not in_table:
                    in_table = True
                    # Add table header separator if missing
                    if '---' not in line:
                        enhanced_lines.append(line)
                        # Create separator row
                        parts = line.split('|')
                        separator = '|' + '|'.join(['---' for _ in range(len(parts)-2)]) + '|'
                        enhanced_lines.append(separator)
                        continue
                enhanced_lines.append(line)
            else:
                in_table = False
                enhanced_lines.append(line)
        
        return '\n'.join(enhanced_lines)
    
    def _enhance_code_formatting(self, content: str) -> str:
        """Enhance code formatting with proper code blocks"""
        # Wrap code-like content in proper markdown code blocks
        if '```' not in content:
            # Detect code patterns
            code_patterns = [
                r'def \w+\(',
                r'function \w+\(',
                r'class \w+',
                r'import \w+',
                r'from \w+ import',
                r'<\w+[^>]*>',
                r'\{\s*\w+:',
            ]
            
            for pattern in code_patterns:
                if re.search(pattern, content):
                    content = f"```\n{content}\n```"
                    break
        
        return content
    
    def _enhance_structured_text_formatting(self, content: str) -> str:
        """Enhance structured text with proper headings and formatting"""
        lines = content.split('\n')
        enhanced_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Convert potential headings
                if line.endswith(':') and len(line.split()) <= 5:
                    line = f"## {line[:-1]}"
                elif line.isupper() and len(line.split()) <= 3:
                    line = f"### {line.title()}"
                # Emphasize key terms
                line = re.sub(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b(?=\s*:)', r'**\1**', line)
            
            enhanced_lines.append(line)
        
        return '\n'.join(enhanced_lines)
    
    def _extract_table_data(self, content: str) -> Optional[pd.DataFrame]:
        """Extract table data from formatted content"""
        try:
            # Look for markdown tables
            table_pattern = r'\|.*\|'
            table_lines = [line for line in content.split('\n') if re.match(table_pattern, line)]
            
            if len(table_lines) >= 2:  # At least header and one data row
                # Parse the table
                headers = [col.strip() for col in table_lines[0].split('|')[1:-1]]
                
                # Skip separator row (usually contains ---)
                data_rows = []
                for line in table_lines[1:]:
                    if '---' not in line:
                        row = [col.strip() for col in line.split('|')[1:-1]]
                        if len(row) == len(headers):
                            data_rows.append(row)
                
                if data_rows:
                    return pd.DataFrame(data_rows, columns=headers)
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to extract table data: {e}")
            return None
    
    def _extract_chart_data(self, content: str, chart_type: ChartType) -> Optional[Dict[str, Any]]:
        """Extract chart data from formatted content with improved validation"""
        try:
            # Look for structured data patterns (month: value, Q1: value, etc.)
            structured_patterns = [
                r'(\w+(?:\s+\w+)?)\s*:\s*\$?(\d+(?:,\d{3})*(?:\.\d+)?)',  # Month: $1,000
                r'(\w+(?:\s+\w+)?)\s+(\d+(?:,\d{3})*(?:\.\d+)?)',         # Month 1000
                r'(\w+)\s*\|\s*(\d+(?:,\d{3})*(?:\.\d+)?)',               # Month | 1000
            ]
            
            labels = []
            values = []
            
            for pattern in structured_patterns:
                matches = re.findall(pattern, content)
                if matches and len(matches) >= 2:
                    for label, value_str in matches:
                        try:
                            # Clean and convert value
                            clean_value = value_str.replace(',', '').replace('$', '')
                            value = float(clean_value)
                            if value > 0:  # Only include positive values
                                labels.append(label.strip())
                                values.append(value)
                        except ValueError:
                            continue
                    break
            
            # Fallback: look for any numerical data
            if len(values) < 2:
                numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', content)
                if len(numbers) >= 2:
                    try:
                        values = [float(n.replace(',', '')) for n in numbers[:8] if float(n.replace(',', '')) > 0]
                        labels = [f"Data Point {i+1}" for i in range(len(values))]
                    except ValueError:
                        return None
            
            # Validate the extracted data
            if len(values) >= 2 and len(labels) == len(values):
                chart_data = {
                    "labels": labels,
                    "values": values,
                    "chart_type": chart_type.value,
                    "title": "Data Visualization"
                }
                
                # Final validation using has_numeric_series
                if has_numeric_series(chart_data):
                    return chart_data
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to extract chart data: {e}")
            return None
    
    def format_response_with_streamlit_components(
        self, 
        query: str, 
        raw_response: str, 
        sources: List[Dict[str, Any]] = None
    ) -> FormattedResponse:
        """Main method to format response with Streamlit components"""
        try:
            # Step 1: Analyze query intent using llava:7b
            intent = self.analyze_query_intent(query)
            
            logger.info(f"Query intent analysis: {intent.response_type.value} (confidence: {intent.confidence:.2f})")
            
            # Step 2: Generate formatted response using qwen3:4b
            formatted_response = self.generate_formatted_response(query, raw_response, intent, sources)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Failed to format response with Streamlit components: {e}")
            # Return basic response as fallback
            return FormattedResponse(
                content=raw_response,
                response_type=ResponseType.PLAIN_TEXT,
                metadata={"formatting_failed": True, "error": str(e)}
            )


# Global instance
intelligent_response_formatter = IntelligentResponseFormatter()