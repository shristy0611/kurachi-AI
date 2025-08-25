"""
Source Attribution Service for Kurachi AI
Handles detailed source citations, confidence scoring, and fact-checking
"""
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import re
import hashlib
import json

from langchain.schema import Document
from langchain_community.llms import Ollama

from config import config
from utils.logger import get_logger

logger = get_logger("source_attribution")


@dataclass
class SourceCitation:
    """Enhanced source citation with detailed attribution"""
    document_id: str
    filename: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    excerpt: str = ""
    relevance_score: float = 0.0
    confidence_score: float = 0.0
    citation_type: str = "direct"  # direct, paraphrase, inference
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "section": self.section,
            "excerpt": self.excerpt,
            "relevance_score": self.relevance_score,
            "confidence_score": self.confidence_score,
            "citation_type": self.citation_type,
            "metadata": self.metadata
        }


@dataclass
class ResponseAttribution:
    """Complete response attribution with sources and confidence"""
    response_content: str
    overall_confidence: float
    sources: List[SourceCitation]
    fact_check_status: str = "not_checked"  # not_checked, verified, uncertain, conflicting
    synthesis_type: str = "single_source"  # single_source, multi_source, inference
    uncertainty_indicators: List[str] = None
    validation_notes: Optional[str] = None
    
    def __post_init__(self):
        if self.uncertainty_indicators is None:
            self.uncertainty_indicators = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "response_content": self.response_content,
            "overall_confidence": self.overall_confidence,
            "sources": [source.to_dict() for source in self.sources],
            "fact_check_status": self.fact_check_status,
            "synthesis_type": self.synthesis_type,
            "uncertainty_indicators": self.uncertainty_indicators,
            "validation_notes": self.validation_notes
        }


class SourceAttributionService:
    """Service for enhanced source attribution and fact-checking"""
    
    def __init__(self):
        self.llm = Ollama(
            model=config.ai.llm_model,
            base_url=config.ai.ollama_base_url,
            temperature=0.2  # Lower temperature for more consistent attribution
        )
        
        # Configuration for attribution
        self.min_confidence_threshold = 0.3
        self.high_confidence_threshold = 0.8
        self.max_excerpt_length = 300
        self.citation_patterns = self._init_citation_patterns()
    
    def _init_citation_patterns(self) -> Dict[str, str]:
        """Initialize patterns for different citation types"""
        return {
            "direct_quote": r'"([^"]+)"',
            "page_reference": r'(?:page|p\.)\s*(\d+)',
            "section_reference": r'(?:section|chapter|part)\s*([^\s,]+)',
            "figure_reference": r'(?:figure|fig\.)\s*(\d+)',
            "table_reference": r'(?:table|tbl\.)\s*(\d+)'
        }
    
    def enhance_response_with_attribution(
        self, 
        response_content: str, 
        source_documents: List[Document],
        query: str
    ) -> ResponseAttribution:
        """Enhance response with detailed source attribution and multi-document synthesis"""
        try:
            # Process source documents into citations
            citations = self._process_source_documents(source_documents, query, response_content)
            
            # Enhanced multi-document synthesis analysis
            synthesis_analysis = self._analyze_multi_document_synthesis(citations, response_content, query)
            
            # Calculate overall confidence with synthesis factors
            overall_confidence = self._calculate_enhanced_overall_confidence(
                citations, response_content, synthesis_analysis
            )
            
            # Determine synthesis type with enhanced analysis
            synthesis_type = self._determine_enhanced_synthesis_type(citations, synthesis_analysis)
            
            # Perform comprehensive fact-checking
            fact_check_status, validation_notes = self._perform_comprehensive_fact_checking(
                response_content, citations, synthesis_analysis
            )
            
            # Identify uncertainty indicators with context
            uncertainty_indicators = self._identify_enhanced_uncertainty_indicators(
                response_content, citations, synthesis_analysis
            )
            
            # Validate response completeness and accuracy
            validation_results = self._validate_response_quality(
                response_content, citations, query, synthesis_analysis
            )
            
            return ResponseAttribution(
                response_content=response_content,
                overall_confidence=overall_confidence,
                sources=citations,
                fact_check_status=fact_check_status,
                synthesis_type=synthesis_type,
                uncertainty_indicators=uncertainty_indicators,
                validation_notes=validation_notes or validation_results.get("notes")
            )
            
        except Exception as e:
            logger.error("Failed to enhance response with attribution", error=e)
            # Return basic attribution on error
            return ResponseAttribution(
                response_content=response_content,
                overall_confidence=0.5,
                sources=[],
                fact_check_status="error",
                synthesis_type="unknown"
            )
    
    def _analyze_multi_document_synthesis(
        self, 
        citations: List[SourceCitation], 
        response: str, 
        query: str
    ) -> Dict[str, Any]:
        """Analyze how well the response synthesizes information from multiple documents"""
        try:
            analysis = {
                "source_diversity": 0.0,
                "information_integration": 0.0,
                "consistency_score": 0.0,
                "coverage_completeness": 0.0,
                "synthesis_quality": 0.0,
                "conflicting_information": [],
                "information_gaps": [],
                "synthesis_notes": []
            }
            
            if len(citations) <= 1:
                analysis["synthesis_quality"] = 0.3  # Limited synthesis with single source
                return analysis
            
            # 1. Analyze source diversity
            analysis["source_diversity"] = self._calculate_source_diversity(citations)
            
            # 2. Analyze information integration
            analysis["information_integration"] = self._analyze_information_integration(citations, response)
            
            # 3. Check for consistency across sources
            analysis["consistency_score"], analysis["conflicting_information"] = self._check_source_consistency(citations)
            
            # 4. Assess coverage completeness
            analysis["coverage_completeness"] = self._assess_coverage_completeness(citations, query, response)
            
            # 5. Identify information gaps
            analysis["information_gaps"] = self._identify_information_gaps(citations, query, response)
            
            # 6. Calculate overall synthesis quality
            analysis["synthesis_quality"] = (
                analysis["source_diversity"] * 0.2 +
                analysis["information_integration"] * 0.3 +
                analysis["consistency_score"] * 0.3 +
                analysis["coverage_completeness"] * 0.2
            )
            
            return analysis
            
        except Exception as e:
            logger.warning("Failed to analyze multi-document synthesis", error=e)
            return {"synthesis_quality": 0.5}
    
    def _calculate_source_diversity(self, citations: List[SourceCitation]) -> float:
        """Calculate diversity of source documents"""
        try:
            if len(citations) <= 1:
                return 0.0
            
            # Check diversity by filename, document type, and content
            unique_files = set(citation.filename for citation in citations)
            unique_sections = set(citation.section for citation in citations if citation.section)
            
            # Calculate diversity score
            file_diversity = len(unique_files) / len(citations)
            section_diversity = len(unique_sections) / len(citations) if unique_sections else 0.5
            
            # Check content diversity by comparing excerpts
            content_diversity = self._calculate_content_diversity(citations)
            
            return (file_diversity * 0.4 + section_diversity * 0.3 + content_diversity * 0.3)
            
        except Exception as e:
            logger.warning("Failed to calculate source diversity", error=e)
            return 0.5
    
    def _calculate_content_diversity(self, citations: List[SourceCitation]) -> float:
        """Calculate diversity of content across citations"""
        try:
            if len(citations) <= 1:
                return 0.0
            
            excerpts = [citation.excerpt for citation in citations if citation.excerpt]
            if len(excerpts) <= 1:
                return 0.5
            
            # Calculate pairwise similarity and take average
            total_similarity = 0
            pairs = 0
            
            for i in range(len(excerpts)):
                for j in range(i + 1, len(excerpts)):
                    similarity = self._calculate_text_similarity(excerpts[i], excerpts[j])
                    total_similarity += similarity
                    pairs += 1
            
            if pairs == 0:
                return 0.5
            
            avg_similarity = total_similarity / pairs
            # Diversity is inverse of similarity
            return 1.0 - avg_similarity
            
        except Exception as e:
            logger.warning("Failed to calculate content diversity", error=e)
            return 0.5
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception as e:
            return 0.0
    
    def _analyze_information_integration(self, citations: List[SourceCitation], response: str) -> float:
        """Analyze how well information from multiple sources is integrated"""
        try:
            if len(citations) <= 1:
                return 0.3
            
            # Check if response references multiple sources
            source_references = 0
            for citation in citations:
                # Look for content from this source in the response
                if citation.excerpt and len(citation.excerpt) > 20:
                    # Check for key phrases from the excerpt in the response
                    excerpt_words = set(citation.excerpt.lower().split())
                    response_words = set(response.lower().split())
                    
                    overlap = len(excerpt_words.intersection(response_words))
                    if overlap >= 3:  # At least 3 words in common
                        source_references += 1
            
            # Integration score based on how many sources are referenced
            integration_score = min(1.0, source_references / len(citations))
            
            # Bonus for synthesis indicators in response
            synthesis_indicators = [
                "according to", "based on", "multiple sources", "various documents",
                "combining information", "synthesis", "overall", "in summary"
            ]
            
            synthesis_bonus = 0
            response_lower = response.lower()
            for indicator in synthesis_indicators:
                if indicator in response_lower:
                    synthesis_bonus += 0.1
            
            return min(1.0, integration_score + synthesis_bonus)
            
        except Exception as e:
            logger.warning("Failed to analyze information integration", error=e)
            return 0.5
    
    def _check_source_consistency(self, citations: List[SourceCitation]) -> Tuple[float, List[str]]:
        """Check for consistency across source documents"""
        try:
            if len(citations) <= 1:
                return 1.0, []
            
            conflicting_info = []
            consistency_score = 1.0
            
            # Use LLM to check for conflicts between sources
            sources_text = "\n\n".join([
                f"Source {i+1} ({citation.filename}): {citation.excerpt}"
                for i, citation in enumerate(citations[:5])  # Limit to top 5 sources
            ])
            
            consistency_prompt = f"""
Analyze the following sources for conflicting or contradictory information:

{sources_text}

Identify any conflicts, contradictions, or inconsistencies between the sources.
List each conflict clearly.

Conflicts found:"""
            
            consistency_response = self.llm.invoke(consistency_prompt)
            
            # Parse response for conflicts
            if "no conflicts" in consistency_response.lower() or "no contradictions" in consistency_response.lower():
                consistency_score = 1.0
            elif "minor" in consistency_response.lower():
                consistency_score = 0.8
                conflicting_info.append("Minor inconsistencies detected between sources")
            elif "conflict" in consistency_response.lower() or "contradiction" in consistency_response.lower():
                consistency_score = 0.5
                conflicting_info.append("Significant conflicts detected between sources")
            
            return consistency_score, conflicting_info
            
        except Exception as e:
            logger.warning("Failed to check source consistency", error=e)
            return 0.7, []
    
    def _assess_coverage_completeness(self, citations: List[SourceCitation], query: str, response: str) -> float:
        """Assess how completely the sources cover the query"""
        try:
            # Extract key concepts from query
            query_words = set(query.lower().split())
            
            # Check how many query concepts are covered by sources
            covered_concepts = set()
            for citation in citations:
                if citation.excerpt:
                    excerpt_words = set(citation.excerpt.lower().split())
                    covered_concepts.update(query_words.intersection(excerpt_words))
            
            if not query_words:
                return 0.5
            
            coverage_ratio = len(covered_concepts) / len(query_words)
            
            # Bonus for comprehensive response
            response_length_bonus = min(0.2, len(response.split()) / 100)
            
            return min(1.0, coverage_ratio + response_length_bonus)
            
        except Exception as e:
            logger.warning("Failed to assess coverage completeness", error=e)
            return 0.5
    
    def _identify_information_gaps(self, citations: List[SourceCitation], query: str, response: str) -> List[str]:
        """Identify potential information gaps in the response"""
        try:
            gaps = []
            
            # Check if query asks for specific information not in sources
            query_lower = query.lower()
            
            # Common information types that might be missing
            info_types = {
                "numbers": ["how many", "what number", "quantity", "amount"],
                "dates": ["when", "what date", "timeline", "schedule"],
                "locations": ["where", "location", "place", "address"],
                "people": ["who", "person", "people", "contact"],
                "procedures": ["how to", "steps", "process", "procedure"],
                "reasons": ["why", "reason", "cause", "because"]
            }
            
            for info_type, keywords in info_types.items():
                if any(keyword in query_lower for keyword in keywords):
                    # Check if this type of information is present in sources
                    has_info = False
                    for citation in citations:
                        if citation.excerpt and any(keyword in citation.excerpt.lower() for keyword in keywords):
                            has_info = True
                            break
                    
                    if not has_info:
                        gaps.append(f"Limited {info_type} information in available sources")
            
            return gaps
            
        except Exception as e:
            logger.warning("Failed to identify information gaps", error=e)
            return []
    
    def _validate_response_quality(
        self, 
        response: str, 
        citations: List[SourceCitation], 
        query: str,
        synthesis_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate overall response quality and provide recommendations"""
        try:
            validation = {
                "quality_score": 0.0,
                "strengths": [],
                "weaknesses": [],
                "recommendations": [],
                "notes": ""
            }
            
            # Calculate quality score
            factors = {
                "source_support": len(citations) > 0,
                "confidence": synthesis_analysis.get("synthesis_quality", 0.5) > 0.6,
                "completeness": len(response.split()) > 20,
                "clarity": not any(word in response.lower() for word in ["unclear", "uncertain", "maybe"]),
                "specificity": any(char.isdigit() for char in response) or len(citations) > 1
            }
            
            quality_score = sum(factors.values()) / len(factors)
            validation["quality_score"] = quality_score
            
            # Identify strengths
            if len(citations) > 2:
                validation["strengths"].append("Multiple source documents support the response")
            if synthesis_analysis.get("consistency_score", 0) > 0.8:
                validation["strengths"].append("Sources are consistent with each other")
            if synthesis_analysis.get("source_diversity", 0) > 0.7:
                validation["strengths"].append("Diverse sources provide comprehensive coverage")
            
            # Identify weaknesses
            if len(citations) == 0:
                validation["weaknesses"].append("No source documents available")
            if synthesis_analysis.get("conflicting_information"):
                validation["weaknesses"].append("Conflicting information detected in sources")
            if synthesis_analysis.get("information_gaps"):
                validation["weaknesses"].append("Some information gaps identified")
            
            # Provide recommendations
            if quality_score < 0.6:
                validation["recommendations"].append("Consider asking more specific questions")
            if len(citations) < 2:
                validation["recommendations"].append("Upload more relevant documents for better coverage")
            if synthesis_analysis.get("consistency_score", 1.0) < 0.7:
                validation["recommendations"].append("Review sources for potential conflicts")
            
            # Generate notes
            notes_parts = []
            if validation["strengths"]:
                notes_parts.append(f"Strengths: {'; '.join(validation['strengths'])}")
            if validation["weaknesses"]:
                notes_parts.append(f"Areas for improvement: {'; '.join(validation['weaknesses'])}")
            
            validation["notes"] = ". ".join(notes_parts)
            
            return validation
            
        except Exception as e:
            logger.warning("Failed to validate response quality", error=e)
            return {"quality_score": 0.5, "notes": "Quality validation failed"}
    
    def _process_source_documents(
        self, 
        documents: List[Document], 
        query: str, 
        response: str
    ) -> List[SourceCitation]:
        """Process source documents into detailed citations"""
        citations = []
        
        for i, doc in enumerate(documents):
            try:
                # Extract metadata
                metadata = doc.metadata or {}
                document_id = metadata.get("document_id", f"doc_{i}")
                filename = metadata.get("filename", "Unknown Document")
                page_number = metadata.get("page", metadata.get("page_number"))
                
                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(doc.page_content, query)
                
                # Calculate confidence score
                confidence_score = self._calculate_confidence_score(doc.page_content, response)
                
                # Determine citation type
                citation_type = self._determine_citation_type(doc.page_content, response)
                
                # Extract relevant excerpt
                excerpt = self._extract_relevant_excerpt(doc.page_content, query, response)
                
                # Extract section information if available
                section = self._extract_section_info(doc.page_content, metadata)
                
                citation = SourceCitation(
                    document_id=document_id,
                    filename=filename,
                    page_number=page_number,
                    section=section,
                    excerpt=excerpt,
                    relevance_score=relevance_score,
                    confidence_score=confidence_score,
                    citation_type=citation_type,
                    metadata=metadata
                )
                
                citations.append(citation)
                
            except Exception as e:
                logger.warning(f"Failed to process source document {i}", error=e)
                continue
        
        # Sort citations by relevance and confidence
        citations.sort(key=lambda x: (x.relevance_score + x.confidence_score) / 2, reverse=True)
        
        return citations
    
    def _calculate_relevance_score(self, content: str, query: str) -> float:
        """Calculate relevance score between content and query"""
        try:
            # Simple keyword-based relevance scoring
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            
            if not query_words:
                return 0.0
            
            # Calculate word overlap
            overlap = len(query_words.intersection(content_words))
            relevance = overlap / len(query_words)
            
            # Boost score for exact phrase matches
            if query.lower() in content.lower():
                relevance += 0.3
            
            # Normalize to 0-1 range
            return min(relevance, 1.0)
            
        except Exception as e:
            logger.warning("Failed to calculate relevance score", error=e)
            return 0.5
    
    def _calculate_confidence_score(self, content: str, response: str) -> float:
        """Calculate confidence score for how well content supports the response"""
        try:
            # Enhanced confidence scoring with multiple factors
            
            # 1. Semantic similarity scoring
            semantic_score = self._calculate_semantic_similarity(content, response)
            
            # 2. Factual alignment scoring
            factual_score = self._calculate_factual_alignment(content, response)
            
            # 3. Completeness scoring
            completeness_score = self._calculate_completeness_score(content, response)
            
            # 4. LLM-based confidence assessment (as fallback/validation)
            llm_score = self._get_llm_confidence_score(content, response)
            
            # Weighted combination of scores
            confidence = (
                semantic_score * 0.3 +
                factual_score * 0.4 +
                completeness_score * 0.2 +
                llm_score * 0.1
            )
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.warning("Failed to calculate confidence score", error=e)
            return 0.5
    
    def _calculate_semantic_similarity(self, content: str, response: str) -> float:
        """Calculate semantic similarity between content and response"""
        try:
            # Simple word-based similarity
            content_words = set(content.lower().split())
            response_words = set(response.lower().split())
            
            if not response_words:
                return 0.0
            
            intersection = content_words.intersection(response_words)
            union = content_words.union(response_words)
            
            # Jaccard similarity
            jaccard = len(intersection) / len(union) if union else 0.0
            
            # Boost for exact phrase matches
            response_phrases = [phrase.strip() for phrase in response.split('.') if len(phrase.strip()) > 10]
            phrase_matches = 0
            for phrase in response_phrases:
                if phrase.lower() in content.lower():
                    phrase_matches += 1
            
            phrase_bonus = min(0.3, phrase_matches * 0.1)
            
            return min(1.0, jaccard + phrase_bonus)
            
        except Exception as e:
            logger.warning("Failed to calculate semantic similarity", error=e)
            return 0.5
    
    def _calculate_factual_alignment(self, content: str, response: str) -> float:
        """Calculate how well facts in response align with content"""
        try:
            # Extract potential facts (numbers, dates, names, etc.)
            import re
            
            # Extract numbers
            content_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', content))
            response_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', response))
            
            # Extract dates
            date_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b'
            content_dates = set(re.findall(date_pattern, content))
            response_dates = set(re.findall(date_pattern, response))
            
            # Extract capitalized words (potential names/entities)
            content_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content))
            response_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response))
            
            # Calculate alignment scores
            number_alignment = self._calculate_set_alignment(content_numbers, response_numbers)
            date_alignment = self._calculate_set_alignment(content_dates, response_dates)
            entity_alignment = self._calculate_set_alignment(content_entities, response_entities)
            
            # Weighted average
            alignment = (number_alignment * 0.4 + date_alignment * 0.3 + entity_alignment * 0.3)
            
            return alignment
            
        except Exception as e:
            logger.warning("Failed to calculate factual alignment", error=e)
            return 0.5
    
    def _calculate_set_alignment(self, content_set: set, response_set: set) -> float:
        """Calculate alignment between two sets of extracted facts"""
        if not response_set:
            return 1.0  # No claims to verify
        
        if not content_set:
            return 0.0  # Claims made but no source facts
        
        # Calculate how many response facts are supported by content
        supported = len(response_set.intersection(content_set))
        total_claims = len(response_set)
        
        return supported / total_claims
    
    def _calculate_completeness_score(self, content: str, response: str) -> float:
        """Calculate how complete the response is given the available content"""
        try:
            # Check if response addresses the main topics in content
            content_sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
            response_sentences = [s.strip() for s in response.split('.') if len(s.strip()) > 10]
            
            if not content_sentences or not response_sentences:
                return 0.5
            
            # Simple heuristic: longer responses with more detail tend to be more complete
            # but only if they're relevant to the content
            
            content_length = len(content.split())
            response_length = len(response.split())
            
            # Optimal response length is roughly 10-30% of content length
            optimal_ratio = 0.2
            actual_ratio = response_length / content_length if content_length > 0 else 0
            
            # Score based on how close to optimal ratio
            if actual_ratio <= optimal_ratio:
                completeness = actual_ratio / optimal_ratio
            else:
                # Penalize overly long responses
                completeness = max(0.5, 1.0 - (actual_ratio - optimal_ratio))
            
            return min(1.0, completeness)
            
        except Exception as e:
            logger.warning("Failed to calculate completeness score", error=e)
            return 0.5
    
    def _get_llm_confidence_score(self, content: str, response: str) -> float:
        """Get LLM-based confidence score as validation"""
        try:
            confidence_prompt = f"""
Analyze how well the source content supports the given response. Rate the confidence on a scale of 0.0 to 1.0.

Source Content:
{content[:800]}

Response:
{response[:400]}

Consider:
- Direct factual support
- Logical consistency
- Completeness of information
- Potential contradictions

Respond with only a number between 0.0 and 1.0:"""
            
            confidence_response = self.llm.invoke(confidence_prompt)
            
            # Extract numerical score
            score_match = re.search(r'(\d+\.?\d*)', confidence_response)
            if score_match:
                score = float(score_match.group(1))
                # Normalize if needed
                if score > 1.0:
                    score = score / 10.0 if score <= 10.0 else 1.0
                return max(0.0, min(1.0, score))
            
            return 0.5  # Default if no score found
            
        except Exception as e:
            logger.warning("Failed to get LLM confidence score", error=e)
            return 0.5
    
    def _determine_citation_type(self, content: str, response: str) -> str:
        """Determine the type of citation (direct, paraphrase, inference)"""
        try:
            # Check for direct quotes
            for pattern in self.citation_patterns.values():
                if re.search(pattern, response, re.IGNORECASE):
                    return "direct"
            
            # Check for paraphrasing (similar concepts, different wording)
            content_words = set(content.lower().split())
            response_words = set(response.lower().split())
            
            overlap_ratio = len(content_words.intersection(response_words)) / len(response_words) if response_words else 0
            
            if overlap_ratio > 0.3:
                return "paraphrase"
            elif overlap_ratio > 0.1:
                return "inference"
            else:
                return "indirect"
                
        except Exception as e:
            logger.warning("Failed to determine citation type", error=e)
            return "unknown"
    
    def _extract_relevant_excerpt(self, content: str, query: str, response: str) -> str:
        """Extract the most relevant excerpt from content"""
        try:
            # Find sentences that contain query keywords or response concepts
            sentences = re.split(r'[.!?]+', content)
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            
            scored_sentences = []
            for sentence in sentences:
                if len(sentence.strip()) < 10:  # Skip very short sentences
                    continue
                
                sentence_words = set(sentence.lower().split())
                query_overlap = len(query_words.intersection(sentence_words))
                response_overlap = len(response_words.intersection(sentence_words))
                
                score = query_overlap * 2 + response_overlap  # Weight query matches higher
                scored_sentences.append((sentence.strip(), score))
            
            # Sort by score and take the best sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            
            # Combine top sentences up to max length
            excerpt = ""
            for sentence, score in scored_sentences:
                if score == 0:
                    break
                if len(excerpt + sentence) <= self.max_excerpt_length:
                    excerpt += sentence + ". "
                else:
                    break
            
            return excerpt.strip() or content[:self.max_excerpt_length] + "..."
            
        except Exception as e:
            logger.warning("Failed to extract relevant excerpt", error=e)
            return content[:self.max_excerpt_length] + "..."
    
    def _extract_section_info(self, content: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract section information from content or metadata"""
        try:
            # Check metadata first
            section = metadata.get("section") or metadata.get("chapter") or metadata.get("heading")
            if section:
                return str(section)
            
            # Try to extract from content
            section_patterns = [
                r'^#+\s*(.+)$',  # Markdown headers
                r'^(\d+\.?\d*\.?\s*.+)$',  # Numbered sections
                r'^([A-Z][^.!?]*):',  # Title-like patterns
            ]
            
            lines = content.split('\n')
            for line in lines[:5]:  # Check first few lines
                for pattern in section_patterns:
                    match = re.match(pattern, line.strip(), re.MULTILINE)
                    if match:
                        return match.group(1).strip()
            
            return None
            
        except Exception as e:
            logger.warning("Failed to extract section info", error=e)
            return None
    
    def _calculate_enhanced_overall_confidence(
        self, 
        citations: List[SourceCitation], 
        response: str,
        synthesis_analysis: Dict[str, Any]
    ) -> float:
        """Calculate enhanced overall confidence with synthesis factors"""
        try:
            if not citations:
                return 0.2  # Very low confidence without sources
            
            # 1. Base confidence from citations
            base_confidence = self._calculate_base_citation_confidence(citations)
            
            # 2. Synthesis quality factor
            synthesis_factor = synthesis_analysis.get("synthesis_quality", 0.5)
            
            # 3. Source consistency factor
            consistency_factor = synthesis_analysis.get("consistency_score", 0.7)
            
            # 4. Coverage completeness factor
            coverage_factor = synthesis_analysis.get("coverage_completeness", 0.5)
            
            # 5. Response quality factors
            response_factors = self._analyze_response_quality_factors(response)
            
            # 6. Multi-document synthesis bonus
            multi_doc_bonus = self._calculate_multi_document_bonus(citations, synthesis_analysis)
            
            # Weighted combination
            confidence = (
                base_confidence * 0.35 +
                synthesis_factor * 0.25 +
                consistency_factor * 0.20 +
                coverage_factor * 0.15 +
                response_factors * 0.05
            ) + multi_doc_bonus
            
            # Apply penalties for uncertainty indicators
            uncertainty_penalty = self._calculate_uncertainty_penalty(response)
            
            final_confidence = confidence - uncertainty_penalty
            return max(0.0, min(1.0, final_confidence))
            
        except Exception as e:
            logger.warning("Failed to calculate enhanced overall confidence", error=e)
            return 0.5
    
    def _calculate_base_citation_confidence(self, citations: List[SourceCitation]) -> float:
        """Calculate base confidence from citation scores"""
        try:
            if not citations:
                return 0.0
            
            # Weight by citation confidence and relevance
            total_weight = 0
            weighted_confidence = 0
            
            for citation in citations:
                # Combined weight from relevance and confidence
                weight = (citation.relevance_score * 0.6 + citation.confidence_score * 0.4)
                weighted_confidence += citation.confidence_score * weight
                total_weight += weight
            
            if total_weight == 0:
                return 0.3
            
            return weighted_confidence / total_weight
            
        except Exception as e:
            logger.warning("Failed to calculate base citation confidence", error=e)
            return 0.5
    
    def _analyze_response_quality_factors(self, response: str) -> float:
        """Analyze response quality factors that affect confidence"""
        try:
            quality_score = 0.5  # Base score
            
            # Length factor (not too short, not too verbose)
            word_count = len(response.split())
            if 20 <= word_count <= 200:
                quality_score += 0.1
            elif word_count < 10:
                quality_score -= 0.2
            
            # Specificity indicators
            specificity_indicators = [
                r'\b\d+\b',  # Numbers
                r'\b\d{4}\b',  # Years
                r'\b[A-Z][a-z]+ \d+\b',  # Dates like "March 15"
                r'\b\$\d+\b',  # Money amounts
                r'\b\d+%\b',  # Percentages
            ]
            
            for pattern in specificity_indicators:
                if re.search(pattern, response):
                    quality_score += 0.05
            
            # Structure indicators
            if any(indicator in response for indicator in ['1.', '2.', '•', '-']):
                quality_score += 0.1  # Structured response
            
            # Professional language indicators
            professional_terms = ['according to', 'based on', 'analysis shows', 'data indicates']
            if any(term in response.lower() for term in professional_terms):
                quality_score += 0.1
            
            return min(1.0, quality_score)
            
        except Exception as e:
            logger.warning("Failed to analyze response quality factors", error=e)
            return 0.5
    
    def _calculate_multi_document_bonus(self, citations: List[SourceCitation], synthesis_analysis: Dict[str, Any]) -> float:
        """Calculate bonus for effective multi-document synthesis"""
        try:
            if len(citations) <= 1:
                return 0.0
            
            # Base bonus for multiple sources
            base_bonus = min(0.15, len(citations) * 0.03)
            
            # Additional bonus for good synthesis
            synthesis_quality = synthesis_analysis.get("synthesis_quality", 0.0)
            synthesis_bonus = synthesis_quality * 0.1
            
            # Bonus for source diversity
            diversity_score = synthesis_analysis.get("source_diversity", 0.0)
            diversity_bonus = diversity_score * 0.05
            
            return base_bonus + synthesis_bonus + diversity_bonus
            
        except Exception as e:
            logger.warning("Failed to calculate multi-document bonus", error=e)
            return 0.0
    
    def _calculate_uncertainty_penalty(self, response: str) -> float:
        """Calculate penalty for uncertainty indicators in response"""
        try:
            penalty = 0.0
            response_lower = response.lower()
            
            # Strong uncertainty indicators
            strong_uncertainty = ['unclear', 'uncertain', 'unknown', 'cannot determine', 'insufficient information']
            for indicator in strong_uncertainty:
                if indicator in response_lower:
                    penalty += 0.15
            
            # Moderate uncertainty indicators
            moderate_uncertainty = ['might', 'may', 'could', 'possibly', 'perhaps', 'likely']
            for indicator in moderate_uncertainty:
                if indicator in response_lower:
                    penalty += 0.05
            
            # Qualification indicators
            qualifications = ['seems', 'appears', 'suggests', 'indicates']
            for indicator in qualifications:
                if indicator in response_lower:
                    penalty += 0.03
            
            return min(0.3, penalty)  # Cap penalty at 0.3
            
        except Exception as e:
            logger.warning("Failed to calculate uncertainty penalty", error=e)
            return 0.0
    
    def _determine_enhanced_synthesis_type(self, citations: List[SourceCitation], synthesis_analysis: Dict[str, Any]) -> str:
        """Determine enhanced synthesis type based on analysis"""
        try:
            source_count = len(citations)
            synthesis_quality = synthesis_analysis.get("synthesis_quality", 0.0)
            integration_score = synthesis_analysis.get("information_integration", 0.0)
            
            if source_count == 0:
                return "no_source"
            elif source_count == 1:
                return "single_source"
            elif source_count == 2:
                if synthesis_quality > 0.7:
                    return "dual_source_synthesis"
                else:
                    return "dual_source_basic"
            elif source_count <= 4:
                if synthesis_quality > 0.8 and integration_score > 0.7:
                    return "multi_source_comprehensive"
                elif synthesis_quality > 0.6:
                    return "multi_source_good"
                else:
                    return "multi_source_basic"
            else:
                if synthesis_quality > 0.8 and integration_score > 0.8:
                    return "comprehensive_synthesis_excellent"
                elif synthesis_quality > 0.6:
                    return "comprehensive_synthesis_good"
                else:
                    return "comprehensive_synthesis_basic"
                
        except Exception as e:
            logger.warning("Failed to determine enhanced synthesis type", error=e)
            return "unknown"
    
    def _determine_synthesis_type(self, citations: List[SourceCitation], response: str) -> str:
        """Determine how the response synthesizes information (legacy method)"""
        try:
            if len(citations) == 0:
                return "no_source"
            elif len(citations) == 1:
                return "single_source"
            elif len(citations) <= 3:
                return "multi_source"
            else:
                return "comprehensive_synthesis"
                
        except Exception as e:
            logger.warning("Failed to determine synthesis type", error=e)
            return "unknown"
    
    def _perform_comprehensive_fact_checking(
        self, 
        response: str, 
        citations: List[SourceCitation],
        synthesis_analysis: Dict[str, Any]
    ) -> Tuple[str, Optional[str]]:
        """Perform comprehensive fact-checking with multi-document validation"""
        try:
            if not citations:
                return "no_sources", "No source documents available for verification"
            
            # 1. Automated fact extraction and verification
            fact_verification = self._verify_extractable_facts(response, citations)
            
            # 2. Cross-source consistency check
            consistency_check = self._cross_verify_sources(citations)
            
            # 3. LLM-based comprehensive fact-checking
            llm_fact_check = self._llm_comprehensive_fact_check(response, citations)
            
            # 4. Synthesis-based validation
            synthesis_validation = self._validate_synthesis_accuracy(response, synthesis_analysis)
            
            # Combine all fact-checking results
            final_status, final_notes = self._combine_fact_check_results(
                fact_verification, consistency_check, llm_fact_check, synthesis_validation
            )
            
            return final_status, final_notes
            
        except Exception as e:
            logger.warning("Failed to perform comprehensive fact-checking", error=e)
            return "error", f"Comprehensive fact-checking failed: {str(e)}"
    
    def _verify_extractable_facts(self, response: str, citations: List[SourceCitation]) -> Dict[str, Any]:
        """Verify extractable facts (numbers, dates, names) against sources"""
        try:
            verification = {
                "status": "verified",
                "verified_facts": [],
                "unverified_facts": [],
                "conflicting_facts": []
            }
            
            # Extract facts from response
            response_facts = self._extract_verifiable_facts(response)
            
            # Check each fact against sources
            for fact_type, facts in response_facts.items():
                for fact in facts:
                    verification_result = self._verify_single_fact(fact, citations, fact_type)
                    
                    if verification_result["status"] == "verified":
                        verification["verified_facts"].append(f"{fact_type}: {fact}")
                    elif verification_result["status"] == "conflicting":
                        verification["conflicting_facts"].append(f"{fact_type}: {fact}")
                        verification["status"] = "conflicting"
                    else:
                        verification["unverified_facts"].append(f"{fact_type}: {fact}")
                        if verification["status"] == "verified":
                            verification["status"] = "partial"
            
            return verification
            
        except Exception as e:
            logger.warning("Failed to verify extractable facts", error=e)
            return {"status": "error", "verified_facts": [], "unverified_facts": [], "conflicting_facts": []}
    
    def _extract_verifiable_facts(self, response: str) -> Dict[str, List[str]]:
        """Extract verifiable facts from response"""
        facts = {
            "numbers": re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', response),
            "dates": re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', response),
            "percentages": re.findall(r'\b\d+(?:\.\d+)?%\b', response),
            "money": re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b', response),
            "proper_nouns": re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response)
        }
        
        # Filter out common words from proper nouns
        common_words = {'The', 'This', 'That', 'These', 'Those', 'According', 'Based', 'However', 'Therefore'}
        facts["proper_nouns"] = [noun for noun in facts["proper_nouns"] if noun not in common_words]
        
        return facts
    
    def _verify_single_fact(self, fact: str, citations: List[SourceCitation], fact_type: str) -> Dict[str, Any]:
        """Verify a single fact against source citations"""
        try:
            supporting_sources = 0
            conflicting_sources = 0
            
            for citation in citations:
                if not citation.excerpt:
                    continue
                
                excerpt_lower = citation.excerpt.lower()
                fact_lower = fact.lower()
                
                if fact_type == "proper_nouns":
                    # For proper nouns, check for exact match or close variants
                    if fact_lower in excerpt_lower:
                        supporting_sources += 1
                else:
                    # For numbers, dates, etc., check for exact match
                    if fact in citation.excerpt:
                        supporting_sources += 1
                    # Check for potential conflicts (different numbers in same context)
                    elif fact_type in ["numbers", "percentages", "money"] and any(char.isdigit() for char in citation.excerpt):
                        # This is a simplified conflict detection
                        conflicting_sources += 1
            
            if supporting_sources > 0 and conflicting_sources == 0:
                return {"status": "verified", "supporting_sources": supporting_sources}
            elif conflicting_sources > 0:
                return {"status": "conflicting", "supporting_sources": supporting_sources, "conflicting_sources": conflicting_sources}
            else:
                return {"status": "unverified", "supporting_sources": 0}
                
        except Exception as e:
            logger.warning(f"Failed to verify single fact: {fact}", error=e)
            return {"status": "error"}
    
    def _cross_verify_sources(self, citations: List[SourceCitation]) -> Dict[str, Any]:
        """Cross-verify information consistency across sources"""
        try:
            if len(citations) <= 1:
                return {"status": "single_source", "consistency": 1.0}
            
            # Use LLM to check for cross-source consistency
            sources_text = "\n\n".join([
                f"Source {i+1} ({citation.filename}, Page {citation.page_number or 'N/A'}): {citation.excerpt}"
                for i, citation in enumerate(citations[:4])  # Limit to top 4 sources
            ])
            
            cross_verify_prompt = f"""
Compare the following sources for factual consistency. Look for:
1. Contradictory facts or figures
2. Inconsistent dates or timelines
3. Conflicting statements about the same topic
4. Different versions of the same information

Sources:
{sources_text}

Analysis:
- Consistency Level: [high/medium/low]
- Conflicts Found: [list any specific conflicts]
- Reliability Assessment: [reliable/questionable/conflicting]

Consistency Level:"""
            
            cross_verify_response = self.llm.invoke(cross_verify_prompt)
            
            # Parse response
            consistency_level = "medium"
            conflicts = []
            
            if "high" in cross_verify_response.lower():
                consistency_level = "high"
            elif "low" in cross_verify_response.lower():
                consistency_level = "low"
            
            # Extract conflicts if mentioned
            if "conflicts found:" in cross_verify_response.lower():
                conflict_section = cross_verify_response.lower().split("conflicts found:")[1]
                if "none" not in conflict_section and "no conflicts" not in conflict_section:
                    conflicts.append("Cross-source conflicts detected")
            
            return {
                "status": consistency_level,
                "consistency": 0.9 if consistency_level == "high" else 0.6 if consistency_level == "medium" else 0.3,
                "conflicts": conflicts
            }
            
        except Exception as e:
            logger.warning("Failed to cross-verify sources", error=e)
            return {"status": "error", "consistency": 0.5}
    
    def _llm_comprehensive_fact_check(self, response: str, citations: List[SourceCitation]) -> Dict[str, Any]:
        """Comprehensive LLM-based fact-checking"""
        try:
            sources_text = "\n\n".join([
                f"Source {i+1} ({citation.filename}): {citation.excerpt}"
                for i, citation in enumerate(citations[:3])
            ])
            
            comprehensive_prompt = f"""
Perform a comprehensive fact-check of the response against the provided sources.

Response to verify:
{response}

Source materials:
{sources_text}

Comprehensive Analysis:
1. Factual Accuracy: Are all facts in the response supported by sources?
2. Completeness: Does the response miss important information from sources?
3. Context: Is the information presented in proper context?
4. Inference Quality: Are any inferences or conclusions well-supported?
5. Potential Issues: Any misleading or problematic statements?

Overall Assessment: [verified/mostly_verified/partially_verified/questionable/conflicting]
Key Issues: [list main concerns if any]
Confidence Level: [high/medium/low]

Overall Assessment:"""
            
            fact_check_response = self.llm.invoke(comprehensive_prompt)
            
            # Parse comprehensive response
            assessment = "partially_verified"
            confidence = "medium"
            issues = []
            
            response_lower = fact_check_response.lower()
            
            if "verified" in response_lower:
                if "mostly_verified" in response_lower:
                    assessment = "mostly_verified"
                elif "partially_verified" in response_lower:
                    assessment = "partially_verified"
                else:
                    assessment = "verified"
            elif "questionable" in response_lower:
                assessment = "questionable"
            elif "conflicting" in response_lower:
                assessment = "conflicting"
            
            if "high" in response_lower:
                confidence = "high"
            elif "low" in response_lower:
                confidence = "low"
            
            # Extract issues
            if "key issues:" in response_lower:
                issues_section = fact_check_response.split("Key Issues:")[1] if "Key Issues:" in fact_check_response else ""
                if issues_section and "none" not in issues_section.lower():
                    issues.append("LLM identified potential issues in fact-checking")
            
            return {
                "status": assessment,
                "confidence": confidence,
                "issues": issues,
                "detailed_analysis": fact_check_response
            }
            
        except Exception as e:
            logger.warning("Failed to perform LLM comprehensive fact-check", error=e)
            return {"status": "error", "confidence": "low", "issues": ["Fact-checking process failed"]}
    
    def _validate_synthesis_accuracy(self, response: str, synthesis_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate accuracy of multi-document synthesis"""
        try:
            validation = {"status": "valid", "issues": []}
            
            synthesis_quality = synthesis_analysis.get("synthesis_quality", 0.5)
            if synthesis_quality < 0.4:
                validation["issues"].append("Low synthesis quality may affect accuracy")
                validation["status"] = "questionable"
            
            conflicting_info = synthesis_analysis.get("conflicting_information", [])
            if conflicting_info:
                validation["issues"].extend(conflicting_info)
                validation["status"] = "conflicting"
            
            information_gaps = synthesis_analysis.get("information_gaps", [])
            if len(information_gaps) > 2:
                validation["issues"].append("Multiple information gaps may affect completeness")
                if validation["status"] == "valid":
                    validation["status"] = "incomplete"
            
            return validation
            
        except Exception as e:
            logger.warning("Failed to validate synthesis accuracy", error=e)
            return {"status": "error", "issues": ["Synthesis validation failed"]}
    
    def _combine_fact_check_results(
        self, 
        fact_verification: Dict[str, Any],
        consistency_check: Dict[str, Any],
        llm_fact_check: Dict[str, Any],
        synthesis_validation: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Combine all fact-checking results into final status and notes"""
        try:
            # Determine overall status
            status_scores = {
                "verified": 4,
                "mostly_verified": 3,
                "partially_verified": 2,
                "questionable": 1,
                "conflicting": 0,
                "error": 0
            }
            
            # Get scores for each check
            fact_score = status_scores.get(fact_verification.get("status", "error"), 0)
            consistency_score = status_scores.get(consistency_check.get("status", "medium"), 2)
            llm_score = status_scores.get(llm_fact_check.get("status", "error"), 0)
            synthesis_score = status_scores.get(synthesis_validation.get("status", "valid"), 3)
            
            # Calculate weighted average
            avg_score = (fact_score * 0.3 + consistency_score * 0.2 + llm_score * 0.3 + synthesis_score * 0.2)
            
            # Determine final status
            if avg_score >= 3.5:
                final_status = "verified"
            elif avg_score >= 2.5:
                final_status = "mostly_verified"
            elif avg_score >= 1.5:
                final_status = "partially_verified"
            elif avg_score >= 0.5:
                final_status = "questionable"
            else:
                final_status = "conflicting"
            
            # Compile notes
            notes_parts = []
            
            # Add verification details
            if fact_verification.get("verified_facts"):
                notes_parts.append(f"Verified facts: {len(fact_verification['verified_facts'])}")
            
            if fact_verification.get("conflicting_facts"):
                notes_parts.append(f"Conflicting facts detected: {len(fact_verification['conflicting_facts'])}")
            
            # Add consistency information
            consistency = consistency_check.get("consistency", 0.5)
            notes_parts.append(f"Source consistency: {consistency:.1%}")
            
            # Add LLM assessment
            llm_confidence = llm_fact_check.get("confidence", "medium")
            notes_parts.append(f"LLM confidence: {llm_confidence}")
            
            # Add synthesis validation
            synthesis_issues = synthesis_validation.get("issues", [])
            if synthesis_issues:
                notes_parts.append(f"Synthesis issues: {len(synthesis_issues)}")
            
            final_notes = "; ".join(notes_parts)
            
            return final_status, final_notes
            
        except Exception as e:
            logger.warning("Failed to combine fact-check results", error=e)
            return "error", "Failed to combine fact-checking results"
    
    def _identify_enhanced_uncertainty_indicators(
        self, 
        response: str, 
        citations: List[SourceCitation],
        synthesis_analysis: Dict[str, Any]
    ) -> List[str]:
        """Identify enhanced uncertainty indicators with context"""
        try:
            indicators = []
            response_lower = response.lower()
            
            # 1. Linguistic uncertainty indicators
            linguistic_indicators = self._identify_linguistic_uncertainty(response_lower)
            indicators.extend(linguistic_indicators)
            
            # 2. Source-based uncertainty indicators
            source_indicators = self._identify_source_based_uncertainty(citations, synthesis_analysis)
            indicators.extend(source_indicators)
            
            # 3. Content-based uncertainty indicators
            content_indicators = self._identify_content_based_uncertainty(response, citations)
            indicators.extend(content_indicators)
            
            # 4. Synthesis-based uncertainty indicators
            synthesis_indicators = self._identify_synthesis_based_uncertainty(synthesis_analysis)
            indicators.extend(synthesis_indicators)
            
            return list(set(indicators))  # Remove duplicates
            
        except Exception as e:
            logger.warning("Failed to identify enhanced uncertainty indicators", error=e)
            return []
    
    def _identify_linguistic_uncertainty(self, response_lower: str) -> List[str]:
        """Identify linguistic uncertainty patterns"""
        indicators = []
        
        uncertainty_patterns = {
            "strong_hedging": {
                "patterns": [r"\b(?:might|may|could|possibly|perhaps)\b"],
                "description": "Strong hedging language indicates uncertainty"
            },
            "weak_hedging": {
                "patterns": [r"\b(?:likely|probably|generally|typically)\b"],
                "description": "Moderate hedging suggests qualified confidence"
            },
            "epistemic_uncertainty": {
                "patterns": [r"\b(?:seems|appears|suggests|indicates|implies)\b"],
                "description": "Epistemic markers show inferential reasoning"
            },
            "explicit_uncertainty": {
                "patterns": [r"\b(?:unclear|uncertain|unknown|unsure|difficult to determine|hard to say)\b"],
                "description": "Explicit uncertainty statements"
            },
            "approximation": {
                "patterns": [r"\b(?:approximately|roughly|about|around|nearly|close to)\b"],
                "description": "Approximation language indicates imprecise information"
            },
            "conditionality": {
                "patterns": [r"\b(?:if|assuming|provided that|given that|depending on)\b"],
                "description": "Conditional statements show context-dependent validity"
            }
        }
        
        for category, info in uncertainty_patterns.items():
            for pattern in info["patterns"]:
                matches = re.findall(pattern, response_lower)
                if matches:
                    unique_matches = list(set(matches))
                    indicators.append(f"{info['description']}: {', '.join(unique_matches)}")
        
        return indicators
    
    def _identify_source_based_uncertainty(
        self, 
        citations: List[SourceCitation],
        synthesis_analysis: Dict[str, Any]
    ) -> List[str]:
        """Identify uncertainty based on source characteristics"""
        indicators = []
        
        # Low source count
        if len(citations) == 0:
            indicators.append("No source documents available for verification")
        elif len(citations) == 1:
            indicators.append("Response based on single source document")
        
        # Low confidence sources
        low_confidence_sources = [c for c in citations if c.confidence_score < 0.5]
        if low_confidence_sources:
            indicators.append(f"{len(low_confidence_sources)} sources have low confidence scores")
        
        # Conflicting information
        if synthesis_analysis.get("conflicting_information"):
            indicators.append("Conflicting information detected across sources")
        
        # Low source diversity
        if synthesis_analysis.get("source_diversity", 1.0) < 0.3:
            indicators.append("Limited diversity in source documents")
        
        # Information gaps
        gaps = synthesis_analysis.get("information_gaps", [])
        if gaps:
            indicators.append(f"Information gaps identified: {len(gaps)} areas")
        
        return indicators
    
    def _identify_content_based_uncertainty(self, response: str, citations: List[SourceCitation]) -> List[str]:
        """Identify uncertainty based on content analysis"""
        indicators = []
        
        # Check for incomplete information patterns
        incomplete_patterns = [
            r"more information needed",
            r"additional details required",
            r"further investigation",
            r"incomplete data",
            r"limited information available"
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, response.lower()):
                indicators.append("Response indicates incomplete information")
                break
        
        # Check for conflicting statements within response
        conflict_patterns = [
            r"however",
            r"on the other hand",
            r"but",
            r"although",
            r"despite"
        ]
        
        conflict_count = sum(1 for pattern in conflict_patterns if re.search(pattern, response.lower()))
        if conflict_count >= 2:
            indicators.append("Response contains multiple conflicting statements")
        
        # Check response length vs source content
        if citations:
            avg_source_length = sum(len(c.excerpt.split()) for c in citations if c.excerpt) / len(citations)
            response_length = len(response.split())
            
            if response_length > avg_source_length * 2:
                indicators.append("Response significantly longer than source material (potential inference)")
            elif response_length < avg_source_length * 0.3:
                indicators.append("Response much shorter than available source material (potential information loss)")
        
        return indicators
    
    def _identify_synthesis_based_uncertainty(self, synthesis_analysis: Dict[str, Any]) -> List[str]:
        """Identify uncertainty based on synthesis quality"""
        indicators = []
        
        synthesis_quality = synthesis_analysis.get("synthesis_quality", 0.5)
        if synthesis_quality < 0.4:
            indicators.append("Low synthesis quality across multiple sources")
        
        consistency_score = synthesis_analysis.get("consistency_score", 1.0)
        if consistency_score < 0.6:
            indicators.append("Inconsistent information across source documents")
        
        coverage_completeness = synthesis_analysis.get("coverage_completeness", 0.5)
        if coverage_completeness < 0.4:
            indicators.append("Incomplete coverage of query topics in available sources")
        
        return indicators
    
    def format_citations_for_display(self, citations: List[SourceCitation]) -> str:
        """Format citations for user display"""
        try:
            if not citations:
                return "No sources available."
            
            formatted_citations = []
            for i, citation in enumerate(citations, 1):
                citation_text = f"**[{i}] {citation.filename}**"
                
                if citation.page_number:
                    citation_text += f" (Page {citation.page_number})"
                
                if citation.section:
                    citation_text += f" - {citation.section}"
                
                citation_text += f"\n*Confidence: {citation.confidence_score:.1%}*"
                citation_text += f"\n*Relevance: {citation.relevance_score:.1%}*"
                
                if citation.excerpt:
                    citation_text += f"\n> {citation.excerpt}"
                
                formatted_citations.append(citation_text)
            
            return "\n\n".join(formatted_citations)
            
        except Exception as e:
            logger.warning("Failed to format citations", error=e)
            return "Error formatting citations."
    
    def generate_confidence_explanation(self, attribution: ResponseAttribution) -> str:
        """Generate human-readable confidence explanation"""
        try:
            confidence = attribution.overall_confidence
            source_count = len(attribution.sources)
            
            if confidence >= self.high_confidence_threshold:
                confidence_level = "High"
                explanation = "The response is well-supported by reliable sources"
            elif confidence >= self.min_confidence_threshold:
                confidence_level = "Medium"
                explanation = "The response has moderate support from available sources"
            else:
                confidence_level = "Low"
                explanation = "The response has limited source support"
            
            details = []
            
            if source_count == 0:
                details.append("No source documents were available")
            elif source_count == 1:
                details.append("Based on a single source document")
            else:
                details.append(f"Synthesized from {source_count} source documents")
            
            if attribution.uncertainty_indicators:
                details.append(f"Contains {len(attribution.uncertainty_indicators)} uncertainty indicators")
            
            if attribution.fact_check_status == "verified":
                details.append("Facts verified against sources")
            elif attribution.fact_check_status == "conflicting":
                details.append("Some conflicting information detected")
            
            explanation_text = f"**Confidence: {confidence_level} ({confidence:.1%})**\n\n{explanation}."
            
            if details:
                explanation_text += f"\n\n**Details:**\n• " + "\n• ".join(details)
            
            return explanation_text
            
        except Exception as e:
            logger.warning("Failed to generate confidence explanation", error=e)
            return f"Confidence: {attribution.overall_confidence:.1%}"


# Global source attribution service instance
source_attribution_service = SourceAttributionService()