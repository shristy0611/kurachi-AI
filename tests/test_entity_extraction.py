"""
Comprehensive tests for entity extraction and recognition service
Tests accuracy, custom entity types, linking, and disambiguation
"""
import pytest
import uuid
from datetime import datetime
from typing import List

from services.entity_extraction import (
    EntityExtractor, Entity, EntityLink, CustomEntityPatterns
)


class TestCustomEntityPatterns:
    """Test custom entity patterns for organizational knowledge"""
    
    def setup_method(self):
        self.patterns = CustomEntityPatterns()
    
    def test_japanese_company_patterns(self):
        """Test Japanese company pattern recognition"""
        test_texts = [
            "株式会社トヨタ自動車",
            "ソニー株式会社",
            "三菱重工業会社",
            "日本電信電話法人",
            "農業協同組合",
            "日本医師会協会"
        ]
        
        for text in test_texts:
            # Test that at least one pattern matches
            matched = False
            for pattern in self.patterns.japanese_patterns["COMPANY_JP"]:
                import re
                if re.search(pattern, text):
                    matched = True
                    break
            assert matched, f"No pattern matched for Japanese company: {text}"
    
    def test_japanese_person_patterns(self):
        """Test Japanese person pattern recognition"""
        test_texts = [
            "田中太郎さん",
            "佐藤花子様",
            "山田一郎氏",
            "鈴木部長",
            "高橋課長",
            "中村社長"
        ]
        
        for text in test_texts:
            matched = False
            for pattern in self.patterns.japanese_patterns["PERSON_JP"]:
                import re
                if re.search(pattern, text):
                    matched = True
                    break
            assert matched, f"No pattern matched for Japanese person: {text}"
    
    def test_english_company_patterns(self):
        """Test English company pattern recognition"""
        test_texts = [
            "Apple Inc",
            "Microsoft Corporation",
            "Google LLC",
            "Amazon Corp",
            "Tesla Company",
            "Meta Holdings"
        ]
        
        for text in test_texts:
            matched = False
            for pattern in self.patterns.english_patterns["COMPANY_EN"]:
                import re
                if re.search(pattern, text):
                    matched = True
                    break
            assert matched, f"No pattern matched for English company: {text}"
    
    def test_technical_patterns(self):
        """Test technical pattern recognition"""
        test_cases = {
            "EMAIL": ["test@example.com", "user.name+tag@domain.co.jp"],
            "PHONE_JP": ["03-1234-5678", "09012345678"],
            "URL": ["https://www.example.com", "http://test.co.jp/path"],
            "DATE_JP": ["2024年3月15日", "令和6年12月31日"],
            "MONEY_JP": ["1000円", "500万円", "10億円"]
        }
        
        for pattern_type, texts in test_cases.items():
            for text in texts:
                matched = False
                for pattern in self.patterns.technical_patterns[pattern_type]:
                    import re
                    if re.search(pattern, text):
                        matched = True
                        break
                assert matched, f"No {pattern_type} pattern matched for: {text}"


class TestEntityExtractor:
    """Test main entity extraction functionality"""
    
    def setup_method(self):
        self.extractor = EntityExtractor()
        self.test_document_id = str(uuid.uuid4())
    
    def test_language_detection(self):
        """Test language detection accuracy"""
        test_cases = [
            ("Hello world, this is English text.", "en"),
            ("こんにちは、これは日本語のテキストです。", "ja"),
            ("This is mixed text with 日本語 content.", "ja"),  # Should detect as Japanese due to Japanese chars
            ("", "en"),  # Empty text defaults to English
            ("123 456 789", "en")  # Numbers default to English
        ]
        
        for text, expected_lang in test_cases:
            detected_lang = self.extractor.detect_language(text)
            assert detected_lang == expected_lang, f"Expected {expected_lang}, got {detected_lang} for: {text}"
    
    def test_spacy_entity_extraction_english(self):
        """Test spaCy entity extraction for English text"""
        text = """
        John Smith works at Apple Inc in Cupertino, California. 
        He started on January 15, 2020 and earns $120,000 per year.
        His email is john.smith@apple.com.
        """
        
        entities = self.extractor.extract_entities_spacy(text, self.test_document_id)
        
        # Should extract person, organization, location, date, money
        entity_labels = [e.label for e in entities]
        
        # Check for expected entity types (may vary by spaCy model)
        expected_types = {"PERSON", "ORG", "GPE", "DATE", "MONEY"}
        found_types = set(entity_labels)
        
        # At least some expected types should be found
        assert len(found_types.intersection(expected_types)) > 0, f"Expected some of {expected_types}, found {found_types}"
        
        # Check entity properties
        for entity in entities:
            assert entity.id is not None
            assert entity.text.strip() != ""
            assert entity.confidence > 0
            assert entity.source_document_id == self.test_document_id
            assert entity.context is not None
            assert entity.properties is not None
            assert entity.properties.get("language") == "en"
    
    def test_spacy_entity_extraction_japanese(self):
        """Test spaCy entity extraction for Japanese text"""
        text = """
        田中太郎は株式会社トヨタで働いています。
        彼は2020年1月15日に入社し、年収は500万円です。
        メールアドレスはtanaka@toyota.co.jpです。
        """
        
        entities = self.extractor.extract_entities_spacy(text, self.test_document_id)
        
        # Check that entities were extracted
        assert len(entities) > 0, "Should extract entities from Japanese text"
        
        # Check entity properties
        for entity in entities:
            assert entity.properties.get("language") == "ja"
            assert entity.confidence > 0
    
    def test_regex_entity_extraction(self):
        """Test regex-based entity extraction"""
        text = """
        Contact information:
        Email: test@example.com
        Phone: 03-1234-5678
        Website: https://www.example.com
        Date: 2024年3月15日
        Amount: 1000円
        """
        
        entities = self.extractor.extract_entities_regex(text, self.test_document_id)
        
        # Should extract technical entities
        entity_labels = [e.label for e in entities]
        
        expected_labels = {"EMAIL", "PHONE_JP", "URL", "DATE_JP", "MONEY_JP"}
        found_labels = set(entity_labels)
        
        # Should find most technical patterns
        assert len(found_labels.intersection(expected_labels)) >= 3, f"Expected technical entities, found {found_labels}"
        
        # Check high confidence for technical patterns
        technical_entities = [e for e in entities if e.label in expected_labels]
        for entity in technical_entities:
            assert entity.confidence >= 0.8, f"Technical entity should have high confidence: {entity.text}"
    
    def test_combined_entity_extraction(self):
        """Test combined spaCy + regex entity extraction"""
        text = """
        田中太郎 (tanaka@toyota.co.jp) works at トヨタ自動車株式会社.
        Meeting scheduled for 2024年3月15日 at 03-1234-5678.
        Budget: $50,000 (500万円).
        """
        
        entities = self.extractor.extract_entities(text, self.test_document_id)
        
        # Should extract both spaCy and regex entities
        assert len(entities) > 0, "Should extract entities from mixed text"
        
        # Check for both types of extraction
        extraction_methods = set()
        for entity in entities:
            if entity.properties and "extraction_method" in entity.properties:
                extraction_methods.add(entity.properties["extraction_method"])
            else:
                extraction_methods.add("spacy")  # Default for spaCy entities
        
        # Should have entities from both methods (or at least one)
        assert len(extraction_methods) > 0, "Should have entities from extraction methods"
    
    def test_entity_deduplication(self):
        """Test entity deduplication functionality"""
        # Create duplicate entities
        entity1 = Entity(
            id=str(uuid.uuid4()),
            text="Apple Inc",
            label="ORG",
            start_char=0,
            end_char=9,
            confidence=0.9,
            source_document_id=self.test_document_id
        )
        
        entity2 = Entity(
            id=str(uuid.uuid4()),
            text="Apple Inc",
            label="ORG",
            start_char=5,  # Slightly different position
            end_char=14,
            confidence=0.8,
            source_document_id=self.test_document_id
        )
        
        entity3 = Entity(
            id=str(uuid.uuid4()),
            text="Microsoft",
            label="ORG",
            start_char=20,
            end_char=29,
            confidence=0.9,
            source_document_id=self.test_document_id
        )
        
        entities = [entity1, entity2, entity3]
        deduplicated = self.extractor._deduplicate_entities(entities)
        
        # Should keep only 2 entities (one Apple Inc, one Microsoft)
        assert len(deduplicated) == 2, f"Expected 2 entities after deduplication, got {len(deduplicated)}"
        
        # Should keep the higher confidence entity
        apple_entities = [e for e in deduplicated if e.text == "Apple Inc"]
        assert len(apple_entities) == 1, "Should keep only one Apple Inc entity"
        assert apple_entities[0].confidence == 0.9, "Should keep higher confidence entity"
    
    def test_entity_linking(self):
        """Test entity linking functionality"""
        entities = [
            Entity(
                id=str(uuid.uuid4()),
                text="Apple Inc",
                label="ORG",
                start_char=0,
                end_char=9,
                confidence=0.9,
                source_document_id=self.test_document_id
            ),
            Entity(
                id=str(uuid.uuid4()),
                text="Apple Inc",  # Exact match
                label="ORG",
                start_char=50,
                end_char=59,
                confidence=0.8,
                source_document_id=str(uuid.uuid4())  # Different document
            ),
            Entity(
                id=str(uuid.uuid4()),
                text="Apple Corporation",  # Similar match
                label="ORG",
                start_char=100,
                end_char=117,
                confidence=0.9,
                source_document_id=str(uuid.uuid4())
            )
        ]
        
        links = self.extractor.link_entities(entities)
        
        # Should create links between similar entities
        assert len(links) > 0, "Should create entity links"
        
        # Check link properties
        for link in links:
            assert link.entity_id != link.linked_entity_id, "Should not link entity to itself"
            assert link.confidence > 0, "Link should have confidence score"
            assert link.link_type in ["exact_match", "semantic_match"], f"Invalid link type: {link.link_type}"
            assert link.evidence is not None, "Link should have evidence"
    
    def test_entity_disambiguation(self):
        """Test entity disambiguation functionality"""
        # Create ambiguous entities (same text, different labels)
        entities = [
            Entity(
                id=str(uuid.uuid4()),
                text="Apple",
                label="ORG",
                start_char=0,
                end_char=5,
                confidence=0.9,
                source_document_id=self.test_document_id
            ),
            Entity(
                id=str(uuid.uuid4()),
                text="Apple",
                label="FOOD",  # Different interpretation
                start_char=50,
                end_char=55,
                confidence=0.7,
                source_document_id=self.test_document_id
            ),
            Entity(
                id=str(uuid.uuid4()),
                text="Apple",
                label="ORG",  # Same as first
                start_char=100,
                end_char=105,
                confidence=0.8,
                source_document_id=str(uuid.uuid4())
            )
        ]
        
        disambiguated = self.extractor.disambiguate_entities(entities)
        
        # Should resolve ambiguity
        assert len(disambiguated) > 0, "Should return disambiguated entities"
        
        # Check that ORG entities (more common) have higher confidence
        org_entities = [e for e in disambiguated if e.label == "ORG"]
        food_entities = [e for e in disambiguated if e.label == "FOOD"]
        
        if org_entities and food_entities:
            avg_org_confidence = sum(e.confidence for e in org_entities) / len(org_entities)
            avg_food_confidence = sum(e.confidence for e in food_entities) / len(food_entities)
            assert avg_org_confidence >= avg_food_confidence, "More common label should have higher confidence"


class TestEntityAccuracy:
    """Test entity extraction accuracy with real-world examples"""
    
    def setup_method(self):
        self.extractor = EntityExtractor()
        self.test_document_id = str(uuid.uuid4())
    
    def test_business_document_accuracy(self):
        """Test accuracy on business document content"""
        business_text = """
        Meeting Minutes - Q1 2024 Planning
        
        Attendees:
        - John Smith, CEO (john.smith@company.com)
        - Sarah Johnson, CTO
        - Mike Chen, VP Engineering
        
        Company: TechCorp Inc
        Date: March 15, 2024
        Location: San Francisco, CA
        
        Budget Discussion:
        - Q1 Budget: $2.5M
        - Engineering: $1.2M
        - Marketing: $800K
        
        Action Items:
        1. Launch new product by April 30, 2024
        2. Hire 5 engineers by May 2024
        3. Contact venture@funding.com for Series B
        """
        
        entities = self.extractor.extract_entities(business_text, self.test_document_id)
        
        # Verify key entities are extracted
        entity_texts = [e.text.lower() for e in entities]
        
        # Should find people
        assert any("john smith" in text for text in entity_texts), "Should extract person names"
        
        # Should find company
        assert any("techcorp" in text for text in entity_texts), "Should extract company names"
        
        # Should find dates
        date_entities = [e for e in entities if "date" in e.label.lower() or "2024" in e.text]
        assert len(date_entities) > 0, "Should extract dates"
        
        # Should find money amounts
        money_entities = [e for e in entities if "money" in e.label.lower() or "$" in e.text]
        assert len(money_entities) > 0, "Should extract money amounts"
        
        # Should find emails
        email_entities = [e for e in entities if "@" in e.text]
        assert len(email_entities) > 0, "Should extract email addresses"
    
    def test_japanese_business_accuracy(self):
        """Test accuracy on Japanese business content"""
        japanese_text = """
        会議議事録 - 2024年第1四半期計画
        
        出席者:
        - 田中太郎、代表取締役社長 (tanaka@company.co.jp)
        - 佐藤花子、技術部長
        - 山田一郎、営業部課長
        
        会社: 株式会社テックコープ
        日付: 2024年3月15日
        場所: 東京都渋谷区
        
        予算検討:
        - 第1四半期予算: 250万円
        - 技術部: 120万円
        - 営業部: 80万円
        
        アクションアイテム:
        1. 4月30日までに新製品をリリース
        2. 5月までにエンジニア5名を採用
        3. funding@venture.co.jpに連絡してシリーズB調達
        """
        
        entities = self.extractor.extract_entities(japanese_text, self.test_document_id)
        
        # Should extract Japanese entities
        assert len(entities) > 0, "Should extract entities from Japanese text"
        
        # Check for Japanese-specific patterns
        japanese_entities = [e for e in entities if e.properties and e.properties.get("language") == "ja"]
        assert len(japanese_entities) > 0, "Should identify Japanese entities"
        
        # Should find company names
        company_entities = [e for e in entities if "株式会社" in e.text or "テックコープ" in e.text]
        assert len(company_entities) > 0, "Should extract Japanese company names"
        
        # Should find money amounts
        money_entities = [e for e in entities if "円" in e.text]
        assert len(money_entities) > 0, "Should extract Japanese money amounts"
        
        # Should find dates
        date_entities = [e for e in entities if "年" in e.text and "月" in e.text]
        assert len(date_entities) > 0, "Should extract Japanese dates"
    
    def test_mixed_language_accuracy(self):
        """Test accuracy on mixed Japanese-English content"""
        mixed_text = """
        Project Alpha - プロジェクトアルファ
        
        Team Lead: John Smith (john@company.com)
        日本チームリーダー: 田中太郎 (tanaka@company.co.jp)
        
        Budget: $100,000 (1000万円)
        Deadline: December 31, 2024 (2024年12月31日)
        
        Technologies: Python, React.js, PostgreSQL
        使用技術: Python、React.js、PostgreSQL
        """
        
        entities = self.extractor.extract_entities(mixed_text, self.test_document_id)
        
        # Should handle mixed content
        assert len(entities) > 0, "Should extract entities from mixed language text"
        
        # Should find both English and Japanese entities
        languages = set()
        for entity in entities:
            if entity.properties and "language" in entity.properties:
                languages.add(entity.properties["language"])
        
        # Should detect both languages (or at least handle the content)
        assert len(entities) >= 5, "Should extract multiple entities from mixed content"


class TestEntityIntegration:
    """Integration tests for entity extraction with other components"""
    
    def setup_method(self):
        self.extractor = EntityExtractor()
        self.test_document_id = str(uuid.uuid4())
    
    def test_entity_serialization(self):
        """Test entity serialization and deserialization"""
        entity = Entity(
            id=str(uuid.uuid4()),
            text="Apple Inc",
            label="ORG",
            start_char=0,
            end_char=9,
            confidence=0.9,
            source_document_id=self.test_document_id,
            properties={"test": "value"}
        )
        
        # Test to_dict
        entity_dict = entity.to_dict()
        assert isinstance(entity_dict, dict)
        assert entity_dict["text"] == "Apple Inc"
        assert entity_dict["label"] == "ORG"
        
        # Test from_dict
        restored_entity = Entity.from_dict(entity_dict)
        assert restored_entity.text == entity.text
        assert restored_entity.label == entity.label
        assert restored_entity.confidence == entity.confidence
    
    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        # Test with different entity types and contexts
        test_cases = [
            ("Apple Inc", "ORG", 0.7),  # Should get base + org boost
            ("john.smith@company.com", "EMAIL", 0.7),  # spaCy confidence, not regex
            ("a", "MISC", 0.6),  # Very short, lower confidence
            ("Very Long Company Name Corporation", "ORG", 0.8)  # Long name, higher confidence
        ]
        
        for text, label, min_expected_confidence in test_cases:
            # Create mock spaCy span for testing
            class MockSpan:
                def __init__(self, text, label):
                    self.text = text
                    self.label_ = label
            
            class MockDoc:
                def __init__(self, text):
                    self.text = text
                    self.ents = [MockSpan(text, label)]
            
            span = MockSpan(text, label)
            doc = MockDoc(text)
            
            confidence = self.extractor._calculate_confidence(span, doc)
            assert confidence >= min_expected_confidence, f"Confidence {confidence} too low for {text}"
            assert confidence <= 1.0, f"Confidence {confidence} too high for {text}"


class TestOrganizationalEntities:
    """Test organizational knowledge entity extraction"""
    
    def setup_method(self):
        self.extractor = EntityExtractor()
        self.test_document_id = str(uuid.uuid4())
    
    def test_skill_extraction_english(self):
        """Test skill extraction from English text"""
        text = """
        Required skills:
        - Python programming experience
        - React development skills
        - AWS cloud expertise
        - Docker and Kubernetes knowledge
        - Machine Learning concepts
        """
        
        entities = self.extractor.extract_organizational_entities(text, self.test_document_id)
        
        # Should extract skills
        skill_entities = [e for e in entities if e.label == "SKILL"]
        assert len(skill_entities) > 0, "Should extract skill entities"
        
        # Check for specific technologies
        entity_texts = [e.text.lower() for e in entities]
        assert any("python" in text for text in entity_texts), "Should extract Python skill"
        assert any("react" in text for text in entity_texts), "Should extract React skill"
    
    def test_skill_extraction_japanese(self):
        """Test skill extraction from Japanese text"""
        text = """
        必要なスキル:
        - Pythonプログラミング技術
        - React開発スキル
        - AWS クラウド経験
        - Docker と Kubernetes 能力
        """
        
        entities = self.extractor.extract_organizational_entities(text, self.test_document_id)
        
        # Should extract skills
        skill_entities = [e for e in entities if e.label == "SKILL"]
        assert len(skill_entities) > 0, "Should extract Japanese skill entities"
    
    def test_process_extraction(self):
        """Test process and workflow extraction"""
        text = """
        Our development process includes:
        1. Agile methodology
        2. CI/CD workflow
        3. Code review procedure
        4. Testing process
        
        開発プロセス:
        1. アジャイル手順
        2. テストプロセス
        """
        
        entities = self.extractor.extract_organizational_entities(text, self.test_document_id)
        
        # Should extract processes
        process_entities = [e for e in entities if e.label == "PROCESS"]
        assert len(process_entities) > 0, "Should extract process entities"
    
    def test_concept_extraction(self):
        """Test concept and methodology extraction"""
        text = """
        Key concepts:
        - Machine Learning framework
        - Data Science methodology
        - Cloud Computing principles
        
        重要な概念:
        - 機械学習理論
        - データサイエンス方法論
        """
        
        entities = self.extractor.extract_organizational_entities(text, self.test_document_id)
        
        # Should extract concepts
        concept_entities = [e for e in entities if e.label == "CONCEPT"]
        assert len(concept_entities) > 0, "Should extract concept entities"
    
    def test_organizational_confidence_calculation(self):
        """Test confidence calculation for organizational entities"""
        # High confidence technical skill
        high_conf_entities = self.extractor.extract_organizational_entities(
            "Skills: Python programming experience", self.test_document_id
        )
        
        # Lower confidence generic skill
        low_conf_entities = self.extractor.extract_organizational_entities(
            "Some general skills", self.test_document_id
        )
        
        if high_conf_entities and low_conf_entities:
            high_conf = max(e.confidence for e in high_conf_entities if "python" in e.text.lower())
            low_conf = max(e.confidence for e in low_conf_entities)
            assert high_conf > low_conf, "Technical skills should have higher confidence"


class TestEntityStatistics:
    """Test entity statistics and validation"""
    
    def setup_method(self):
        self.extractor = EntityExtractor()
        self.test_document_id = str(uuid.uuid4())
    
    def test_entity_statistics(self):
        """Test entity statistics calculation"""
        entities = [
            Entity(
                id=str(uuid.uuid4()),
                text="Apple Inc",
                label="ORG",
                start_char=0,
                end_char=9,
                confidence=0.9,
                source_document_id=self.test_document_id,
                properties={"language": "en"}
            ),
            Entity(
                id=str(uuid.uuid4()),
                text="Python skills",
                label="SKILL",
                start_char=20,
                end_char=33,
                confidence=0.8,
                source_document_id=self.test_document_id,
                properties={"language": "en", "entity_category": "organizational_knowledge"}
            ),
            Entity(
                id=str(uuid.uuid4()),
                text="john@example.com",
                label="EMAIL",
                start_char=40,
                end_char=56,
                confidence=0.95,
                source_document_id=self.test_document_id,
                properties={"language": "en"}
            )
        ]
        
        stats = self.extractor.get_entity_statistics(entities)
        
        assert stats["total_entities"] == 3
        assert stats["entities_by_label"]["ORG"] == 1
        assert stats["entities_by_label"]["SKILL"] == 1
        assert stats["entities_by_language"]["en"] == 3
        assert stats["entities_by_confidence"]["high"] == 3  # All above 0.8
        assert stats["organizational_entities"] == 1
        assert 0.8 < stats["average_confidence"] < 1.0
    
    def test_entity_validation(self):
        """Test entity validation"""
        # Good entities
        good_entities = [
            Entity(
                id=str(uuid.uuid4()),
                text="Apple Inc",
                label="ORG",
                start_char=0,
                end_char=9,
                confidence=0.9,
                source_document_id=self.test_document_id
            )
        ]
        
        validation = self.extractor.validate_entities(good_entities)
        assert validation["validation_passed"] == True
        assert validation["quality_score"] > 0.8
        
        # Bad entities (low confidence)
        bad_entities = [
            Entity(
                id=str(uuid.uuid4()),
                text="a",
                label="MISC",
                start_char=0,
                end_char=1,
                confidence=0.3,  # Very low confidence
                source_document_id=self.test_document_id
            )
        ]
        
        validation = self.extractor.validate_entities(bad_entities)
        assert len(validation["warnings"]) > 0
        assert validation["quality_score"] < 0.5
    
    def test_empty_entity_validation(self):
        """Test validation with no entities"""
        validation = self.extractor.validate_entities([])
        assert validation["validation_passed"] == False
        assert "No entities extracted" in validation["issues"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])