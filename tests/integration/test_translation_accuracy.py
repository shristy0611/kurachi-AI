#!/usr/bin/env python3
"""
Translation Accuracy Test Script for Japanese Business Documents
Tests the qwen3:4b translation service with real business scenarios
"""
import asyncio
import json
from typing import List, Dict, Any
from services.translation_service import translation_service, Language, TranslationQuality
from utils.logger import get_logger

logger = get_logger("translation_accuracy_test")


class TranslationAccuracyTester:
    """Test translation accuracy with business document samples"""
    
    def __init__(self):
        self.test_cases = self._load_test_cases()
        self.results = []
    
    def _load_test_cases(self) -> List[Dict[str, Any]]:
        """Load test cases for Japanese business documents"""
        return [
            {
                "category": "business_email",
                "japanese": "お疲れ様です。明日の会議の件でご連絡いたします。資料の準備ができましたので、ご確認をお願いいたします。",
                "expected_english": "Thank you for your hard work. I am contacting you regarding tomorrow's meeting. The materials are ready, so please review them.",
                "context": "Business email communication"
            },
            {
                "category": "meeting_minutes",
                "japanese": "会議の議事録：プロジェクトの進捗について話し合いました。来週までに報告書を提出する予定です。",
                "expected_english": "Meeting minutes: We discussed the project progress. We plan to submit the report by next week.",
                "context": "Meeting minutes"
            },
            {
                "category": "technical_document",
                "japanese": "データベース接続エラーが発生しました。システム管理者に連絡してください。",
                "expected_english": "A database connection error occurred. Please contact the system administrator.",
                "context": "Technical documentation"
            },
            {
                "category": "business_proposal",
                "japanese": "新しいマーケティング戦略を提案いたします。売上向上のため、デジタル広告に投資することをお勧めします。",
                "expected_english": "We propose a new marketing strategy. We recommend investing in digital advertising to improve sales.",
                "context": "Business proposal"
            },
            {
                "category": "customer_service",
                "japanese": "お客様からのお問い合わせありがとうございます。担当者が確認後、ご連絡いたします。",
                "expected_english": "Thank you for your inquiry. A representative will contact you after confirmation.",
                "context": "Customer service response"
            },
            {
                "category": "financial_report",
                "japanese": "第3四半期の売上は前年同期比15%増加しました。利益率も改善されています。",
                "expected_english": "Third quarter sales increased by 15% compared to the same period last year. The profit margin has also improved.",
                "context": "Financial report"
            },
            {
                "category": "hr_document",
                "japanese": "新入社員研修プログラムを開始します。参加者は来週月曜日までに登録してください。",
                "expected_english": "We will start the new employee training program. Participants should register by Monday next week.",
                "context": "HR documentation"
            },
            {
                "category": "contract_terms",
                "japanese": "契約期間は2024年4月1日から2025年3月31日までです。更新については別途協議いたします。",
                "expected_english": "The contract period is from April 1, 2024 to March 31, 2025. Renewal will be discussed separately.",
                "context": "Contract terms"
            }
        ]
    
    def test_translation_quality(self) -> Dict[str, Any]:
        """Test translation quality across different business document types"""
        logger.info("Starting translation accuracy tests...")
        
        total_tests = len(self.test_cases)
        passed_tests = 0
        
        for i, test_case in enumerate(self.test_cases, 1):
            logger.info(f"Testing case {i}/{total_tests}: {test_case['category']}")
            
            # Test Japanese to English
            ja_to_en_result = translation_service.translate(
                test_case["japanese"],
                Language.ENGLISH,
                Language.JAPANESE,
                TranslationQuality.BUSINESS,
                context=test_case["context"]
            )
            
            # Test English to Japanese (reverse translation)
            en_to_ja_result = translation_service.translate(
                ja_to_en_result["translated_text"],
                Language.JAPANESE,
                Language.ENGLISH,
                TranslationQuality.BUSINESS,
                context=test_case["context"]
            )
            
            # Evaluate results
            test_result = {
                "category": test_case["category"],
                "original_japanese": test_case["japanese"],
                "expected_english": test_case["expected_english"],
                "translated_english": ja_to_en_result["translated_text"],
                "back_translated_japanese": en_to_ja_result["translated_text"],
                "ja_to_en_confidence": ja_to_en_result["confidence"],
                "en_to_ja_confidence": en_to_ja_result["confidence"],
                "translation_method": ja_to_en_result.get("method", "unknown"),
                "context": test_case["context"]
            }
            
            # Simple quality assessment
            quality_score = self._assess_translation_quality(test_case, ja_to_en_result)
            test_result["quality_score"] = quality_score
            test_result["passed"] = quality_score >= 0.7
            
            if test_result["passed"]:
                passed_tests += 1
            
            self.results.append(test_result)
            
            # Log results
            logger.info(f"  Original: {test_case['japanese']}")
            logger.info(f"  Translated: {ja_to_en_result['translated_text']}")
            logger.info(f"  Confidence: {ja_to_en_result['confidence']:.2f}")
            logger.info(f"  Quality Score: {quality_score:.2f}")
            logger.info(f"  Status: {'PASS' if test_result['passed'] else 'FAIL'}")
            logger.info("")
        
        # Calculate overall results
        overall_results = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": passed_tests / total_tests,
            "average_confidence": sum(r["ja_to_en_confidence"] for r in self.results) / total_tests,
            "average_quality_score": sum(r["quality_score"] for r in self.results) / total_tests,
            "results_by_category": self._group_results_by_category()
        }
        
        logger.info("=== TRANSLATION ACCURACY TEST RESULTS ===")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed Tests: {passed_tests}")
        logger.info(f"Pass Rate: {overall_results['pass_rate']:.1%}")
        logger.info(f"Average Confidence: {overall_results['average_confidence']:.2f}")
        logger.info(f"Average Quality Score: {overall_results['average_quality_score']:.2f}")
        
        return overall_results
    
    def _assess_translation_quality(self, test_case: Dict[str, Any], result: Dict[str, Any]) -> float:
        """Assess translation quality using simple heuristics"""
        score = 0.0
        
        # Base score from confidence
        score += result["confidence"] * 0.4
        
        # Length similarity (translations should be roughly similar length)
        original_len = len(test_case["japanese"])
        translated_len = len(result["translated_text"])
        length_ratio = min(original_len, translated_len) / max(original_len, translated_len)
        score += length_ratio * 0.2
        
        # Check if translation contains English characters (for ja->en)
        english_chars = sum(1 for c in result["translated_text"] if c.isalpha() and ord(c) < 128)
        english_ratio = english_chars / len(result["translated_text"]) if result["translated_text"] else 0
        if english_ratio > 0.5:
            score += 0.2
        
        # Check if translation is different from original (not just copied)
        if result["translated_text"].lower() != test_case["japanese"].lower():
            score += 0.2
        
        return min(1.0, score)
    
    def _group_results_by_category(self) -> Dict[str, Dict[str, Any]]:
        """Group results by document category"""
        categories = {}
        
        for result in self.results:
            category = result["category"]
            if category not in categories:
                categories[category] = {
                    "tests": 0,
                    "passed": 0,
                    "total_confidence": 0,
                    "total_quality": 0
                }
            
            cat_data = categories[category]
            cat_data["tests"] += 1
            if result["passed"]:
                cat_data["passed"] += 1
            cat_data["total_confidence"] += result["ja_to_en_confidence"]
            cat_data["total_quality"] += result["quality_score"]
        
        # Calculate averages
        for category, data in categories.items():
            data["pass_rate"] = data["passed"] / data["tests"]
            data["avg_confidence"] = data["total_confidence"] / data["tests"]
            data["avg_quality"] = data["total_quality"] / data["tests"]
        
        return categories
    
    def save_results(self, filename: str = "translation_test_results.json"):
        """Save test results to JSON file"""
        output = {
            "test_summary": {
                "total_tests": len(self.results),
                "passed_tests": sum(1 for r in self.results if r["passed"]),
                "pass_rate": sum(1 for r in self.results if r["passed"]) / len(self.results),
                "average_confidence": sum(r["ja_to_en_confidence"] for r in self.results) / len(self.results),
                "average_quality_score": sum(r["quality_score"] for r in self.results) / len(self.results)
            },
            "detailed_results": self.results,
            "category_breakdown": self._group_results_by_category()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Test results saved to {filename}")


def main():
    """Run translation accuracy tests"""
    tester = TranslationAccuracyTester()
    
    try:
        results = tester.test_translation_quality()
        tester.save_results()
        
        # Print summary
        print("\n" + "="*50)
        print("TRANSLATION ACCURACY TEST SUMMARY")
        print("="*50)
        print(f"Pass Rate: {results['pass_rate']:.1%}")
        print(f"Average Confidence: {results['average_confidence']:.2f}")
        print(f"Average Quality Score: {results['average_quality_score']:.2f}")
        
        print("\nResults by Category:")
        for category, data in results["results_by_category"].items():
            print(f"  {category}: {data['pass_rate']:.1%} pass rate, {data['avg_confidence']:.2f} confidence")
        
        if results['pass_rate'] >= 0.8:
            print("\n✅ Translation service meets quality requirements!")
        else:
            print("\n⚠️  Translation service needs improvement.")
            
    except Exception as e:
        logger.error(f"Translation accuracy test failed: {e}")
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    main()