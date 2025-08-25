#!/usr/bin/env python3
"""
Demo script for Intelligent Local Translation System
Showcases key features: context-aware translation, quality assessment, caching, and document translation
"""
import time
from pathlib import Path

from services.translation_service import (
    translation_service, 
    Language, 
    TranslationQuality, 
    TranslationContext
)
from utils.logger import get_logger

logger = get_logger("translation_demo")


def demo_context_aware_translation():
    """Demonstrate context-aware translation with terminology preservation"""
    print("\n🎯 Context-Aware Translation Demo")
    print("=" * 50)
    
    test_cases = [
        {
            "text": "データベースサーバーの設定を変更する前に、必ずバックアップを作成してください。APIキーの設定は管理者権限が必要です。",
            "context": TranslationContext.TECHNICAL_MANUAL,
            "description": "Technical Manual"
        },
        {
            "text": "第3四半期の売上は前年同期比15%増の5億円となりました。営業利益は8000万円で、目標を上回る結果となっています。",
            "context": TranslationContext.FINANCIAL_REPORT,
            "description": "Financial Report"
        },
        {
            "text": "田中様、いつもお世話になっております。来週の会議の件でご連絡いたします。資料の準備ができましたので、事前にご確認いただけますでしょうか。",
            "context": TranslationContext.EMAIL_COMMUNICATION,
            "description": "Business Email"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['description']}:")
        print(f"Original (Japanese): {case['text']}")
        
        result = translation_service.translate(
            case['text'],
            target_language=Language.ENGLISH,
            quality=TranslationQuality.BUSINESS,
            context_type=case['context'],
            preserve_terminology=True
        )
        
        print(f"Translation: {result['translated_text']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Quality: {result['quality_assessment']}")
        print(f"Preserved Terms: {result.get('preserved_terms', 0)}")
        print(f"Method: {result.get('method', 'unknown')}")


def demo_translation_caching():
    """Demonstrate translation caching for performance"""
    print("\n⚡ Translation Caching Demo")
    print("=" * 50)
    
    test_text = "これはキャッシュのデモンストレーションです。同じテキストを複数回翻訳して、キャッシュの効果を確認します。"
    
    print(f"Test Text: {test_text}")
    
    # First translation (not cached)
    print("\n1. First translation (not cached):")
    start_time = time.time()
    result1 = translation_service.translate(
        test_text,
        target_language=Language.ENGLISH,
        quality=TranslationQuality.BUSINESS
    )
    first_duration = time.time() - start_time
    
    print(f"Translation: {result1['translated_text']}")
    print(f"Time: {first_duration:.3f} seconds")
    print(f"Cached: {result1.get('cached', False)}")
    
    # Second translation (should be cached)
    print("\n2. Second translation (should be cached):")
    start_time = time.time()
    result2 = translation_service.translate(
        test_text,
        target_language=Language.ENGLISH,
        quality=TranslationQuality.BUSINESS
    )
    second_duration = time.time() - start_time
    
    print(f"Translation: {result2['translated_text']}")
    print(f"Time: {second_duration:.3f} seconds")
    print(f"Cached: {result2.get('cached', False)}")
    
    if second_duration > 0:
        speed_improvement = first_duration / second_duration
        print(f"Speed Improvement: {speed_improvement:.1f}x faster")


def demo_quality_assessment():
    """Demonstrate translation quality assessment"""
    print("\n📊 Quality Assessment Demo")
    print("=" * 50)
    
    test_cases = [
        {
            "text": "こんにちは、元気ですか？",
            "description": "Simple greeting"
        },
        {
            "text": "弊社の新しいクラウドベースのソリューションは、企業のデジタル変革を支援し、業務効率を30%向上させることができます。",
            "description": "Business proposal"
        },
        {
            "text": "システムのパフォーマンスを最適化するために、データベースのインデックスを再構築し、クエリの実行時間を短縮する必要があります。",
            "description": "Technical content"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['description']}:")
        print(f"Original: {case['text']}")
        
        result = translation_service.translate(
            case['text'],
            target_language=Language.ENGLISH,
            quality=TranslationQuality.BUSINESS
        )
        
        validation = translation_service.validate_translation_quality(
            case['text'],
            result['translated_text'],
            Language.ENGLISH
        )
        
        print(f"Translation: {result['translated_text']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Quality Assessment: {result['quality_assessment']}")
        print(f"Validation Score: {validation['overall_score']:.3f}")
        print(f"Issues: {len(validation['issues'])}")


def demo_document_translation():
    """Demonstrate document-to-document translation"""
    print("\n📄 Document Translation Demo")
    print("=" * 50)
    
    # Create a sample Japanese document
    sample_doc_path = Path("sample_business_doc_ja.txt")
    sample_content = """
会社概要

株式会社テクノロジーソリューションズは、2020年に設立された革新的なIT企業です。

事業内容：
• ソフトウェア開発
• クラウドソリューション
• データ分析サービス
• AI・機械学習コンサルティング

財務情報（2023年度）：
• 売上高: 12億円
• 営業利益: 2億4000万円
• 従業員数: 85名
• 本社所在地: 東京都渋谷区

技術スタック：
• プログラミング言語: Python, JavaScript, Go
• データベース: PostgreSQL, MongoDB, Redis
• クラウドプラットフォーム: AWS, Azure
• 開発ツール: Docker, Kubernetes, Git

お問い合わせ：
Email: info@techsolutions.co.jp
電話: 03-1234-5678
"""
    
    try:
        # Create sample document
        with open(sample_doc_path, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        
        print(f"Created sample document: {sample_doc_path}")
        print(f"Document size: {len(sample_content)} characters")
        
        # Translate document
        print("\nTranslating document...")
        result = translation_service.translate_document_to_document(
            str(sample_doc_path),
            Language.ENGLISH
        )
        
        if result.get("success", False):
            print(f"✅ Translation successful!")
            print(f"Output file: {result['output_document']}")
            print(f"Source language: {result['source_language']}")
            print(f"Target language: {result['target_language']}")
            print(f"Translation confidence: {result['translation_confidence']:.3f}")
            print(f"Context type: {result['context_type']}")
            print(f"Preserved terms: {result['preserved_terms']}")
            
            # Show a preview of the translated content
            output_path = Path(result['output_document'])
            if output_path.exists():
                with open(output_path, 'r', encoding='utf-8') as f:
                    translated_content = f.read()
                
                print(f"\nTranslated content preview (first 300 characters):")
                print("-" * 50)
                print(translated_content[:300] + "..." if len(translated_content) > 300 else translated_content)
                
                # Clean up
                output_path.unlink()
        else:
            print(f"❌ Translation failed: {result.get('error', 'Unknown error')}")
        
        # Clean up sample document
        sample_doc_path.unlink()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        if sample_doc_path.exists():
            sample_doc_path.unlink()


def demo_system_statistics():
    """Show system statistics and capabilities"""
    print("\n📈 System Statistics")
    print("=" * 50)
    
    stats = translation_service.get_translation_statistics()
    
    print("Cache Statistics:")
    cache_stats = stats.get("cache_statistics", {})
    print(f"  • Total cached translations: {cache_stats.get('total_cached_translations', 0)}")
    print(f"  • Average confidence: {cache_stats.get('average_confidence', 0):.3f}")
    print(f"  • Total cache uses: {cache_stats.get('total_cache_uses', 0)}")
    print(f"  • Language pairs supported: {cache_stats.get('supported_language_pairs', 0)}")
    
    print(f"\nSystem Capabilities:")
    print(f"  • Technical terms loaded: {stats.get('technical_terms_loaded', 0)}")
    print(f"  • Fallback translation available: {'Yes' if stats.get('fallback_available', False) else 'No'}")
    
    print(f"\nQuality Distribution:")
    quality_dist = stats.get("quality_distribution", {})
    for quality, count in quality_dist.items():
        print(f"  • {quality.title()}: {count}")
    
    print(f"\nTranslation Methods:")
    method_dist = stats.get("method_distribution", {})
    for method, count in method_dist.items():
        print(f"  • {method}: {count}")
    
    print(f"\nSupported Languages:")
    languages = translation_service.get_supported_languages()
    for lang in languages:
        print(f"  • {lang['name']} ({lang['native_name']}) - {lang['code']}")
    
    print(f"\nSupported Contexts:")
    contexts = translation_service.get_supported_contexts()
    for context in contexts[:5]:  # Show first 5
        print(f"  • {context['name']}")
    print(f"  ... and {len(contexts) - 5} more")


def main():
    """Run the intelligent translation system demo"""
    print("🚀 Intelligent Local Translation System Demo")
    print("=" * 60)
    print("Showcasing advanced features:")
    print("• Context-aware translation with terminology preservation")
    print("• Translation quality assessment and validation")
    print("• Performance caching for consistency")
    print("• Document-to-document translation")
    print("• System statistics and monitoring")
    
    try:
        # Run demos
        demo_context_aware_translation()
        demo_translation_caching()
        demo_quality_assessment()
        demo_document_translation()
        demo_system_statistics()
        
        print("\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        print("🎯 Key features demonstrated:")
        print("   • Context-aware translation preserves technical terminology")
        print("   • Caching provides significant performance improvements")
        print("   • Quality assessment ensures translation reliability")
        print("   • Document translation supports business workflows")
        print("   • System monitoring provides operational insights")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())