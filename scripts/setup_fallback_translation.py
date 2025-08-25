#!/usr/bin/env python3
"""
Setup script for Helsinki-NLP fallback translation models
"""
import subprocess
import sys
from pathlib import Path
from utils.logger import get_logger

logger = get_logger("setup_fallback_translation")


def install_requirements():
    """Install required packages for fallback translation"""
    packages = [
        "transformers==4.30.0",
        "torch==2.0.0", 
        "sentencepiece==0.1.99"
    ]
    
    logger.info("Installing fallback translation requirements...")
    
    for package in packages:
        try:
            logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logger.info(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install {package}: {e}")
            return False
    
    return True


def download_models():
    """Download Helsinki-NLP models"""
    try:
        from transformers import MarianMTModel, MarianTokenizer
        
        models = [
            "Helsinki-NLP/opus-mt-ja-en",
            "Helsinki-NLP/opus-mt-en-ja"
        ]
        
        for model_name in models:
            logger.info(f"Downloading {model_name}...")
            
            # Download tokenizer and model
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            
            logger.info(f"✅ {model_name} downloaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download models: {e}")
        return False


def test_fallback_translation():
    """Test the fallback translation system"""
    try:
        from services.fallback_translation import fallback_translation_service
        
        logger.info("Testing fallback translation...")
        
        # Test Japanese to English
        ja_text = "こんにちは、世界"
        result_ja_en = fallback_translation_service.translate_ja_to_en(ja_text)
        
        if result_ja_en["confidence"] > 0:
            logger.info(f"✅ JA->EN: '{ja_text}' -> '{result_ja_en['translated_text']}'")
        else:
            logger.error("❌ Japanese to English translation failed")
            return False
        
        # Test English to Japanese
        en_text = "Hello, world"
        result_en_ja = fallback_translation_service.translate_en_to_ja(en_text)
        
        if result_en_ja["confidence"] > 0:
            logger.info(f"✅ EN->JA: '{en_text}' -> '{result_en_ja['translated_text']}'")
        else:
            logger.error("❌ English to Japanese translation failed")
            return False
        
        logger.info("✅ Fallback translation system working correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Fallback translation test failed: {e}")
        return False


def main():
    """Main setup function"""
    print("Setting up Helsinki-NLP fallback translation...")
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        return False
    
    # Download models
    if not download_models():
        print("❌ Failed to download models")
        return False
    
    # Test system
    if not test_fallback_translation():
        print("❌ Fallback translation test failed")
        return False
    
    print("✅ Helsinki-NLP fallback translation setup complete!")
    print("\nTo enable fallback translation, uncomment the transformers dependencies in requirements.txt")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)