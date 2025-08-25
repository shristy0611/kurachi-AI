"""
Kurachi AI - Main Application Entry Point (SOTA Enhanced)
Enhanced bilingual RAG chatbot with business-grade features
Now using SOTA architecture for optimal performance
"""
import sys
import os
from pathlib import Path
import time

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# SOTA imports for modern architecture
from services.sota_config import sota_config, get_config, validate_config
from services.sota_registry import sota_registry, get_performance_summary
from services.sota_observability import sota_observability, trace_operation, record_metric, MetricType
from utils.logger import get_logger
from ui.streamlit_app import main as streamlit_main

logger = get_logger("main")


def main():
    """Main entry point for Kurachi AI with SOTA enhancements"""
    startup_start = time.time()
    
    with trace_operation("application_startup", component="main"):
        try:
            # Get SOTA configuration
            config = get_config()
            
            # Validate configuration
            config_issues = validate_config()
            if config_issues:
                logger.warning(f"Configuration issues found: {len(config_issues)}")
                for issue in config_issues:
                    logger.warning(f"  - {issue}")
            
            logger.info(f"Starting Kurachi AI with SOTA architecture")
            logger.info(f"Environment: {getattr(config, 'environment', 'unknown')}")
            logger.info(f"Debug mode: {getattr(config, 'debug', False)}")
            
            # Initialize SOTA services registry
            logger.info("Initializing SOTA service registry...")
            registry_start = time.time()
            
            # Pre-load critical services for faster UI startup
            sota_registry.get_service('sota_config')
            sota_registry.get_service('sota_observability')
            
            registry_time = (time.time() - registry_start) * 1000
            logger.info(f"Service registry initialized in {registry_time:.2f}ms")
            
            # Check if Ollama is accessible using SOTA config
            try:
                from langchain_community.llms import Ollama
                ai_config = getattr(config, 'ai', None)
                if ai_config:
                    test_llm = Ollama(
                        model=getattr(ai_config, 'llm_model', 'qwen3:4b'),
                        base_url=getattr(ai_config, 'ollama_base_url', 'http://localhost:11434')
                    )
                    logger.info("Ollama connection verified with SOTA configuration")
                else:
                    logger.warning("AI configuration not found, using defaults")
            except Exception as e:
                logger.error("Failed to connect to Ollama", error=e)
                ai_config = getattr(config, 'ai', None)
                base_url = getattr(ai_config, 'ollama_base_url', 'http://localhost:11434') if ai_config else 'http://localhost:11434'
                llm_model = getattr(ai_config, 'llm_model', 'qwen3:4b') if ai_config else 'qwen3:4b'
                
                print(f"Error: Cannot connect to Ollama at {base_url}")
                print("Please ensure Ollama is running and the models are available:")
                print(f"  - {llm_model}")
                print(f"  - {getattr(ai_config, 'embedding_model', 'nomic-embed-text') if ai_config else 'nomic-embed-text'}")
                
                # Record startup failure
                record_metric("startup_failure", 1, MetricType.COUNTER, {"reason": "ollama_connection"})
                sys.exit(1)
            
            # Record startup performance
            startup_time = (time.time() - startup_start) * 1000
            record_metric("startup_time_ms", startup_time, MetricType.HISTOGRAM)
            
            # Get performance summary
            perf_summary = get_performance_summary()
            logger.info(f"SOTA Performance Summary:")
            logger.info(f"  - Services available: {perf_summary.get('total_services', 0)}")
            logger.info(f"  - Services loaded: {perf_summary.get('loaded_services', 0)}")
            logger.info(f"  - {perf_summary.get('load_efficiency', 'Efficiency metrics available')}")
            logger.info(f"  - Total startup time: {startup_time:.2f}ms")
            
            # Start Streamlit app with SOTA enhancements
            logger.info("Starting Streamlit interface with SOTA features...")
            streamlit_main()
            
        except KeyboardInterrupt:
            logger.info("Application stopped by user")
            record_metric("shutdown_reason", 1, MetricType.COUNTER, {"reason": "user_interrupt"})
            sys.exit(0)
        except Exception as e:
            logger.error("Application failed to start", error=e)
            record_metric("startup_failure", 1, MetricType.COUNTER, {"reason": "exception", "error": str(e)})
            
            config = get_config()
            if getattr(config, 'debug', False):
                raise
            sys.exit(1)


if __name__ == "__main__":
    main()