#!/usr/bin/env python3
"""
Real Data Accuracy Test for Intelligent Chunking System
Tests the system against 4 real documents to measure accuracy and performance
"""
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Any

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from langchain.schema import Document as LangChainDocument
from services.intelligent_chunking import intelligent_chunking_service
from services.document_ingestion import enhanced_ingestion_service
from services.document_processors import processor_factory

class DocumentAccuracyTester:
    """Test accuracy of document processing and intelligent chunking"""
    
    def __init__(self):
        self.test_data_dir = Path("tests/test_data")
        self.results = {}
        
    def test_all_documents(self) -> Dict[str, Any]:
        """Test all documents in the test data directory"""
        print("🧪 Real Data Accuracy Test - Intelligent Chunking System")
        print("=" * 70)
        
        if not self.test_data_dir.exists():
            print(f"❌ Test data directory not found: {self.test_data_dir}")
            return {}
        
        test_files = list(self.test_data_dir.glob("*"))
        print(f"📁 Found {len(test_files)} test files:")
        for file in test_files:
            print(f"  - {file.name}")
        
        print(f"\n{'='*70}")
        
        # Test each file
        for test_file in test_files:
            if test_file.is_file():
                print(f"\n🔍 Testing: {test_file.name}")
                print("-" * 50)
                result = self.test_single_document(test_file)
                self.results[test_file.name] = result
        
        return self.results
    
    def test_single_document(self, file_path: Path) -> Dict[str, Any]:
        """Test a single document and return detailed results"""
        result = {
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "file_type": file_path.suffix.lower(),
            "processing_success": False,
            "processing_time": 0.0,
            "error_message": None,
            "processor_used": None,
            "extraction_method": None,
            "chunks_created": 0,
            "chunk_types": {},
            "content_preview": "",
            "accuracy_assessment": {},
            "issues_found": []
        }
        
        try:
            start_time = time.time()
            
            # Step 1: File validation
            print("1️⃣ Validating file...")
            validation = enhanced_ingestion_service.validate_file(str(file_path))
            
            if not validation["valid"]:
                result["error_message"] = f"Validation failed: {validation['errors']}"
                print(f"❌ {result['error_message']}")
                return result
            
            print(f"✅ File validation passed")
            result["file_info"] = validation["file_info"]
            
            # Step 2: Get processor
            print("2️⃣ Getting document processor...")
            processor = processor_factory.get_processor(str(file_path))
            
            if not processor:
                result["error_message"] = f"No processor available for {file_path.suffix}"
                print(f"❌ {result['error_message']}")
                return result
            
            result["processor_used"] = processor.__class__.__name__
            print(f"✅ Using processor: {result['processor_used']}")
            
            # Step 3: Process document
            print("3️⃣ Processing document...")
            processing_result = processor.process(str(file_path))
            
            if not processing_result.success:
                result["error_message"] = processing_result.error_message
                print(f"❌ Processing failed: {result['error_message']}")
                return result
            
            result["extraction_method"] = processing_result.metadata.get("extraction_method", "unknown")
            result["processing_time"] = processing_result.processing_time or 0.0
            
            print(f"✅ Document processed successfully")
            print(f"   Method: {result['extraction_method']}")
            print(f"   Time: {result['processing_time']:.2f}s")
            print(f"   Documents extracted: {len(processing_result.documents)}")
            
            # Step 4: Intelligent chunking
            print("4️⃣ Applying intelligent chunking...")
            
            if not processing_result.documents:
                result["error_message"] = "No documents extracted from file"
                print(f"❌ {result['error_message']}")
                return result
            
            # Add metadata to documents
            for doc in processing_result.documents:
                doc.metadata.update({
                    "source": str(file_path),
                    "filename": file_path.name,
                    "file_type": file_path.suffix.lower(),
                    "extraction_method": result["extraction_method"]
                })
            
            # Apply intelligent chunking
            chunks = intelligent_chunking_service.chunk_documents(processing_result.documents)
            
            result["chunks_created"] = len(chunks)
            result["processing_success"] = True
            
            print(f"✅ Intelligent chunking completed")
            print(f"   Chunks created: {len(chunks)}")
            
            # Step 5: Analyze chunks
            print("5️⃣ Analyzing chunk quality...")
            self.analyze_chunks(chunks, result)
            
            # Step 6: Content preview
            if chunks:
                first_chunk_content = chunks[0].page_content[:200]
                result["content_preview"] = first_chunk_content.replace('\n', ' ')
                print(f"📄 Content preview: {result['content_preview']}...")
            
            result["processing_time"] = time.time() - start_time
            
            return result
            
        except Exception as e:
            result["error_message"] = str(e)
            result["processing_time"] = time.time() - start_time
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return result
    
    def analyze_chunks(self, chunks: List[LangChainDocument], result: Dict[str, Any]):
        """Analyze chunk quality and accuracy"""
        if not chunks:
            result["issues_found"].append("No chunks created")
            return
        
        # Get chunk statistics
        stats = intelligent_chunking_service.get_chunk_statistics(chunks)
        result["chunk_types"] = stats.get("chunk_types", {})
        
        print(f"   Chunk types: {result['chunk_types']}")
        print(f"   Total words: {stats.get('total_words', 0)}")
        print(f"   Avg words/chunk: {stats.get('average_words_per_chunk', 0):.1f}")
        
        # Accuracy assessment
        accuracy = {
            "total_chunks": len(chunks),
            "chunks_with_metadata": 0,
            "chunks_with_source": 0,
            "chunks_with_type": 0,
            "chunks_with_content": 0,
            "empty_chunks": 0,
            "very_short_chunks": 0,
            "very_long_chunks": 0,
            "metadata_richness_score": 0.0,
            "content_quality_score": 0.0
        }
        
        for chunk in chunks:
            # Check metadata completeness
            if chunk.metadata:
                accuracy["chunks_with_metadata"] += 1
                
                if chunk.metadata.get("source"):
                    accuracy["chunks_with_source"] += 1
                
                if chunk.metadata.get("chunk_type"):
                    accuracy["chunks_with_type"] += 1
            
            # Check content quality
            content = chunk.page_content.strip()
            if content:
                accuracy["chunks_with_content"] += 1
                
                word_count = len(content.split())
                if word_count == 0:
                    accuracy["empty_chunks"] += 1
                elif word_count < 5:
                    accuracy["very_short_chunks"] += 1
                elif word_count > 500:
                    accuracy["very_long_chunks"] += 1
        
        # Calculate scores
        total_chunks = len(chunks)
        if total_chunks > 0:
            accuracy["metadata_richness_score"] = (
                accuracy["chunks_with_metadata"] + 
                accuracy["chunks_with_source"] + 
                accuracy["chunks_with_type"]
            ) / (total_chunks * 3) * 100
            
            accuracy["content_quality_score"] = (
                accuracy["chunks_with_content"] - 
                accuracy["empty_chunks"] - 
                accuracy["very_short_chunks"] * 0.5
            ) / total_chunks * 100
        
        result["accuracy_assessment"] = accuracy
        
        # Identify issues
        issues = []
        if accuracy["empty_chunks"] > 0:
            issues.append(f"{accuracy['empty_chunks']} empty chunks")
        
        if accuracy["very_short_chunks"] > total_chunks * 0.3:
            issues.append(f"Too many short chunks ({accuracy['very_short_chunks']})")
        
        if accuracy["chunks_with_metadata"] < total_chunks * 0.8:
            issues.append("Poor metadata coverage")
        
        if accuracy["content_quality_score"] < 70:
            issues.append("Low content quality score")
        
        result["issues_found"] = issues
        
        print(f"   Metadata richness: {accuracy['metadata_richness_score']:.1f}%")
        print(f"   Content quality: {accuracy['content_quality_score']:.1f}%")
        
        if issues:
            print(f"   ⚠️  Issues: {', '.join(issues)}")
        else:
            print(f"   ✅ No major issues found")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a comprehensive summary report"""
        if not self.results:
            return {}
        
        print(f"\n{'='*70}")
        print("📊 COMPREHENSIVE ACCURACY REPORT")
        print("=" * 35)
        
        summary = {
            "total_files_tested": len(self.results),
            "successful_processing": 0,
            "failed_processing": 0,
            "total_chunks_created": 0,
            "average_processing_time": 0.0,
            "file_type_performance": {},
            "processor_performance": {},
            "overall_accuracy_score": 0.0,
            "recommendations": []
        }
        
        processing_times = []
        accuracy_scores = []
        
        for file_name, result in self.results.items():
            print(f"\n📄 {file_name}")
            print(f"   Status: {'✅ SUCCESS' if result['processing_success'] else '❌ FAILED'}")
            
            if result["processing_success"]:
                summary["successful_processing"] += 1
                summary["total_chunks_created"] += result["chunks_created"]
                processing_times.append(result["processing_time"])
                
                # File type performance
                file_type = result["file_type"]
                if file_type not in summary["file_type_performance"]:
                    summary["file_type_performance"][file_type] = {
                        "count": 0, "success": 0, "avg_chunks": 0, "avg_time": 0
                    }
                
                perf = summary["file_type_performance"][file_type]
                perf["count"] += 1
                perf["success"] += 1
                perf["avg_chunks"] += result["chunks_created"]
                perf["avg_time"] += result["processing_time"]
                
                # Processor performance
                processor = result["processor_used"]
                if processor not in summary["processor_performance"]:
                    summary["processor_performance"][processor] = {
                        "count": 0, "success": 0, "total_chunks": 0
                    }
                
                proc_perf = summary["processor_performance"][processor]
                proc_perf["count"] += 1
                proc_perf["success"] += 1
                proc_perf["total_chunks"] += result["chunks_created"]
                
                # Accuracy assessment
                accuracy = result.get("accuracy_assessment", {})
                if accuracy:
                    metadata_score = accuracy.get("metadata_richness_score", 0)
                    content_score = accuracy.get("content_quality_score", 0)
                    overall_score = (metadata_score + content_score) / 2
                    accuracy_scores.append(overall_score)
                    
                    print(f"   Chunks: {result['chunks_created']}")
                    print(f"   Time: {result['processing_time']:.2f}s")
                    print(f"   Accuracy: {overall_score:.1f}%")
                    
                    if result["issues_found"]:
                        print(f"   Issues: {', '.join(result['issues_found'])}")
                
            else:
                summary["failed_processing"] += 1
                print(f"   Error: {result['error_message']}")
        
        # Calculate averages
        if processing_times:
            summary["average_processing_time"] = sum(processing_times) / len(processing_times)
        
        if accuracy_scores:
            summary["overall_accuracy_score"] = sum(accuracy_scores) / len(accuracy_scores)
        
        # Finalize file type performance
        for file_type, perf in summary["file_type_performance"].items():
            if perf["count"] > 0:
                perf["avg_chunks"] = perf["avg_chunks"] / perf["count"]
                perf["avg_time"] = perf["avg_time"] / perf["count"]
        
        # Generate recommendations
        recommendations = []
        
        if summary["failed_processing"] > 0:
            recommendations.append(f"Fix {summary['failed_processing']} failed processing cases")
        
        if summary["overall_accuracy_score"] < 80:
            recommendations.append("Improve overall accuracy (currently below 80%)")
        
        if summary["average_processing_time"] > 10:
            recommendations.append("Optimize processing speed (currently > 10s average)")
        
        # Check for specific issues across files
        common_issues = {}
        for result in self.results.values():
            for issue in result.get("issues_found", []):
                common_issues[issue] = common_issues.get(issue, 0) + 1
        
        for issue, count in common_issues.items():
            if count > 1:
                recommendations.append(f"Address common issue: {issue} (affects {count} files)")
        
        summary["recommendations"] = recommendations
        
        # Print summary
        print(f"\n📈 OVERALL PERFORMANCE")
        print(f"   Success rate: {summary['successful_processing']}/{summary['total_files_tested']} ({summary['successful_processing']/summary['total_files_tested']*100:.1f}%)")
        print(f"   Total chunks: {summary['total_chunks_created']}")
        print(f"   Avg processing time: {summary['average_processing_time']:.2f}s")
        print(f"   Overall accuracy: {summary['overall_accuracy_score']:.1f}%")
        
        print(f"\n🔧 FILE TYPE PERFORMANCE")
        for file_type, perf in summary["file_type_performance"].items():
            print(f"   {file_type}: {perf['success']}/{perf['count']} success, {perf['avg_chunks']:.1f} avg chunks, {perf['avg_time']:.2f}s avg time")
        
        print(f"\n⚙️  PROCESSOR PERFORMANCE")
        for processor, perf in summary["processor_performance"].items():
            print(f"   {processor}: {perf['success']}/{perf['count']} success, {perf['total_chunks']} total chunks")
        
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        else:
            print(f"\n🎉 No major issues found - system performing well!")
        
        return summary


def main():
    """Run the real data accuracy test"""
    print("🚀 Starting Real Data Accuracy Test")
    print("Using Python 3.11 + Full Intelligent Chunking")
    print("=" * 70)
    
    # Check Python version
    import sys
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor != 11:
        print("⚠️  Warning: Not using Python 3.11 as expected")
    
    # Initialize tester
    tester = DocumentAccuracyTester()
    
    # Run tests
    results = tester.test_all_documents()
    
    if not results:
        print("❌ No test results generated")
        return False
    
    # Generate summary
    summary = tester.generate_summary_report()
    
    # Final assessment
    success_rate = summary.get("successful_processing", 0) / summary.get("total_files_tested", 1) * 100
    accuracy_score = summary.get("overall_accuracy_score", 0)
    
    print(f"\n{'='*70}")
    print("🎯 FINAL ASSESSMENT")
    print("=" * 20)
    
    if success_rate >= 100 and accuracy_score >= 80:
        print("🎉 EXCELLENT: System performing at high accuracy!")
        grade = "A"
    elif success_rate >= 75 and accuracy_score >= 70:
        print("✅ GOOD: System performing well with minor issues")
        grade = "B"
    elif success_rate >= 50 and accuracy_score >= 60:
        print("⚠️  FAIR: System working but needs improvement")
        grade = "C"
    else:
        print("❌ POOR: System needs significant improvement")
        grade = "D"
    
    print(f"📊 Grade: {grade}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    print(f"🎯 Accuracy Score: {accuracy_score:.1f}%")
    
    return success_rate >= 75 and accuracy_score >= 70


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)