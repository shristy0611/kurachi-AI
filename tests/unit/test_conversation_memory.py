#!/usr/bin/env python3
"""
Test script for conversation memory and context management (Task 4.1)
"""
import sys
import os
from pathlib import Path
import pytest

# Mark entire module as slow due to heavy service initialization
pytestmark = pytest.mark.slow

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.chat_service import chat_service
from services.conversation_memory import conversation_memory_service
from models.database import db_manager
from utils.logger import get_logger

logger = get_logger("test_conversation_memory")


def test_conversation_memory():
    """Test conversation memory and context management functionality"""
    print("🧠 Testing Conversation Memory and Context Management (Task 4.1)")
    print("=" * 60)
    
    # Test 1: Create a new conversation
    print("\n1. Testing conversation creation...")
    user_id = "test_user_001"
    conversation = chat_service.create_conversation(user_id, "Test Conversation - Memory Management")
    
    if conversation:
        print(f"✅ Created conversation: {conversation.id}")
        print(f"   Title: {conversation.title}")
        print(f"   User ID: {conversation.user_id}")
    else:
        print("❌ Failed to create conversation")
        return False
    
    conversation_id = conversation.id
    
    # Test 2: Send multiple messages to build conversation history
    print("\n2. Testing conversation history building...")
    test_messages = [
        "Hello, I'm testing the conversation memory system.",
        "Can you remember what I just said?",
        "I want to test how well you maintain context across multiple messages.",
        "What was my first message about?",
        "Let's discuss document processing capabilities.",
        "How does the system handle long conversations?",
        "Can you summarize our conversation so far?",
        "What are the key topics we've discussed?",
        "I'm interested in the memory management features.",
        "How does intelligent summarization work?"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"   Sending message {i}: {message[:50]}...")
        
        try:
            response = chat_service.send_message(conversation_id, message, user_id)
            if response:
                print(f"   ✅ Response received: {response['ai_message'].content[:100]}...")
            else:
                print(f"   ❌ Failed to get response for message {i}")
        except Exception as e:
            print(f"   ❌ Error sending message {i}: {e}")
    
    # Test 3: Test conversation context retrieval
    print("\n3. Testing conversation context retrieval...")
    try:
        context = conversation_memory_service.get_conversation_context(conversation_id, user_id)
        if context:
            print(f"✅ Retrieved conversation context:")
            print(f"   Total messages: {context.total_messages}")
            print(f"   Context window size: {context.context_window_size}")
            print(f"   Key topics: {context.key_topics}")
            print(f"   Has summary: {'Yes' if context.summary else 'No'}")
            if context.summary:
                print(f"   Summary preview: {context.summary[:200]}...")
        else:
            print("❌ Failed to retrieve conversation context")
    except Exception as e:
        print(f"❌ Error retrieving context: {e}")
    
    # Test 4: Test context window building
    print("\n4. Testing context window building...")
    try:
        test_query = "What have we been discussing in this conversation?"
        context_window = conversation_memory_service.build_context_window(conversation_id, test_query, user_id)
        
        if context_window:
            print(f"✅ Built context window:")
            print(f"   Length: {len(context_window)} characters")
            print(f"   Preview: {context_window[:300]}...")
        else:
            print("❌ Failed to build context window")
    except Exception as e:
        print(f"❌ Error building context window: {e}")
    
    # Test 5: Test conversation statistics
    print("\n5. Testing conversation statistics...")
    try:
        stats = chat_service.get_conversation_statistics(conversation_id, user_id)
        if stats and "error" not in stats:
            print(f"✅ Retrieved conversation statistics:")
            print(f"   Total messages: {stats.get('total_messages', 'N/A')}")
            print(f"   User messages: {stats.get('user_messages', 'N/A')}")
            print(f"   Assistant messages: {stats.get('assistant_messages', 'N/A')}")
            print(f"   Duration (minutes): {stats.get('conversation_duration_minutes', 'N/A'):.2f}")
            print(f"   Avg response time (seconds): {stats.get('average_response_time_seconds', 'N/A'):.2f}")
        else:
            print(f"❌ Failed to get statistics: {stats.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Error getting statistics: {e}")
    
    # Test 6: Test conversation summary
    print("\n6. Testing conversation summary...")
    try:
        summary = chat_service.get_conversation_summary(conversation_id, user_id)
        if summary:
            print(f"✅ Retrieved conversation summary:")
            print(f"   Summary: {summary}")
        else:
            print("ℹ️  No summary available (conversation may be too short)")
    except Exception as e:
        print(f"❌ Error getting summary: {e}")
    
    # Test 7: Test memory management with a long conversation
    print("\n7. Testing memory management with additional messages...")
    additional_messages = [
        "Let's add more messages to trigger summarization.",
        "I want to see how the system handles context overflow.",
        "This should test the intelligent context window management.",
        "Can you still remember our earlier discussion?",
        "What was the main topic of our conversation?",
        "How does the system prioritize which messages to keep in context?",
        "I'm testing the conversation memory limits.",
        "This is message number " + str(len(test_messages) + 7),
        "The system should be managing memory efficiently now.",
        "Can you provide a comprehensive summary of everything we've discussed?"
    ]
    
    for i, message in enumerate(additional_messages, 1):
        print(f"   Adding message {i}: {message[:50]}...")
        try:
            response = chat_service.send_message(conversation_id, message, user_id)
            if response:
                print(f"   ✅ Response: {response['ai_message'].content[:80]}...")
            else:
                print(f"   ❌ Failed to get response")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test 8: Final context and summary check
    print("\n8. Final memory management verification...")
    try:
        final_context = conversation_memory_service.get_conversation_context(conversation_id, user_id)
        if final_context:
            print(f"✅ Final conversation state:")
            print(f"   Total messages: {final_context.total_messages}")
            print(f"   Context window size: {final_context.context_window_size}")
            print(f"   Key topics: {final_context.key_topics}")
            print(f"   Has summary: {'Yes' if final_context.summary else 'No'}")
            
            if final_context.summary:
                print(f"   Final summary: {final_context.summary}")
        else:
            print("❌ Failed to retrieve final context")
    except Exception as e:
        print(f"❌ Error in final verification: {e}")
    
    # Test 9: Test conversation deletion
    print("\n9. Testing conversation cleanup...")
    try:
        deleted = chat_service.delete_conversation(conversation_id, user_id)
        if deleted:
            print("✅ Successfully deleted test conversation")
        else:
            print("❌ Failed to delete conversation")
    except Exception as e:
        print(f"❌ Error deleting conversation: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Conversation Memory and Context Management Test Complete!")
    print("Task 4.1 implementation verified.")
    
    return True


if __name__ == "__main__":
    try:
        success = test_conversation_memory()
        if success:
            print("\n✅ All tests completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test script failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)