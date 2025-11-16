import json
import os
from csv_processor import CSVProcessor
from simple_vector_database import SimpleVectorDatabase  # Changed import
from chatbot import NullclassChatbot

class NullclassChatbotSystem:
    def __init__(self):
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # Initialize components with simple vector database
        self.vector_db = SimpleVectorDatabase(  # Changed class
            index_path=self.config['vector_database']['index_path'],
            metadata_path=self.config['vector_database']['metadata_path']
        )
        
        self.csv_processor = CSVProcessor(self.config['data_sources']['csv_file'])
        self.chatbot = NullclassChatbot(
            self.vector_db, 
            similarity_threshold=self.config['chatbot']['similarity_threshold']
        )
    
    def initialize_knowledge_base(self):
        """Initialize the knowledge base from CSV"""
        print("🚀 Initializing Nullclass Chatbot Knowledge Base...")
        
        # Load and process CSV data
        documents = self.csv_processor.load_and_process_data()
        
        if documents:
            # Add to vector database
            self.vector_db.add_documents(documents)
            print("✅ Knowledge base initialized successfully!")
        else:
            print("❌ Failed to initialize knowledge base")
    
    def start_chat(self):
        """Start the chat interface"""
        print("\n" + "="*60)
        print("🤖 Nullclass Virtual Assistant")
        print("="*60)
        print("I can help you with questions about Nullclass courses!")
        print("Type 'exit' to quit, 'stats' for system info, 'history' for chat history")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() == 'exit':
                    print("Thank you for using Nullclass Assistant! 👋")
                    break
                elif user_input.lower() == 'stats':
                    self.show_stats()
                elif user_input.lower() == 'history':
                    self.show_history()
                elif user_input.lower() == 'clear':
                    self.vector_db.clear_database()
                    print("Knowledge base cleared. Re-initializing...")
                    self.initialize_knowledge_base()
                elif user_input:
                    response = self.chatbot.get_response(user_input)
                    print(f"🤖 Bot: {response}")
                else:
                    print("Please enter a question.")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def show_stats(self):
        """Show system statistics"""
        stats = self.chatbot.get_system_stats()
        csv_stats = self.csv_processor.get_statistics()
        
        print("\n📊 System Statistics:")
        print(f"   Knowledge Base Documents: {stats['knowledge_base_documents']}")
        print(f"   Conversation History: {stats['conversation_history_length']} messages")
        print(f"   Similarity Threshold: {stats['similarity_threshold']}")
        if 'total_questions' in csv_stats:
            print(f"   CSV Questions: {csv_stats['total_questions']}")
    
    def show_history(self):
        """Show conversation history"""
        history = self.chatbot.get_conversation_history()
        if not history:
            print("No conversation history yet.")
            return
            
        print("\n📝 Conversation History:")
        for i, chat in enumerate(history, 1):
            print(f"\n{i}. You: {chat['user']}")
            print(f"   Bot: {chat['bot'][:100]}..." if len(chat['bot']) > 100 else f"   Bot: {chat['bot']}")

def main():
    system = NullclassChatbotSystem()
    
    # Initialize knowledge base
    system.initialize_knowledge_base()
    
    # Start chat interface
    system.start_chat()

if __name__ == "__main__":
    main()
