import os
import json
from typing import List
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.gemini import Gemini as llm_provider
from llama_index.embeddings.gemini import GeminiEmbedding as embedding
from llama_index.core.postprocessor import SentenceTransformerRerank
import os
from dotenv import load_dotenv
from test_evalutate import run_test_evaluation
# Load environment variables from .env file
load_dotenv()

class EducationAgentRAG:
    def __init__(self, content_dir="mark_down_content", persist_dir="persisted_vector_store", metadata_path="metadata.json"):
        """
        Initialize the RAG system for Australian education agent content
        
        Args:
            content_dir: Directory containing markdown files
            persist_dir: Directory to persist vector store
            metadata_path: Path to the metadata.json file containing source URLs
        """
        self.content_dir = content_dir
        self.persist_dir = persist_dir
        self.metadata_path = metadata_path
        self.metadata = self.load_metadata()
        self.setup_llm()
        self.load_or_create_index()
        
    def load_metadata(self):
        """Load metadata mapping from file paths to URLs"""
        try:
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return {}
            
    def setup_llm(self):
        """Configure LLM and embedding settings"""
        # Configure OpenAI settings
        Settings.llm = llm_provider(model="models/gemini-2.0-flash-thinking-exp", temperature=0)
        Settings.embed_model = embedding(model="text-embedding-004")
        
    def load_documents(self) -> List[Document]:
        """Load and process markdown documents from content directory"""
        # Load markdown files from directory
        print(f"Loading documents from {self.content_dir}...")
        documents = SimpleDirectoryReader(
            self.content_dir, 
            recursive=True
        ).load_data()
        
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def load_or_create_index(self):
        """Load index from disk if it exists, otherwise create a new one"""
        try:
            if os.path.exists(self.persist_dir):
                print(f"Loading existing index from {self.persist_dir}")
                storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
                self.index = load_index_from_storage(storage_context)
            else:
                print(f"Creating new index and persisting to {self.persist_dir}")
                # Load documents
                documents = self.load_documents()
                # Create index
                self.index = VectorStoreIndex.from_documents(documents=documents)
                # Persist index
                self.index.storage_context.persist(persist_dir=self.persist_dir)
        except Exception as e:
            print(f"An error occurred: {e}")
            raise
            
    def query(self, query_text: str, num_results: int = 5, sources=False) -> str:
        """
        Query the RAG system with a question
        
        Args:
            query_text: The question to ask
            num_results: Number of top documents to retrieve
            
        Returns:
            str: The response from the RAG system
        """
        
        # Add reranking for better results
        reranker = SentenceTransformerRerank(
            model="mixedbread-ai/mxbai-rerank-xsmall-v1",
            top_n=num_results
        )
        
        # Create query engine with retrieved context
        query_engine = self.index.as_query_engine(
            similarity_top_k=num_results,
            node_postprocessors=[reranker],
            response_mode="compact",
        )
        
        # Execute query
        response = query_engine.query(query_text)
        if not sources:
            return response.response
        # Format the response with source information
        formatted_response = f"{response.response}\n\nSources:\n"
        
        for i, source_node in enumerate(response.source_nodes):
            file_path = source_node.node.metadata.get("file_path", "Unknown")
            
            if file_path == "Unknown":
                formatted_response += f"{i+1}. Unknown\n"
                continue
                
            # Extract just the filename from the path for display
            file_name = os.path.basename(file_path)
            
            # Get the key for metadata lookup by converting Windows path to the exact format in metadata.json
            # Our goal is to transform "C:/Users/.../mark_down_content/b5-cricos/b5-cricos.md"
            # into "mark_down_content\\b5-cricos\\b5-cricos.md"
            
            # First normalize to forward slashes
            normalized_path = file_path.replace("\\", "/")
            
            # Find the position of 'mark_down_content' in the path
            if "mark_down_content" in normalized_path:
                # Extract from 'mark_down_content' to end
                start_idx = normalized_path.find("mark_down_content")
                relative_path = normalized_path[start_idx:]
                
                # Convert to the format used in metadata.json (with backslashes)
                metadata_key = relative_path.replace("/", "\\")
                
                # Look up URL in metadata
                url = self.metadata.get(metadata_key)
                
                if url:
                    formatted_response += f"{i+1}. [{url}]({url})\n"
                else:
                    formatted_response += f"{i+1}. {file_name}\n"
            else:
                formatted_response += f"{i+1}. {file_name}\n"
        
        return formatted_response

if __name__ == "__main__":
    rag = EducationAgentRAG()
    
    print("Education Agent ICEF RAG System Test")
    print("1. Interactive Query Mode")
    print("2. Run Test Evaluation")
    choice = input("Select mode (1/2): ")
    
    if choice == "2":
        run_test_evaluation(rag, test_file="test.json", output_file="tests_result/rag_icef.json")
    else:
        # Interactive query loop
        print("Type 'exit' to quit")
        
        while True:
            query = input("\nEnter your question: ")
            if query.lower() in ["exit", "quit"]:
                break
                
            # Process query and display response
            response = rag.query(query)
            print("\n" + response)