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
import researcher
import asyncio
from test_evalutate import run_test_evaluation
# Load environment variables from .env file
load_dotenv()
trace = False
if trace:
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HTTPSpanExporter,
    )
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    # Add Phoenix exporter
    span_phoenix_processor = SimpleSpanProcessor(
        HTTPSpanExporter(endpoint="https://app.phoenix.arize.com/v1/traces")
    )

    # Add span processor to tracer
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(span_phoenix_processor)

    # Instrument LlamaIndex
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)



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
        Settings.llm = llm_provider(model="models/gemini-2.0-flash", temperature=0)
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
            
    def query(self, query_text: str) -> str:
        response = asyncio.run(researcher.conduct_research(query_text, deep_research=False, report_source="hybrid"))
        return response

if __name__ == "__main__":
    # Create RAG system
    rag = EducationAgentRAG()
    
    print("Australian Education Agent GPTR RAG System")
    print("1. Interactive Query Mode")
    print("2. Run Test Evaluation")
    choice = input("Select mode (1/2): ")
    
    if choice == "2":
        run_test_evaluation(rag, test_file="test.json", output_file="tests_result/gptr_rag.json")
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