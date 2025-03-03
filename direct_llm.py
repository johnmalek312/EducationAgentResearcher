from llama_index.llms.gemini import Gemini as llm_provider
from llama_index.core import Settings
from dotenv import load_dotenv
from test_evalutate import run_test_evaluation

# Load environment variables from .env file
load_dotenv()


class DirectLLMTest:
    def __init__(self):
        """Initialize the direct LLM tester (without RAG)"""
        self.setup_llm()
        
    def setup_llm(self):
        """Configure LLM settings"""
        Settings.llm = llm_provider(model="models/gemini-2.0-flash-thinking-exp", temperature=0)
        
    def query(self, query_text: str) -> str:
        """
        Query the LLM directly without RAG
        
        Args:
            query_text: The question to ask
            
        Returns:
            str: The direct response from the LLM
        """
        llm = Settings.llm
        prompt = f"""
        Answer the following question about Australian education agent regulations and requirements:
        
        {query_text}
        
        Please provide a detailed and accurate response based on your knowledge.
        """
        response = llm.complete(prompt)
        return response.text


if __name__ == "__main__":
    rag = DirectLLMTest()
    
    print("Direct LLM Tester (No RAG)")
    print("1. Interactive Query Mode")
    print("2. Run Test Evaluation")
    choice = input("Select mode (1/2): ")
    
    if choice == "2":
        run_test_evaluation(rag, test_file="test.json", output_file="tests_result/direct_llm.json")
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