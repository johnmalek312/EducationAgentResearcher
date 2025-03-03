import asyncio
from dotenv import load_dotenv
import researcher
from test_evalutate import run_test_evaluation
# Load environment variables from .env file
load_dotenv()
class EducationAgentRAG:
    def query(self, query_text: str) -> str:

        # I want to change this from rag to using gptresearcher

        """
        Returns the query response from gptresearcher
        """
        research_results = asyncio.run(researcher.conduct_research("This questions refers to Australia:\n\n"+query_text))
        return research_results


if __name__ == "__main__":
    rag = EducationAgentRAG() # this is not rag pipeline
    
    print("Australian Education Agent GPTR Direct System")
    print("1. Interactive Query Mode")
    print("2. Run Test Evaluation")
    choice = input("Select mode (1/2): ")
    
    if choice == "2":
        run_test_evaluation(rag, test_file="test.json", output_file="tests_result/gptr_direct.json")
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