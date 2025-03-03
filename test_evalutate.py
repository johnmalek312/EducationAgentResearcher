import json
import time
import re
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from llama_index.llms.gemini import Gemini
from dotenv import load_dotenv
load_dotenv()

# Define structured output schema for answer extraction
class AnswerChoice(BaseModel):
    """Represents the selected answer choice(s) for a multiple choice question."""
    selected_indices: List[int] = Field(
        description="List of selected answer indices (1-4). If the answer is ambiguous, select the most probable option."
    )

def load_test_questions(test_file_path: str = "test.json") -> List[Dict[str, Any]]:
    """Load test questions from a JSON file"""
    try:
        with open(test_file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading test questions: {e}")
        return []
        
def evaluate_answer(question_data: Dict[str, Any], rag_response: str) -> Tuple[bool, List[int], List[int]]:
    """
    Evaluate if the RAG system's answer indices match the correct answer indices
    
    Args:
        question_data: The question data including correct answer indices
        rag_response: The response from the RAG system
        
    Returns:
        Tuple[bool, List[int], List[int]]: Whether answer is correct, correct indices, extracted indices
    """
    question = question_data["question"]
    answers = question_data["answers"]
    correct_indices = question_data["correct_answer"]
    
    # Extract answer indices from RAG response using structured output
    extracted_indices = extract_answer_indices(rag_response, question, answers)
    
    # Check if extracted indices match correct indices
    is_correct = sorted(extracted_indices) == sorted(correct_indices)
    
    return is_correct, correct_indices, extracted_indices

def extract_answer_indices(response: str, question: str, choices: List[str]) -> List[int]:
    """
    Extract the chosen answer indices (1-4) from the RAG response using structured output
    
    Args:
        response: RAG system response
        question: Original question
        choices: List of possible answers
        
    Returns:
        List[int]: The extracted answer indices (1-based)
    """
    # Format choices with indices for the prompt
    choices_text = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
    
    extraction_prompt = f"""
    Based on the following multiple-choice question, answer choices, and response, determine which answer option number(s) 
    (1, 2, 3, or 4) is being indicated as correct in the response.

    Question: {question}
    
    Answer choices:
    {choices_text}
    
    Response from system: {response}
    
    If the response is ambiguous, select the most probable answer option(s).
    """
    
    try:
        # Use structured LLM to extract the answer indices
        llm = Gemini(model_name="models/gemini-2.0-flash-lite")
        structured_llm = llm.as_structured_llm(AnswerChoice)
        
        # Get structured output
        result = structured_llm.complete(extraction_prompt).raw
        
        # Extract indices from structured output
        indices = result.selected_indices
        
        # Validate indices are in range
        valid_indices = [idx for idx in indices if 1 <= idx <= len(choices)]
        
        return valid_indices if valid_indices else []
        
    except Exception as e:
        print(f"Error extracting answer indices: {e}")
        
        # Fallback 1: Try direct extraction of numbers from response
        try:
            indices = []
            numbers = re.findall(r'\b[1-4]\b', response)
            for num in numbers:
                idx = int(num)
                if 1 <= idx <= len(choices):
                    indices.append(idx)
            
            if indices:
                return sorted(list(set(indices)))
        except:
            pass
        
        # Fallback 2: Try text matching
        indices = []
        for i, choice in enumerate(choices, 1):
            if choice.lower() in response.lower():
                indices.append(i)
        
        return sorted(list(set(indices))) if indices else []
            
def run_test(rag, test_file_path: str = "test.json", verbose: bool = True, rate_limit: int = -1) -> Dict[str, Any]:
    """
    Run tests on the RAG system using questions from a test file
    Rate limited to specified number of questions per minute
    
    Args:
        rag: The RAG system to test
        test_file_path: Path to the test questions file
        verbose: Whether to print detailed results for each question
        rate_limit: Number of questions per minute (default: 5, -1 to disable rate limiting)
        
    Returns:
        Dict: Statistics about the test results
    """
    questions = load_test_questions(test_file_path)
    if not questions:
        return {"error": "No test questions loaded"}
        
    results = {
        "total_questions": len(questions),
        "correct_answers": 0,
        "incorrect_answers": 0,
        "question_results": []
    }
    
    print(f"Starting evaluation on {len(questions)} questions...")
    if rate_limit == -1:
        target_interval = 0
    else:
        print(f"Rate limited to {rate_limit} questions per minute.")
        # Calculate target time between question starts (seconds)
        target_interval = 60 / rate_limit
    
    start_time = time.time()
    last_question_start_time = start_time
    
    for i, question in enumerate(questions):
        question_text = question['question'] + "\n\nAnswer options:\n" + "\n".join(question['answers'])
        print(f"\nProcessing question {i+1}/{len(questions)}: {question_text[:60]}...")
        
        # Add rate limiting (but not before the first question and not if disabled)
        if i > 0 and rate_limit != -1:
            current_time = time.time()
            time_since_last_question = current_time - last_question_start_time
            
            # If we're going too fast, wait until the target interval is reached
            if time_since_last_question < target_interval:
                wait_seconds = target_interval - time_since_last_question
                print(f"Rate limiting: waiting {wait_seconds:.1f} seconds...")
                time.sleep(wait_seconds)
        
        # Record the start time of processing this question
        last_question_start_time = time.time()
        
        # Query RAG system
        try:
            rag_response = rag.query(question_text)
            
            # Evaluate answer
            is_correct, correct_indices, extracted_indices = evaluate_answer(question, rag_response)
            
            # Update results
            if is_correct:
                results["correct_answers"] += 1
            else:
                results["incorrect_answers"] += 1
            
            # Get answer texts for display purposes
            answers = question.get("answers", [])
            correct_answer_texts = [answers[idx-1] for idx in correct_indices if 1 <= idx <= len(answers)]
            extracted_answer_texts = [answers[idx-1] for idx in extracted_indices if 1 <= idx <= len(answers)]
            
            question_result = {
                "question": question_text,
                "correct_indices": correct_indices,
                "correct_answers": correct_answer_texts,
                "extracted_indices": extracted_indices,
                "extracted_answers": extracted_answer_texts,
                "is_correct": is_correct,
                "full_response": rag_response
            }
            
            results["question_results"].append(question_result)
            
            if verbose:
                status = "✓" if is_correct else "✗"
                print(f"{status} | Question: {question_text[:60]}...")
                print(f"Expected indices: {correct_indices} → {', '.join(correct_answer_texts)}")
                print(f"Extracted indices: {extracted_indices} → {', '.join(extracted_answer_texts)}")
                
            # Show progress and estimated time remaining
            questions_left = len(questions) - (i + 1)
            est_minutes_remaining = 0 if rate_limit == -1 else (questions_left * target_interval) / 60
            print(f"Progress: {i+1}/{len(questions)} ({(i+1)/len(questions):.1%})")
            if rate_limit != -1:
                print(f"Est. time remaining: {est_minutes_remaining:.1f} minutes")
            
        except Exception as e:
            print(f"Error processing question {i+1}: {e}")
            
    # Calculate accuracy
    results["accuracy"] = results["correct_answers"] / results["total_questions"] if results["total_questions"] > 0 else 0
    results["test_time_seconds"] = time.time() - start_time
    
    # Print summary
    print(f"\n=== Test Results Summary ===")
    print(f"Total questions: {results['total_questions']}")
    print(f"Correct answers: {results['correct_answers']}")
    print(f"Incorrect answers: {results['incorrect_answers']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Time taken: {results['test_time_seconds']:.2f} seconds")
    
    return results

def save_test_results(results: Dict[str, Any], output_file: str = "test_results.json"):
    """Save test results to a JSON file"""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Test results saved to {output_file}")
    except Exception as e:
        print(f"Error saving test results: {e}")

def run_test_evaluation(rag, test_file: str = "test.json",output_file: str = ""):
    # Run test evaluation
    test_file = input(f"Enter test file path (default: {test_file}): ") or test_file
    verbose = input(f"Show detailed results? (y/n) (default: y): ").lower() != 'n'
    results = run_test(rag, test_file, verbose)
    
    # Save results
    save_results = input(f"Save results to file? (y/n) (default: y): ").lower() != 'n'
    if save_results:
        output_file = input(f"Enter output file name (default: {output_file}): ") or output_file
        save_test_results(results, output_file)
