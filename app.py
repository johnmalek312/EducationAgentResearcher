from flask import Flask, render_template, request, jsonify
from content import EducationAgentRAG
from direct_gpt_researcher import EducationAgentRAG as DirectGPTResearcher
from direct_llm import DirectLLMTest
from gpt_researcher_rag import EducationAgentRAG as GPTResearcherRAG
import os
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize all pipeline systems
rag_pipeline = EducationAgentRAG()
direct_gptr_pipeline = DirectGPTResearcher()
direct_llm_pipeline = DirectLLMTest()
gptr_rag_pipeline = GPTResearcherRAG()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def process_query():
    data = request.get_json()
    query_text = data.get('query', '')
    pipeline = data.get('pipeline', 'rag')  # Default to RAG pipeline
    
    logger.info(f"Processing query with pipeline: {pipeline}")
    
    if not query_text:
        return jsonify({'error': 'No query provided'}), 400
    

    try:
        # Select the appropriate pipeline based on the request
        if pipeline == 'rag':
            num_results = int(data.get('num_results', 5))
            include_sources = data.get('include_sources', False)
            
            response = rag_pipeline.query(query_text, num_results, sources=include_sources)
        elif pipeline == 'direct_gptr':
            response = direct_gptr_pipeline.query(query_text)
        elif pipeline == 'direct_llm':
            response = direct_llm_pipeline.query(query_text)
        elif pipeline == 'gptr_rag':
            response = gptr_rag_pipeline.query(query_text)
        else:
            return jsonify({'error': 'Invalid pipeline specified'}), 400
            
        logger.info("Query processed successfully")
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True)