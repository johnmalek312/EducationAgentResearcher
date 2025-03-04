# Australian Education Agent Search System

A comprehensive question-answering system for Australian education agent regulations and requirements using multiple AI approaches. The system provides accurate answers to questions about ESOS, CRICOS, student visas, and other critical regulatory frameworks.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
    - [Query Endpoint](#query-endpoint)
- [Pipeline Descriptions](#pipeline-descriptions)
    - [RAG Pipeline](#rag-pipeline)
    - [Direct LLM](#direct-llm)
    - [GPT Researcher](#gpt-researcher)
    - [GPT Researcher Hybrid](#gpt-researcher-hybrid)
- [Testing Framework](#testing-framework)
- [Test Results](#test-results)

## Overview

This system evaluates four different approaches to answer questions about Australian education agent regulations:

1. **RAG Pipeline**: Retrieval-augmented generation using local knowledge base (97.5% accuracy)
2. **Direct LLM**: Pure LLM responses without retrieval (80% accuracy)
3. **GPT Researcher**: Web search-based research agent (90% accuracy)
4. **GPT Researcher Hybrid**: Combined web search with local document retrieval (90% accuracy)

## Features

- Multiple AI pipelines for answering domain-specific questions
- Web interface for easy querying
- RESTful API for integration with other systems
- Comprehensive test suite with 40 regulatory multiple-choice questions
- Source citation for verified information

## Installation

1. Clone the repository:
     ```bash
     git clone https://github.com/yourusername/AU-Education-Agent-Search.git
     cd AU-Education-Agent-Search
     ```

2. Install required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. Set up environment variables:
     Create a .env file with your API keys:
     ```
     GOOGLE_API_KEY=your_google_api_key
     ```

4. Prepare the document store:
     Ensure your markdown content is in the mark_down_content directory.

## Dependencies

Key dependencies include:
```
python-dotenv==1.0.1
flask==3.1.0
llama-index==0.12.22
llama-index-llms-gemini==0.4.11
llama-index-embeddings-gemini==0.3.1
langchain-google-genai==2.0.10
sentence-transformers==3.4.1
gpt-researcher==0.12.8
pydantic==2.10.6
```

## Usage

Run the Flask server:
```bash
python app.py
```

Visit `http://localhost:5000` in your browser to access the web interface.

## API Endpoints

### Query Endpoint

```
POST /query
```

Request body:
```json
{
    "query": "What is CRICOS?",
    "pipeline": "rag",
    "num_results": 5,
    "include_sources": true
}
```

Parameters:
- `pipeline`: One of `rag`, `direct_llm`, `direct_gptr`, or `gptr_rag` (default: `rag`)
- `num_results`: Number of documents to retrieve (only used for rag pipeline, default: 5)
- `include_sources`: Whether to include source information (only used for rag pipeline, default: false)

## Pipeline Descriptions

### RAG Pipeline
Uses Gemini embeddings and models with a local knowledge base to retrieve and generate accurate answers. This approach achieved the highest accuracy (97.5%) in testing.

### Direct LLM
Directly queries the Gemini model without document retrieval. While simpler to implement, it achieved 80% accuracy on regulatory questions.

### GPT Researcher
Uses web search capabilities to find relevant information online and generates comprehensive answers. Achieved 90% accuracy.

### GPT Researcher Hybrid
Combines local document retrieval with web search capabilities. Also achieved 90% accuracy.

## Testing Framework

The system includes a comprehensive test framework that evaluates performance against 40 multiple-choice questions covering:
- ESOS Act and National Code requirements
- Student visa conditions and compliance
- PRISMS reporting obligations
- Tuition Protection Service
- Privacy regulations
- Accommodation requirements for underage students
- International student work rights

Run tests using:
```bash
python content.py  # Choose option 2 for testing
```

## Test Results

| Pipeline | Accuracy | Notes |
|----------|----------|-------|
| RAG | 97.5% | Best accuracy, requires high-quality knowledge base |
| Direct LLM | 80.0% | Faster but less accurate |
| GPT Researcher | 90.0% | Good for accessing up-to-date information |
| GPT Researcher Hybrid | 90.0% | Balanced approach |
