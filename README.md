# MCQ and Q&A Generator

A tool that generates high-quality Multiple Choice Questions (MCQs) and Question-Answer (Q&A) pairs from PDF documents and predefined subject categories.

## Overview

This tool analyzes PDF content and generates educational content in two forms:

* **MCQs from PDFs** : Extracts knowledge from uploaded PDFs to create challenging multiple-choice questions
* **Predefined Q&A Pairs** : Creates additional Q&A pairs based on predefined categories and concepts related to pdf content.

## Features

- Extracts keywords from PDFs and predefined categories.
- Generates validated MCQs/QnA with difficulty levels (Easy, Moderate, Hard).
- Saves outputs in CSV and JSON formats.
- Avoids questions about references, citations, and paper metadata

## Requirements

* Python 3.8+
* Azure OpenAI account with API access
* Required Python packages (see `requirements.txt`)

## Installation

1. **Clone this repository:**

   `git clone https://github.com/yourusername/mcq-generator.git`
   `cd mcq-generator`
2. **Install dependencies:**

   `pip install -r requirements.txt`
3. **Create a `.env` file with your Azure OpenAI credentials:**

   `AZURE_OPENAI_ENDPOINT=your_azure_endpoint`
   `AZURE_OPENAI_API_KEY=your_api_key`
   `AZURE_OPENAI_API_VERSION=your_api_version`

## Custom Categories

You can define your own categories and keywords by modifying the `PREDEFINED_CATEGORIES` dictionary:

`PREDEFINED_CATEGORIES = {     "Your Category": [         "Keyword1", "Keyword2", "Keyword3"     ],     "Another Category": [         "ConceptA", "ConceptB", "ConceptC"     ] }`

## Customizing Output

The tool generates structured outputs that adhere to the following schemas:

### MCQ Structure

json

`{     "question": "Clear question text",     "options": {         "A": "First option",         "B": "Second option",         "C": "Third option",         "D": "Fourth option"     },     "correct_answer": "A",     "explanation": "Why this answer is correct",     "source": "PDF",     "difficulty": "Hard" }`

### Q&A Structure

json

`{     "question": "Question text",     "answer": "Detailed answer",     "source": "Predefined",     "difficulty": "Moderate" }`

## Troubleshooting

* **API Rate Limits** : The tool uses exponential backoff for API limitations
* **PDF Parsing Issues** : If PDF content isn't properly extracted, try different PDFs or check file permissions
* **Generation Quality** : Adjust temperature settings (lower for more factual, higher for more variety)

## Acknowledgements

* Uses Azure OpenAI for natural language processing
* LangChain for document processing
* Pydantic for data validation
