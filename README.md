# Course Generator API

A FastAPI-based service that generates structured course content using various Language Models (LLMs) through LangChain. The service can work with multiple LLM providers including Gemini, OpenAI, and OpenRouter.

## Features

- Generate complete course structures with multiple chapters
- Dynamic content generation based on previous chapters for coherent progression
- Supports multiple languages
- Automatic page count estimation
- Chapter summaries for content continuity
- Configurable course length
- Logging and LangSmith observability integration

## Prerequisites

- Python 3.8+
- FastAPI
- LangChain
- One of the following API keys:
  - Google Gemini
  - OpenAI
  - OpenRouter

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd course_generator
```

2. Install required packages:
```bash
pip install fastapi uvicorn langchain python-dotenv langchain-google-genai langchain-openai pydantic langsmith
```

3. Set up environment variables:
   - Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   - Edit `.env` and add your API keys

## Environment Variables

Configure the following environment variables in your `.env` file:

```env
# LangSmith Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=course-generator

# LLM API Keys (use at least one)
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

## Usage

1. Start the server:
```bash
python api.py
```

2. The API will be available at `http://localhost:8000`

3. Generate a course using the `/generate-course` endpoint:
```bash
curl -X POST "http://localhost:8000/generate-course" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "Introduction to Python",
       "length": 8,
       "language": "English"
     }'
```

## API Endpoints

### POST /generate-course

Generate a complete course structure and content.

Request body:
```json
{
    "name": "string",         // Required: Course name
    "length": 8,             // Optional: Number of chapters (default: 8)
    "language": "English"    // Optional: Content language (default: "English")
}
```

Response:
```json
{
    "course_name": "string",
    "language": "string",
    "num_chapters": 0,
    "total_pages": 0,
    "chapters": [
        {
            "chapter_number": 0,
            "title": "string",
            "content": "string",
            "summary": "string",
            "page_start": 0,
            "page_count": 0
        }
    ]
}
```

## Logging

The application includes comprehensive logging:
- Console output for immediate feedback
- File-based logging in `course_generator.log`
- LangSmith integration for LLM observability (if configured)

## Development

The project uses:
- FastAPI for the web framework
- Pydantic for data validation
- LangChain for LLM integration
- python-dotenv for environment management

## Error Handling

The API includes robust error handling for:
- Missing API keys
- LLM generation failures
- Invalid requests
- Content generation issues
