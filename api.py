from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Optional, Dict, Any
import os
import logging
import sys
from dotenv import load_dotenv
from pathlib import Path
from langsmith import Client
from langchain.callbacks import LangChainTracer

# Load environment variables from .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('course_generator.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Course Generator API", description="Generate a course using LangChain with Gemini, OpenAI, or OpenRouter")

# Configure LangSmith for observability
logger.info("Configuring LangSmith for observability")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key:
    langsmith_client = Client()
    tracer = LangChainTracer()
    logger.info("LangSmith tracing configured successfully")
else:
    logger.warning("LANGCHAIN_API_KEY not set. LangSmith tracing will be disabled")

# Initialize LLMs (Gemini, OpenAI, or OpenRouter)
logger.info("Initializing LLM configuration")
gemini_api_key = os.getenv("GEMINI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if gemini_api_key:
    logger.info("Using Gemini as LLM provider")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key)
elif openai_api_key:
    logger.info("Using OpenAI as LLM provider")
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
elif openrouter_api_key:
    logger.info("Using OpenRouter as LLM provider")
    llm = ChatOpenAI(
        model="anthropic/claude-3.5-sonnet",
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1"
    )
else:
    logger.error("No LLM API keys found in environment variables")
    raise ValueError("One of GEMINI_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY must be set in environment variables")

# Define Pydantic model for request
class CourseRequest(BaseModel):
    name: str  # Mandatory
    length: Optional[int] = 8  # Default number of chapters
    language: Optional[str] = "English"  # Default language

# Define prompt templates
chapter_prompt = PromptTemplate(
    input_variables=["name", "num_chapters", "language"],
    template="""
    Generate {num_chapters} chapter titles for a course named "{name}".
    The chapters should form a logical progression, covering the topic comprehensively.
    Output only the list of chapter titles, numbered from 1 to {num_chapters}, one per line.
    Language: {language}
    """
)

content_prompt = PromptTemplate(
    input_variables=["name", "chapter", "previous_summary", "language", "page_start"],
    template="""
    Course: "{name}"
    Current Chapter Title: "{chapter}"
    
    Previous Chapters Summary: {previous_summary}
    
    Generate detailed content for this chapter, ensuring it builds logically on the previous chapters.
    Include key concepts, explanations, examples, and perhaps exercises or key takeaways.
    Make it educational and engaging, with content approximately fitting 2-4 pages (assume ~500 words per page).
    Write in {language}.
    
    Output only the chapter content, starting directly with the text (no numbering or extra headers).
    """
)

summary_prompt = PromptTemplate(
    input_variables=["content"],
    template="""
    Summarize the following chapter content in 2-3 sentences, focusing on key points for continuity:
    {content}
    """
)

async def generate_chapters(name: str, num_chapters: int, language: str) -> list[str]:
    logger.info(f"Generating {num_chapters} chapters for course: {name} in {language}")
    chain = chapter_prompt | llm | StrOutputParser()
    response = await chain.ainvoke({
        "name": name,
        "num_chapters": num_chapters,
        "language": language
    })
    
    if not response:
        logger.error("Failed to generate chapters - empty response from LLM")
        raise ValueError("Failed to generate chapters")
    
    # Parse numbered titles
    lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
    chapters = []
    for line in lines:
        if line[0].isdigit() and ':' in line:
            title = line.split(':', 1)[1].strip()
            chapters.append(title)
        else:
            chapters.append(line)
    
    if len(chapters) != num_chapters:
        chapters = [f"Chapter {i+1}: {line.strip()}" for i, line in enumerate(lines[:num_chapters])]
    
    return chapters

async def generate_chapter_content(name: str, chapter: str, previous_summary: str, language: str, page_start: int) -> tuple[str, int]:
    logger.info(f"Generating content for chapter: {chapter} starting at page {page_start}")
    chain = content_prompt | llm | StrOutputParser()
    content = await chain.ainvoke({
        "name": name,
        "chapter": chapter,
        "previous_summary": previous_summary if previous_summary else "None (this is the first chapter)",
        "language": language,
        "page_start": page_start
    })
    
    if not content:
        logger.error(f"Failed to generate content for chapter: {chapter} - empty response from LLM")
        raise ValueError(f"Failed to generate content for chapter: {chapter}")
    
    # Estimate page count: ~500 words per page
    word_count = len(content.split())
    page_count = max(1, (word_count // 500) + 1)  # At least 1 page
    return content, page_count

async def generate_summary(content: str) -> str:
    logger.info("Generating chapter summary")
    chain = summary_prompt | llm | StrOutputParser()
    summary = await chain.ainvoke({"content": content})
    if not summary:
        logger.warning("Generated empty summary")
    return summary.strip() if summary else ""

@app.post("/generate-course", response_model=Dict[str, Any])
async def generate_course(request: CourseRequest = Body(...)):
    try:
        chapters = await generate_chapters(request.name, request.length, request.language)
        
        course = {
            "course_name": request.name,
            "language": request.language,
            "num_chapters": len(chapters),
            "total_pages": 0,
            "chapters": []
        }
        
        previous_summary = ""
        current_page = 1
        
        for i, chapter_title in enumerate(chapters, 1):
            content, page_count = await generate_chapter_content(
                request.name, chapter_title, previous_summary, request.language, current_page
            )
            
            previous_summary = await generate_summary(content) or f"Key points from Chapter {i}: {chapter_title}"
            
            course["chapters"].append({
                "chapter_number": i,
                "title": chapter_title,
                "content": content,
                "summary": previous_summary,
                "page_start": current_page,
                "page_count": page_count
            })
            
            course["total_pages"] += page_count
            current_page += page_count
        
        return course
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating course: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)