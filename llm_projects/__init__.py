"""
LLM Projects package

A collection of LLM-powered tools including RAG systems, 
auto prompt engineering, and document processing.

Main exports:
- RAG_Setup: Vector database and retrieval system
- Agent: LLM agent with RAG capabilities  
- Runner: High-level RAG runner interface
- PromptEngineer: Automated prompt optimization
- PDF_Scraper: PDF text extraction utility
"""

# RAG System exports
from llm_projects.rag.rag import RAG_Setup
from llm_projects.rag.agent import Agent
from llm_projects.rag.runner import Runner

# Auto Prompt Engineering exports
from llm_projects.auto_prompt_engineering.prompt_eng import PromptEngineer

# Document Processing exports  
from llm_projects.document_scraper.pdf_scraper import PDF_Scraper

__version__ = "0.1.0"

__all__ = [
    "RAG_Setup",
    "Agent", 
    "Runner",
    "PromptEngineer",
    "PDF_Scraper"
] 