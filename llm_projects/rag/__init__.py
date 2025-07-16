"""
RAG (Retrieval Augmented Generation) package

Provides vector-based document retrieval and LLM integration.

Classes:
- RAG_Setup: Core vector database and semantic search
- Agent: LLM agent with RAG context injection
- Runner: High-level interface for RAG interactions
"""

from .rag import RAG_Setup
from .agent import Agent
from .runner import Runner

__all__ = ["RAG_Setup", "Agent", "Runner"] 