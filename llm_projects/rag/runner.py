from llm_projects.rag.agent import Agent
from llm_projects.rag.rag import RAG_Setup
from typing import List, Dict, Optional, Union
import json

class Runner:
    DEFAULT_SYSTEM_PROMPT = """Use the context below to answer the question. If the context does not answer the question reply with "IDK" and nothing else. Do not deviate from these instructions.
        Context:
        {context}

        Question: {user_input}

        Answer:
    """

    def __init__(self, 
                 model_name: str = "llama3",
                 doc_paths: List[str] = None,
                 k_nearest_chunks: int = 2,
                 sentences_per_chunk: int = 20,
                 system_prompt: Optional[str] = None):
        """
        Initialize the Runner with configuration for the RAG system.
        
        Args:
            model_name: Name of the LLM model to use
            doc_paths: List of paths to documents for RAG
            k_nearest_chunks: Number of nearest chunks to retrieve
            sentences_per_chunk: Number of sentences per chunk
            system_prompt: Custom system prompt (optional)
        """
        if doc_paths is None:
            doc_paths = ["llm_projects/document_scraper/scraped_docs/bibi.txt"]
            
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.agent = Agent(
            model_name,
            RAG_Setup(
                doc_paths=doc_paths,
                k_nearest_chunks=k_nearest_chunks,
                sentences_per_chunk=sentences_per_chunk
            ),
            self.system_prompt
        )

    def process_single_query(self, query: str) -> Dict[str, str]:
        """
        Process a single query and return the response with context.
        
        Args:
            query: The user's question
            
        Returns:
            Dictionary containing the response and context
        """
        context = self.agent.pull_context(query)
        response = self.agent.query_ollama(query, context)
        
        return {
            "query": query,
            "response": response,
            "context": context
        }

    def process_structured_data(self, data: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """
        Process a list of structured queries from a file or direct input.
        
        Args:
            data: Either a path to a JSON file or a list of dictionaries with queries
            
        Returns:
            List of dictionaries containing queries and responses
        """
        if isinstance(data, str):
            with open(data, 'r') as f:
                queries = json.load(f)
        else:
            queries = data
            
        results = []
        for query_data in queries:
            if isinstance(query_data, dict) and "query" in query_data:
                result = self.process_single_query(query_data["query"])
                results.append(result)
            else:
                raise ValueError("Each query item must be a dictionary with a 'query' key")
                
        return results

    def interactive_session(self):
        """Run an interactive session where users can input queries directly."""
        print("🧠 Talk to Ollama! Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit"]:
                break
                
            result = self.process_single_query(user_input)
            print(f"Context: {result['context']}")
            print("Ollama:", result["response"])


if __name__ == "__main__":
    # Example of interactive usage
    runner = Runner()
    runner.interactive_session() 