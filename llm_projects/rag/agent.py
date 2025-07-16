from llm_projects.rag.rag import RAG_Setup
import ollama
from datetime import datetime

class Agent:
    def __init__(self, ollama_model: str, rag: RAG_Setup, system_prompt: str):
        self.ollama_model = ollama_model
        self.rag = rag
        self.system_prompt = system_prompt
        self.prompt_history = []  # Track prompt changes over time
        self.performance_metrics = []  # Track performance for each prompt version

    def query_ollama(self, user_input: str, context: str) -> str:
        system_prompt = self.system_prompt.format(context=context, user_input=user_input)

        response = ollama.generate(
            model = self.ollama_model, 
            prompt = system_prompt
        )

        return response["response"]
    
    def pull_context(self, user_input: str):
        return self.rag.retrieve_k_context(user_input)

    def update_system_prompt(self, new_prompt: str, reason: str = ""):
        """Update the system prompt and track the change."""
        old_prompt = self.system_prompt
        self.system_prompt = new_prompt
        
        # Track the change
        change_record = {
            "timestamp": datetime.now().isoformat(),
            "old_prompt": old_prompt,
            "new_prompt": new_prompt,
            "reason": reason,
            "version": len(self.prompt_history) + 1
        }
        self.prompt_history.append(change_record)
        
        print(f"System prompt updated (version {change_record['version']})")
        if reason:
            print(f"Reason: {reason}")

    def add_performance_metric(self, metric_data: dict):
        """Add performance metrics for the current prompt version."""
        metric_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt_version": len(self.prompt_history),
            "system_prompt": self.system_prompt,
            **metric_data
        }
        self.performance_metrics.append(metric_entry)

    def get_prompt_evolution_summary(self):
        """Get a summary of how the prompt has evolved."""
        return {
            "total_versions": len(self.prompt_history) + 1,  # +1 for initial prompt
            "current_prompt": self.system_prompt,
            "history": self.prompt_history,
            "performance_metrics": self.performance_metrics
        }
