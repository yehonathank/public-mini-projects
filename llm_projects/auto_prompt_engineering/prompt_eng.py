"""
PLAN:
1. Setup a rag pipeline (done in rag.py)
2. Create a DB in JSON format of ground truth questions and expected answers (done)
3. Randomize order of db loading and split the DB into training and test sets (done)
4. Run a tester.py to test the rag pipeline with the test set (done)
5. Evaluate the results with an LLM evaluator one by one, return an explanation of the result and how it can be improved
6. Send system prompt, question, context, answer to a prompt enigneering agent
7. Load the prompt engineering agent with a rag pipeline for internet's best advice for better prompt engineering
8. Prompt engineer will send a new system prompt to the tester.py pipline
9. Repeat 3 times
10. Update the runner.py with a new system prompt for freeform user testing
11. Future work: automate the improvement of context chunking methods
"""

import json
import random
import os
from pathlib import Path

from llm_projects.rag.rag import RAG_Setup
from llm_projects.rag.agent import Agent 



class PromptEngineer:
    def __init__(self, rag_pipeline: RAG_Setup, db_path: str, agent: Agent, iteration_count: int, 
                 prompt_eng_rag_pipeline: RAG_Setup = None):
        self.rag_pipeline = rag_pipeline
        self.db_path = db_path
        self.agent = agent
        self.iteration_count = iteration_count
        
        # Prompt engineering specific components
        self.prompt_eng_rag_pipeline = prompt_eng_rag_pipeline
        self.prompt_engineer_agent = None
        
        # Initialize prompt engineering agent if RAG pipeline provided
        if self.prompt_eng_rag_pipeline:
            self._initialize_prompt_engineer_agent()

        self.load_db()
        self.split_db()

    def _initialize_prompt_engineer_agent(self):
        """Initialize the prompt engineering agent with specialized system prompt."""
        prompt_eng_system_prompt = """You are an expert prompt engineer. Your task is to analyze evaluation feedback and improve system prompts for better performance.

Given evaluation data, you should:
1. Identify key weaknesses in the current system prompt
2. Apply prompt engineering best practices from your knowledge base
3. Generate an improved system prompt that addresses the identified issues
4. Focus on clarity, specificity, and effectiveness

Use the context: {context} to inform your recommendations.
User input: {user_input}

Provide your analysis and the improved prompt in a clear, structured format."""

        self.prompt_engineer_agent = Agent("llama3", self.prompt_eng_rag_pipeline, prompt_eng_system_prompt)

    def load_db(self):
        with open(self.db_path, 'r') as f:
            self.db = json.load(f)

    def split_db(self):
        random.shuffle(self.db)
        split_idx = int(len(self.db) * 0.8)
        self.train_db = self.db[:split_idx]
        self.test_db = self.db[split_idx:]
    
    def run_rag_on_training_set(self):
        """Run the RAG pipeline on the testing set and print results."""
        print(f"Running RAG on {len(self.train_db)} test questions...")
        
        # Collect all evaluations for this iteration
        all_evaluations = []
        
        for i, test_item in enumerate(self.train_db):
            if "question" in test_item:
                question = test_item["question"]
                expected_answer = test_item.get("answer", "No expected answer provided")
                
                # Get context from RAG pipeline
                context = self.rag_pipeline.retrieve_k_context(question)
                
                # Get response from agent
                response = self.agent.query_ollama(question, context)
                
                print(f"\n--- Test {i+1} ---")
                print(f"Question: {question}")
                print(f"Expected: {expected_answer}")
                print(f"Got: {response}")
                print(f"Context: {context}")
                print("-" * 50)
                
                # Collect evaluation instead of saving immediately
                evaluation = self.evaluate_single_result(question, expected_answer, response, context)
                all_evaluations.append({
                    "question": question,
                    "expected_answer": expected_answer,
                    "actual_response": response,
                    "context": context,
                    "evaluation": evaluation
                })
        
        # Save all evaluations for this iteration
        evaluation_file = self.save_iteration_results(all_evaluations)
        
        # Run prompt engineering if we have the prompt engineering agent
        if self.prompt_engineer_agent and evaluation_file:
            print(f"\n{'='*60}")
            print("PROMPT ENGINEERING PHASE")
            print(f"{'='*60}")
            self.prompt_engineer_agent_method(evaluation_file)
        
        return evaluation_file
    
    def evaluate_single_result(self, question, expected_answer, response, context):
        """Evaluate a single QA pair and return the evaluation text."""
        # Create evaluator agent with specialized system prompt
        evaluator_prompt = f"""You are an expert prompt evaluator. Analyze the following QA interaction and provide detailed feedback:

Question: {question}
Expected Answer: {expected_answer} 
Actual Response: {response}
Context Provided: {context}
Agent's System Prompt: {self.agent.system_prompt}

Evaluate:
1. Answer Accuracy - How well does the response match the expected answer?
2. Context Usage - Did the agent effectively use the provided context?
3. System Prompt Effectiveness - Is the current system prompt guiding the agent appropriately?
4. Suggested Improvements - What changes to the system prompt could improve performance?

Provide your evaluation in a clear, structured format."""

        evaluator = Agent("llama3", self.rag_pipeline, evaluator_prompt)
        evaluation = evaluator.query_ollama("Please provide your evaluation.", "")
        print("\nEvaluation:")
        print(evaluation)
        return evaluation

    def save_iteration_results(self, all_evaluations):
        """Save all evaluations from current iteration to a file."""
        # Create evaluation directory if it doesn't exist
        db_name = Path(self.db_path).stem  # Gets filename without extension
        eval_dir = Path(f"{db_name}_evaluation")
        eval_dir.mkdir(exist_ok=True)

        # Get the next iteration number (handle gaps properly)
        existing_files = list(eval_dir.glob(f"{db_name}_evaluation*.json"))
        if existing_files:
            # Extract numbers from existing files and find the next available number
            existing_numbers = []
            for file in existing_files:
                try:
                    # Extract number from filename like "bibi_qa_evaluation3.json"
                    num_str = file.stem.replace(f"{db_name}_evaluation", "")
                    if num_str.isdigit():
                        existing_numbers.append(int(num_str))
                except:
                    pass
            current_iteration = max(existing_numbers) + 1 if existing_numbers else 0
        else:
            current_iteration = 0

        if current_iteration < self.iteration_count:
            iteration_data = {
                "iteration": current_iteration,
                "system_prompt": self.agent.system_prompt,
                "evaluations": all_evaluations,
                "total_questions": len(all_evaluations)
            }

            output_file = eval_dir / f"{db_name}_evaluation{current_iteration}.json"
            with open(output_file, "w") as f:
                json.dump(iteration_data, f, indent=4)
            
            print(f"\nSaved iteration {current_iteration} results to {output_file}")
            return output_file
        else:
            print(f"Evaluation for {db_name} has reached the maximum number of iterations ({self.iteration_count}).")
            return None

    def prompt_engineer_agent_method(self, evaluation_file_path: str):
        """
        Analyze evaluation data and generate improved system prompt using prompt engineering best practices.
        
        Args:
            evaluation_file_path: Path to the evaluation JSON file
        """
        if not self.prompt_engineer_agent:
            print("Warning: Prompt engineering agent not initialized. Skipping prompt improvement.")
            return
            
        print("Loading evaluation data for prompt engineering analysis...")
        
        # Load evaluation data
        with open(evaluation_file_path, 'r') as f:
            evaluation_data = json.load(f)
        
        # Extract key feedback points and metrics
        analysis_summary = self._extract_evaluation_insights(evaluation_data)
        
        # Query RAG for relevant prompt engineering techniques
        prompt_eng_query = f"""Based on these evaluation insights: {analysis_summary}
        
What are the best prompt engineering techniques to address these specific issues? 
Focus on system prompt improvements, formatting, instruction clarity, and response guidance."""
        
        # Get context from prompt engineering RAG
        prompt_eng_context = self.prompt_eng_rag_pipeline.retrieve_k_context(prompt_eng_query)
        
        # Generate specific prompt improvements
        improvement_query = f"""Current system prompt: {self.agent.system_prompt}

Evaluation insights: {analysis_summary}

Based on the prompt engineering best practices in the context, provide:
1. Specific weaknesses in the current system prompt
2. A completely rewritten and improved system prompt
3. Explanation of the changes made and why they should improve performance

Focus on making the prompt more specific, clear, and effective for the identified issues."""
        
        improved_prompt_response = self.prompt_engineer_agent.query_ollama(improvement_query, str(prompt_eng_context))
        
        print(f"\n{'='*60}")
        print("PROMPT ENGINEERING ANALYSIS")
        print(f"{'='*60}")
        print(improved_prompt_response)
        
        # Extract the new system prompt from the response
        new_system_prompt = self._extract_new_prompt_from_response(improved_prompt_response)
        
        if new_system_prompt:
            # Update the agent's system prompt
            reason = f"Automated improvement based on evaluation iteration {evaluation_data['iteration']}"
            self.agent.update_system_prompt(new_system_prompt, reason)
            
            # Add performance metrics
            self.agent.add_performance_metric({
                "evaluation_file": str(evaluation_file_path),
                "issues_identified": analysis_summary,
                "improvement_response": improved_prompt_response
            })
            
            # Save prompt evolution to file
            self._save_prompt_evolution(evaluation_data['iteration'])
        else:
            print("Warning: Could not extract new system prompt from response.")

    def _extract_evaluation_insights(self, evaluation_data: dict) -> str:
        """Extract key insights and patterns from evaluation data."""
        evaluations = evaluation_data.get('evaluations', [])
        
        # Collect all evaluation feedback
        all_feedback = []
        for eval_item in evaluations:
            all_feedback.append(eval_item.get('evaluation', ''))
        
        # Combine feedback for pattern analysis
        combined_feedback = "\n\n".join(all_feedback)
        
        # Create summary of common issues
        insight_query = f"""Analyze this evaluation feedback and identify the top 3 most common issues and improvement areas:

{combined_feedback}

Provide a concise summary of the main problems that need to be addressed in the system prompt."""
        
        # Use the main RAG pipeline to analyze patterns
        temp_analyzer = Agent("llama3", self.rag_pipeline, 
                            "You are an expert at analyzing evaluation feedback. Provide concise, actionable insights.")
        insights = temp_analyzer.query_ollama(insight_query, "")
        
        return insights

    def _extract_new_prompt_from_response(self, response: str) -> str:
        """Extract the new system prompt from the prompt engineer's response."""
        # Look for common patterns where the new prompt might be presented
        lines = response.split('\n')
        
        # Try to find sections that contain the new prompt
        new_prompt_indicators = [
            "improved system prompt:",
            "new system prompt:",
            "revised prompt:",
            "updated prompt:",
            "recommended prompt:"
        ]
        
        collecting_prompt = False
        prompt_lines = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if this line indicates the start of a new prompt
            if any(indicator in line_lower for indicator in new_prompt_indicators):
                collecting_prompt = True
                # If the prompt starts on the same line, extract it
                for indicator in new_prompt_indicators:
                    if indicator in line_lower:
                        prompt_start = line_lower.find(indicator) + len(indicator)
                        if prompt_start < len(line):
                            potential_prompt = line[prompt_start:].strip()
                            if potential_prompt:
                                prompt_lines.append(potential_prompt)
                        break
                continue
            
            # If we're collecting and hit explanatory text, stop
            if collecting_prompt:
                if line_lower.startswith(('explanation:', 'changes made:', 'rationale:', 'why this works:')):
                    break
                if line.strip():  # Non-empty line
                    prompt_lines.append(line)
        
        if prompt_lines:
            new_prompt = '\n'.join(prompt_lines).strip()
            # Clean up any quotation marks or formatting
            new_prompt = new_prompt.strip('"\'`')
            return new_prompt
        
        # If no clear prompt found, try to extract from the middle section
        print("Warning: Could not automatically extract new prompt. Manual review needed.")
        return None

    def _save_prompt_evolution(self, iteration: int):
        """Save the prompt evolution history to a file."""
        from datetime import datetime
        
        db_name = Path(self.db_path).stem
        evolution_file = Path(f"{db_name}_prompt_evolution.json")
        
        evolution_data = {
            "current_iteration": iteration,
            "agent_evolution": self.agent.get_prompt_evolution_summary(),
            "timestamp": datetime.now().isoformat(),
            "working_directory": str(Path().cwd())
        }
        
        with open(evolution_file, 'w') as f:
            json.dump(evolution_data, f, indent=4)
        
        print(f"Prompt evolution saved to {evolution_file}")