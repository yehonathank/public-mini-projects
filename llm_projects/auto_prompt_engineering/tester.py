"""
Enhanced tester with automated prompt engineering capabilities.
"""

from llm_projects.rag.rag import RAG_Setup
from llm_projects.rag.agent import Agent
from llm_projects.auto_prompt_engineering.prompt_eng import PromptEngineer


def create_prompt_engineering_knowledge_base():
    """Create a basic prompt engineering knowledge base for testing."""
    prompt_eng_docs = """
# Prompt Engineering Best Practices

## Core Principles

1. **Be Specific and Clear**
   - Use precise language and avoid ambiguity
   - Clearly define the task and expected output format
   - Provide context and background information

2. **Use Examples (Few-Shot Learning)**
   - Provide 1-3 examples of desired input-output pairs
   - Examples should cover edge cases and variations
   - Maintain consistent formatting across examples

3. **Structure and Formatting**
   - Use clear headers and bullet points
   - Separate instructions from examples
   - Use consistent formatting for similar elements

4. **Context and Constraints**
   - Provide relevant background information
   - Set clear boundaries and limitations
   - Specify output length and format requirements

## Common Issues and Solutions

### Issue: Inconsistent Response Format
**Solution:** Use explicit formatting instructions and examples
**Example:** "Respond in JSON format with keys: 'answer', 'confidence', 'reasoning'"

### Issue: Hallucination or Inaccurate Information
**Solution:** Emphasize accuracy and fact-checking
**Example:** "Base your answer strictly on the provided context. If information is not available, state 'Information not provided in context.'"

### Issue: Incomplete or Vague Responses
**Solution:** Request specific details and comprehensive coverage
**Example:** "Provide a detailed explanation including: 1) Main concept, 2) Key components, 3) Practical applications"

### Issue: Off-Topic Responses
**Solution:** Clearly define scope and boundaries
**Example:** "Focus only on technical aspects. Do not discuss pricing, marketing, or business strategy."

## Advanced Techniques

1. **Chain of Thought Prompting**
   - Ask the model to show its reasoning step-by-step
   - Use phrases like "Think through this step by step"
   - Break complex problems into smaller components

2. **Role-Based Prompting**
   - Assign specific expertise roles to the model
   - Examples: "You are an expert software engineer", "As a medical professional"
   - Tailor the role to match the domain expertise needed

3. **Constraint-Based Prompting**
   - Set explicit limits and requirements
   - Use "You must" or "You cannot" statements
   - Define acceptable and unacceptable responses

4. **Context Optimization**
   - Prioritize most relevant information first
   - Remove redundant or conflicting information
   - Maintain context relevance to the query

## Question-Answering Specific Guidelines

1. **Answer Accuracy**
   - Emphasize factual correctness
   - Request citation of source material
   - Distinguish between facts and interpretations

2. **Context Usage**
   - Explicitly instruct to use provided context
   - Handle cases where context is insufficient
   - Balance context with general knowledge

3. **Response Completeness**
   - Request comprehensive answers
   - Ask for key points and details
   - Specify minimum content requirements
    """
    
    # Save to temporary file for RAG processing
    import tempfile
    import os
    
    temp_dir = tempfile.mkdtemp()
    prompt_eng_file = os.path.join(temp_dir, "prompt_engineering_guide.txt")
    
    with open(prompt_eng_file, 'w') as f:
        f.write(prompt_eng_docs)
    
    return prompt_eng_file


def run_enhanced_prompt_engineering_test():
    """Run the enhanced prompt engineering test with automated improvement."""
    
    print("Setting up enhanced prompt engineering test...")
    
    # Create prompt engineering knowledge base
    prompt_eng_doc_path = create_prompt_engineering_knowledge_base()
    
    # Setup main RAG pipeline (for document QA)
    main_docs = ["../document_scraper/scraped_docs/bibi.txt"]  # Adjust path as needed
    main_rag = RAG_Setup(main_docs, k_nearest_chunks=3, sentences_per_chunk=5)
    main_rag.run_setup()
    
    # Setup prompt engineering RAG pipeline
    prompt_eng_rag = RAG_Setup([prompt_eng_doc_path], k_nearest_chunks=5, sentences_per_chunk=10)
    prompt_eng_rag.run_setup()
    
    # Initial system prompt (intentionally basic for demonstration)
    initial_system_prompt = """Answer the question based on the context provided.
    
Context: {context}
Question: {user_input}

Provide an answer."""
    
    # Create agent
    agent = Agent("llama3", main_rag, initial_system_prompt)
    
    # Create prompt engineer with both pipelines
    prompt_engineer = PromptEngineer(
        rag_pipeline=main_rag,
        db_path="bibi_qa.json",  # Adjust path as needed
        agent=agent,
        iteration_count=3,
        prompt_eng_rag_pipeline=prompt_eng_rag
    )
    
    print("Starting automated prompt engineering process...")
    print(f"Initial prompt: {initial_system_prompt}")
    print("-" * 80)
    
    # Run multiple iterations with automated prompt improvement
    for iteration in range(3):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration + 1}")
        print(f"{'='*80}")
        
        # Run evaluation on training set
        evaluation_file = prompt_engineer.run_rag_on_training_set()
        
        if iteration < 2:  # Don't wait after last iteration
            print(f"\nIteration {iteration + 1} complete. Prompt has been automatically updated.")
            print(f"Current prompt: {agent.system_prompt}")
            print("\nContinuing to next iteration...")
        else:
            print(f"\nAll iterations complete!")
    
    # Print final results
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    
    evolution_summary = agent.get_prompt_evolution_summary()
    print(f"Total prompt versions: {evolution_summary['total_versions']}")
    print(f"Final prompt:\n{evolution_summary['current_prompt']}")
    
    print("\nPrompt Evolution History:")
    for i, change in enumerate(evolution_summary['history']):
        print(f"Version {change['version']}: {change['reason']}")
    
    # Cleanup
    import shutil
    import os
    shutil.rmtree(os.path.dirname(prompt_eng_doc_path))
    
    return prompt_engineer


if __name__ == "__main__":
    run_enhanced_prompt_engineering_test()
    