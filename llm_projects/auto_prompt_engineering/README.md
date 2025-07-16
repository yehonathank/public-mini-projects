# Automated Prompt Engineering System

This system implements an automated prompt engineering pipeline that evaluates, analyzes, and improves system prompts using RAG-based feedback and prompt engineering best practices.

## Overview

The system works by:
1. **Evaluation**: Running your agent on a test dataset and collecting performance feedback
2. **Analysis**: Using LLM evaluators to analyze response quality and identify issues
3. **Improvement**: Applying prompt engineering best practices to generate improved prompts
4. **Iteration**: Repeating the process to continuously refine the system prompt

## Key Components

### 1. PromptEngineer Class
Main orchestrator that manages the entire prompt engineering workflow.

**Parameters:**
- `rag_pipeline`: Main RAG pipeline for document QA
- `db_path`: Path to your ground truth Q&A JSON file
- `agent`: The agent whose prompt you want to improve
- `iteration_count`: Number of improvement iterations to run
- `prompt_eng_rag_pipeline`: RAG pipeline loaded with prompt engineering knowledge

### 2. Enhanced Agent Class
Extended with prompt tracking and updating capabilities:
- `update_system_prompt()`: Dynamically update the system prompt
- `add_performance_metric()`: Track performance across iterations
- `get_prompt_evolution_summary()`: Get history of prompt changes

## Usage Example

```python
from llm_projects.rag.rag import RAG_Setup
from llm_projects.rag.agent import Agent
from llm_projects.auto_prompt_engineering.prompt_eng import PromptEngineer

# Setup main RAG pipeline
main_rag = RAG_Setup(["path/to/documents.txt"], k_nearest_chunks=3, sentences_per_chunk=5)
main_rag.run_setup()

# Setup prompt engineering knowledge base
prompt_eng_rag = RAG_Setup(["path/to/prompt_engineering_guide.txt"], k_nearest_chunks=5, sentences_per_chunk=10)
prompt_eng_rag.run_setup()

# Create agent with initial prompt
initial_prompt = "Answer questions based on the provided context..."
agent = Agent("llama3", main_rag, initial_prompt)

# Create prompt engineer
prompt_engineer = PromptEngineer(
    rag_pipeline=main_rag,
    db_path="qa_dataset.json",
    agent=agent,
    iteration_count=3,
    prompt_eng_rag_pipeline=prompt_eng_rag
)

# Run automated prompt engineering
prompt_engineer.run_rag_on_training_set()
```

## Output Files

The system generates several output files:

### Evaluation Files
- `{dataset}_evaluation{iteration}.json`: Detailed evaluation results for each iteration
- Contains questions, expected answers, actual responses, and LLM evaluations

### Prompt Evolution Files
- `{dataset}_prompt_evolution.json`: Complete history of prompt changes
- Tracks what changes were made and why

## File Structure

```
{dataset}_evaluation/
├── {dataset}_evaluation0.json    # First iteration results
├── {dataset}_evaluation1.json    # Second iteration results
└── {dataset}_evaluation2.json    # Third iteration results

{dataset}_prompt_evolution.json   # Complete prompt history
```

## Evaluation JSON Structure

```json
{
  "iteration": 0,
  "system_prompt": "Current system prompt...",
  "total_questions": 5,
  "evaluations": [
    {
      "question": "What is...?",
      "expected_answer": "The answer is...",
      "actual_response": "According to the context...",
      "context": "Relevant context...",
      "evaluation": "LLM evaluation feedback..."
    }
  ]
}
```

## Prompt Engineering Knowledge Base

The system requires a knowledge base of prompt engineering best practices. Include content covering:

- Core prompt engineering principles
- Common issues and solutions
- Advanced techniques (Chain of Thought, Role-based prompting, etc.)
- Domain-specific guidelines
- Examples of good and bad prompts

## Configuration Options

### Iteration Control
- `iteration_count`: Number of improvement cycles (typically 3-5)
- Each iteration includes evaluation + prompt improvement

### RAG Settings
- Main pipeline: Tune for your specific documents
- Prompt engineering pipeline: Optimize for prompt engineering content

### Evaluation Scope
- Training/test split: 80/20 by default
- Evaluation runs on training set for prompt improvement
- Use test set for final validation

## Best Practices

1. **Start Simple**: Begin with a basic system prompt to see maximum improvement
2. **Quality Data**: Ensure your Q&A dataset has good coverage of expected use cases  
3. **Knowledge Base**: Include comprehensive prompt engineering guidance
4. **Monitor Progress**: Review evaluation files to understand what's improving
5. **Manual Review**: Check automatically generated prompts before production use

## Troubleshooting

### Common Issues

1. **No Prompt Extraction**: If automatic prompt extraction fails, manually review the prompt engineering response
2. **Poor Evaluations**: Ensure your evaluator prompts are specific and detailed
3. **No Improvement**: Check that your prompt engineering knowledge base has relevant guidance
4. **File Conflicts**: The system handles iteration numbering automatically, including gaps

### Debug Output

The system provides extensive console output showing:
- Evaluation results for each question
- Prompt engineering analysis
- System prompt updates
- File locations for saved results

## Advanced Usage

### Custom Evaluators
You can modify the evaluation prompt in `evaluate_single_result()` to focus on specific aspects relevant to your use case.

### Custom Prompt Engineering
Override `_extract_new_prompt_from_response()` to handle different prompt formats or implement manual review workflows.

### Integration
The system is designed to integrate with existing RAG pipelines and can be extended for other model types beyond Ollama. 