# Implement Filum Agent

## Overview
---
The Filum Agent is an intelligent pain point analysis system that combines **weighted keyword matching** and **semantic search** to provide accurate feature recommendations based on customer pain points.

## Installation
---
To install the Filum Agent, follow these steps:

1. Clone the repository
  ```bash
  git clone https://github.com/danielway2k3/filum-agent.git
  cd filum-agent
  ```
2. Create and activate Conda environment
  ```bash
  conda create -n filum-agent python=3.11
  conda activate filum-agent
  ```
3. Install dependencies
  ```bash
  pip install -r requirements.txt
  ```
4. Generate embeddings (required before first use)
  ```bash
  python embedding.py
  ```
5. Run the agent
  ```bash
  python main.py
  ```

## Project Structure
```
filum-agent/
├── agent.py              # Main agent class with hybrid scoring
├── embedding.py           # Generate pre-computed embeddings 
├── utils.py              # Text processing utilities
├── main.py               # Example usage and testing
├── knowledge_base.json   # Feature database
├── kb.npy               # Pre-computed embeddings (auto-generated)
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Algorithm Overview

The agent uses a 3-step hybrid approach:

1. **Pre-processing**: Normalizes input text (lowercase, remove punctuation and stop-words)
2. **Weighted Keyword Matching**: 
   - Keywords: weight = 1
   - Associated pain points: weight = 2
3. **Semantic Search**: Uses pre-computed embeddings with cosine similarity
4. **Final Score**: `Score_final = alpha * Score_keyword + (1-alpha) * Score_semantic`

Default alpha = 0.4 (40% keyword, 60% semantic)

## Usage

```python
from agent import PainPointAgent

# Initialize agent (uses kb.npy embeddings)
agent = PainPointAgent("knowledge_base.json", "kb.npy", alpha=0.4)

# Analyze pain point
result = agent.find_solutions({
    "pain_point_description": "Our support agents are overwhelmed by repetitive questions."
})

print(result)
```

## Notes
- Run `python embedding.py` whenever `knowledge_base.json` is updated
- File `kb.npy` contains pre-computed embeddings for fast performance
- Embeddings use `paraphrase-multilingual-MiniLM-L12-v2` model