# Implement Filum Agent

## Overview
---
The Filum Agent is an intelligent pain point analysis system that combines **weighted keyword matching** and **semantic search** to provide accurate feature recommendations based on customer pain points.

## Installation
---
To install the Filum Agent, follow these steps:

1. Clone the repository
  ```bash
  git clone https://github.com/yourusername/filum-agent.git
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
4. Run the agent
  ```bash
  python main.py
  ```

## Algorithm Overview

The agent uses a 3-step process:

1. **Pre-processing**: Normalizes input text (lowercase, remove punctuation and stop-words)
2. **Weighted Keyword Matching**: 
   - Keywords: weight = 1
   - Associated pain points: weight = 2
3. **Semantic Search**: Uses sentence-transformers to calculate cosine similarity
4. **Final Score**: `Score_final = alpha * Score_keyword + (1-alpha) * Score_semantic`

Default alpha = 0.2 (20% keyword, 80% semantic)