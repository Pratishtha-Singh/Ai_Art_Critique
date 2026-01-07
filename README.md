# AI Art Critic

An AI-powered art critique system using Kedro and retrieval-augmented generation.

## Overview
This project generates intelligent, contextual critiques of artworks by retrieving similar pieces from a dataset of over 1,600 artworks and using a local LLM for analysis.

## Features
- Retrieval-Augmented Generation (RAG) for context-aware critiques
- Local LLM with GPT4All (Phi-3 Mini) - no API costs
- Faiss vector search for fast similarity retrieval
- Kedro pipelines for reproducible, scalable execution
- Error handling and timeouts for robustness

## Setup
1. Clone the repository:
   ```
   git clone https://github.com/Pratishtha-Singh/Ai_Art_Critique.git
   cd ai-art-critic
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run data processing (if needed):
   ```
   kedro run --pipeline data_processing
   ```

4. Generate critiques:
   ```
   kedro run --pipeline critique_generation --params query="Critique Van Gogh style"
   ```

## Project Structure
- `src/ai_art_critic/`: Source code with Kedro pipelines
- `notebooks/`: Jupyter notebooks for prototyping
- `data/`: Datasets and model inputs
- `conf/`: Configuration files

## Technologies
- Kedro: Pipeline orchestration
- Faiss: Vector similarity search
- GPT4All: Local LLM inference
- Transformers: BERT embeddings
- LangChain: LLM integration

## Contributing
Feel free to open issues or submit pull requests!