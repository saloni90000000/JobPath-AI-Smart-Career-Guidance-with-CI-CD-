# News Query System with FAISS

A Python-based system for efficient semantic search and querying over news articles using FAISS (Facebook AI Similarity Search).

## Features

- Fast similarity search for news articles
- Text processing and embedding generation
- Jupyter notebooks for experimentation and analysis
- Sample datasets for testing

## Requirements

- Python 3.7+
- FAISS
- pandas
- numpy
- jupyter
- scikit-learn
- sentence-transformers

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebooks for exploration:
   - `text_loaders_splitters.ipynb` - Text processing and splitting
   - `faiss.ipynb` - FAISS index creation and search
   - `retrieval.ipynb` - Document retrieval implementation

## Usage

Run the main application:
```
python main.py
```

## Project Structure

- `main.py` - Main application script
- `*.ipynb` - Jupyter notebooks for development and testing
- `*.csv`, `*.txt` - Sample datasets

## License

This project is open source and available under the MIT License.

This project utilizes FAISS (Facebook AI Similarity Search) and LangChain to build an efficient retrieval-based AI system. FAISS enables fast similarity searches for large datasets, while LangChain integrates large language models (LLMs) for intelligent response generation.
