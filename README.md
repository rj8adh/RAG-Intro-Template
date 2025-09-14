# Simple RAG Starter: RAG without a Vector Database

This repository provides a very basic, beginner-friendly Python script to demonstrate the core principles of Retrieval-Augmented Generation (RAG). It's designed to accompany the blog post "[Introduction to RAG: The Powerhouse Behind Chatbots](<Your Blog Post Link Here>)" and show the fundamental steps of RAG in a runnable way.

### Things To Keep In Mind

-   **Minimalist:** Everything is in a single, well-commented Python script.
-   **No Vector Database:** To keep things simple, this example performs the "retrieval" step in-memory using a Python list and NumPy for similarity calculations.
-   **Only Need One API Key:** Uses the Gemini Pro model for generation and Google's embedding model, so you only need one API key.

### How It Works (The RAG Steps)

This script follows the 3-step RAG process explained in the blog post:

1.  **"Indexing" the Knowledge Base:** We start with a small list of text documents (our "knowledge base"). We use the Google Embedding model to convert each document into a numerical vector (embedding). These are stored in a simple Python list.

2.  **Retrieval (The "R"):**
    -   When you provide a query (e.g., "What is the weather like?"), we first convert that query into an embedding using the same model.
    -   We then calculate the "similarity" between the query's embedding and all the document embeddings in our knowledge base. We use Cosine Similarity (calculated via a dot product on normalized vectors) to find which document is most semantically related to the query.
    -   The most similar document(s) are "retrieved."

3.  **Augment & Generate (The "AG"):**
    -   We take the original query and the retrieved document(s) and combine them into a single, comprehensive prompt for the LLM.
    -   The prompt looks something like this: `Using this information: [retrieved document], answer the following question: [original query]`.
    -   We send this augmented prompt to the Gemini model, which generates a final answer grounded in the provided context.

### Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/rj8adh/RAG-Intro-Template
    ```
    (cd into the directory if not already in)

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Your API Key:**
    -   Get a Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey).
    -   Rename the `.env.example` file to `.env`.
    -   Open the `.env` file and paste your API key:
        ```
        GOOGLE_API_KEY="YOUR_API_KEY_HERE"
        ```

### How to Run

Simply run the main Python script in your terminal:

```bash
python basic_rag.py
```


You can open `basic_rag.py` to change the `user_query` variable and ask different questions about the provided documents!
