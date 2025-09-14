import os
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Configure the Google Generative AI with the API key
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    print(os.getenv("GOOGLE_API_KEY"))
except AttributeError:
    print("Error: GOOGLE_API_KEY not found. Please set it in your .env file.")
    exit()

# --- 1. The Knowledge Base ---
# This is our simple, "database" of documents.
# In a real-world scenario, this would probably just be in a vector database
documents = [
    "The weather in Arizona has hit a record low of 114 degrees.",
    "Chipotle is crowned the greatest restaurant chain in the world.",
    "Why is RAG so much easier to use than I expected?"
]

print("Original Documents:")
for doc in documents:
    print(f"- {doc}")
print("-" * 20)


# --- 2. Embedding the Documents ---
# This is the "indexing" step. We turn our text documents into numerical vectors/coords so we can do mathematical comparisons later.

# We will be using one of google's embedding models
embedding_model = "models/text-embedding-004"


# Create the embeddings one by one
# NOTE: (you usually don't have to do this, but we're not using a vector database for this simplified example)
document_embeddings = []
for doc in documents:
    try:
        # The actual embedding to get the vectors
        embedding = genai.embed_content(
            model=embedding_model,
            content=doc,
            task_type="RETRIEVAL_DOCUMENT",
            title="News Headlines"
        )['embedding']
        document_embeddings.append((doc, embedding))
    except Exception as e:
        print(f"Embedding failed for document: {doc}\nError: {e}")
        document_embeddings.append((doc, None))


def find_best_passage(query, document_embeddings, top_k=1):
    """
    Finds the most relevant passage from the document embeddings based on the query.

    Args:
        query (str): The user's question.
        document_embeddings (list): A list of tuples, where each tuple contains
                                    the original document text and its embedding.
        top_k (int): The number of top passages to retrieve.

    Returns:
        str: The most relevant passage(s) from the documents.
    """
    # Embed the user's query. The task_type is "RETRIEVAL_QUERY" for the query.
    query_embedding = genai.embed_content(
        model=embedding_model,
        content=query,
        task_type="RETRIEVAL_QUERY"
    )['embedding']

    # --- 3. The Retrieval Step ---
    # We calculate the similarity between the query embedding and all document embeddings.
    # Cosine similarity is the most commonly used for finding the closeness of each vector.
    dot_products = np.dot(np.stack(list(zip(*document_embeddings))[1]), query_embedding)
    
    # Get the indices of the top_k most similar documents (higher the value = the more context you give to the chatbot)
    top_indices = np.argsort(dot_products)[-top_k:][::-1]
    
    # Retrieve the corresponding documents
    retrieved_passages = [document_embeddings[i][0] for i in top_indices]
    
    return "\n".join(retrieved_passages)


# --- 4. The Augment & Generate Step ---

# Define a user query
user_query = "What's the weather like"

# 1. Retrieve the most relevant passage
retrieved_context = find_best_passage(user_query, document_embeddings)

print(f"User Query: {user_query}")
print(f"Retrieved Context:\n- {retrieved_context}")
print("-" * 20)


# 2. Augment the prompt with the retrieved context
prompt = f"""
You are a News Headline chatbot.
Answer the user's question based ONLY on the following context.
If the context doesn't contain the answer, say "I'm sorry, I don't have that information."
Start your response with "The information you're looking for is: "

Context:
{retrieved_context}

Question:
{user_query}
"""

# 3. Generate the final answer using an LLM
generative_model = genai.GenerativeModel('gemini-1.5-flash')
final_answer = generative_model.generate_content(prompt)

print("Generated Answer:")
print(final_answer.text)

"""
In the end, this code simply finds the 1st most relevant headline and passes that to an AI, and that AI will
in turn generate a response using the headline it was given.
"""