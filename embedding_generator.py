from sentence_transformers import SentenceTransformer
import numpy as np

# Load the MiniLM model
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(text_list):
    """
    Generate vector embeddings for a list of text strings.
    
    Parameters:
        text_list (List[str]): The list of text documents or chunks.

    Returns:
        List[np.array]: List of embedding vectors.
    """
    try:
        embeddings = model.encode(text_list, show_progress_bar=True)
        return embeddings
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return []
