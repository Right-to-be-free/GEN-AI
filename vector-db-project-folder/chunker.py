import re
from typing import List

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Split long text into chunks with optional overlap.

    Args:
        text (str): The input text.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of words to overlap between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    words = re.findall(r'\S+', text)  # Split by any whitespace

    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks
