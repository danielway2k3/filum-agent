"""
Embedding Generation Script for Filum Agent
Generates embeddings for knowledge_base.json and saves to kb.npy
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from utils import load_knowledge_base


def generate_embeddings():
    """Generate and save embeddings for knowledge base"""
    print("Loading knowledge base...")
    knowledge_base = load_knowledge_base("knowledge_base.json")

    print("Loading sentence transformer model...")
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("Generating embeddings...")
    texts = []
    for feature in knowledge_base:
        text = (
            feature["description"]
            + " "
            + " ".join(feature.get("associated_pain_points", []))
        )
        texts.append(text)

    embeddings = model.encode(texts, show_progress_bar=True)

    print("Saving embeddings to kb.npy...")
    np.save("kb.npy", embeddings)

    print(f"✓ Saved {embeddings.shape[0]} embeddings to kb.npy")


if __name__ == "__main__":
    generate_embeddings()
