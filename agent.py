from utils import normalize_text, load_knowledge_base
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os


class PainPointAgent:
    def __init__(self, knowledge_base_path, embeddings_path="kb.npy", alpha=0.4):
        """
        Initialize the PainPointAgent with pre-computed embeddings

        Args:
            knowledge_base_path (str): Path to knowledge base JSON file
            embeddings_path (str): Path to pre-computed embeddings .npy file
            alpha (float): Weight for keyword vs semantic scoring (0-1)
        """
        self.knowledge_base = load_knowledge_base(knowledge_base_path)
        if not self.knowledge_base:
            raise ValueError("Knowledge base is empty or could not be loaded.")

        # Load semantic model (only for user input embedding)
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.alpha = alpha  # Weight for keyword vs semantic score

        # Load pre-computed embeddings
        self._load_precomputed_embeddings(embeddings_path)

    def _load_precomputed_embeddings(self, embeddings_path):
        """Load pre-computed embeddings from .npy file"""
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"Embeddings file not found: {embeddings_path}\n"
                f"Please run 'python embedding.py' to generate embeddings first."
            )

        print(f"Loading pre-computed embeddings from: {embeddings_path}")
        embeddings = np.load(embeddings_path)

        # Verify dimensions match
        if len(self.knowledge_base) != embeddings.shape[0]:
            raise ValueError(
                f"Dimension mismatch: Knowledge base has {len(self.knowledge_base)} features "
                f"but embeddings file has {embeddings.shape[0]} embeddings. "
                f"Please regenerate embeddings with 'python embedding.py'"
            )

        # Assign embeddings to features
        for i, feature in enumerate(self.knowledge_base):
            feature["embedding"] = embeddings[i]

        print(f"✓ Loaded {embeddings.shape[0]} pre-computed embeddings")

    def _calculate_keyword_score(self, normalized_pain_point, feature):
        """Step 1: Weighted keyword matching"""
        match_count = 0

        # Match with keywords (weight = 1)
        for keyword in feature.get("keywords", []):
            if normalize_text(keyword) in normalized_pain_point:
                match_count += 1

        # Match with associated_pain_points (weight = 2)
        for pain_point in feature.get("associated_pain_points", []):
            if normalize_text(pain_point) in normalized_pain_point:
                match_count += 2

        total_possible = len(feature.get("keywords", [])) + 2 * len(
            feature.get("associated_pain_points", [])
        )
        return match_count / total_possible if total_possible > 0 else 0

    def _calculate_semantic_score(self, pain_point_embedding, feature):
        """Step 2: Calculate semantic similarity score using pre-computed embeddings"""
        feature_embedding = feature["embedding"].reshape(1, -1)
        pain_point_embedding = pain_point_embedding.reshape(1, -1)

        similarity = cosine_similarity(pain_point_embedding, feature_embedding)[0][0]
        return float(
            max(0, similarity)
        ) 

    def find_solutions(self, user_input, k=5):
        """Find solutions that match user's pain points"""
        # Pre-process input
        pain_point = user_input.get("pain_point_description", "")
        normalized_pain_point = normalize_text(pain_point)

        # Create embedding for pain point
        pain_point_embedding = self.model.encode(pain_point)

        solutions = []

        for feature in self.knowledge_base:
            # Step 1: Calculate keyword score
            keyword_score = self._calculate_keyword_score(
                normalized_pain_point, feature
            )

            # Step 2: Calculate semantic score using pre-computed embeddings
            semantic_score = self._calculate_semantic_score(
                pain_point_embedding, feature
            )

            # Step 3: Calculate final score
            final_score = float(
                self.alpha * keyword_score + (1 - self.alpha) * semantic_score
            )

            # Only add to results if score > 0
            if final_score > 0:
                solutions.append(
                    {
                        "feature_name": feature["feature_name"],
                        "product_category": feature["product_category"],
                        "how_it_helps": feature["description"],
                        "relevance_score": round(final_score, 2),
                        "keyword_score": round(float(keyword_score), 2),
                        "semantic_score": round(float(semantic_score), 2),
                    }
                )

        # Sort by relevance score in descending order
        solutions.sort(key=lambda x: x["relevance_score"], reverse=True)

        return {"suggested_solutions": solutions[:k]}
