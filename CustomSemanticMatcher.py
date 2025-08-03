import os
from openai import AzureOpenAI, azure_endpoint
from dotenv import load_dotenv
import os
import numpy as np
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple
import json
import re

# Load environment variables from .env file
load_dotenv()


class CustomSemanticMatcher:
    def __init__(self):
        """
        Initialize the Azure OpenAI client for embeddings

        Args:
            azure_endpoint: Your Azure OpenAI endpoint
            api_key: Your Azure OpenAI API key
            api_version: API version (default: "2023-05-15")
            deployment_name: Name of your embedding model deployment (default: text-embedding-3-small)
        """

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=subscription_key,
            api_version="2024-12-01-preview",
        )

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text using Azure OpenAI

        Args:
            text: Input text to embed

        Returns:
            List of float values representing the embedding
        """
        deployment_name = "text-embedding-3-small"
        try:
            response = self.client.embeddings.create(
                input=text,
                model=deployment_name
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding for text: {text[:50]}...")
            print(f"Error: {e}")
            return []

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            if embedding:  # Only add if embedding was successful
                embeddings.append(embedding)
            else:
                # Add zero vector as placeholder for failed embeddings
                # text-embedding-3-small has 1536 dimensions
                embeddings.append([0.0] * 1536)
        return embeddings

    def preprocess_text(self, text: str) -> str:
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text.strip()

    def calculate_keyword_similarity(self, slide_texts: List[str], summary_texts: List[str]) -> np.ndarray:
        processed_slide_texts = [
            self.preprocess_text(text) for text in slide_texts]
        processed_summary_texts = [
            self.preprocess_text(text) for text in summary_texts]

        # Combine all texts for TF-IDF fitting
        all_texts = processed_slide_texts + processed_summary_texts

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),  # Include both unigrams and bigrams
            min_df=1,
            max_df=0.95
        )

        # Fit and transform all texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Split back into slides and summaries
        slide_tfidf = tfidf_matrix[:len(processed_slide_texts)]
        summary_tfidf = tfidf_matrix[len(processed_slide_texts):]

        # Calculate cosine similarity
        keyword_similarity = cosine_similarity(slide_tfidf, summary_tfidf)

        return keyword_similarity

    def find_semantic_matches(self, list_slide_summaries: List[Dict],
                              list_summaries: List[Dict],
                              top_k: int = 1,
                              use_hybrid: bool = True,
                              semantic_weight: float = 0.6,
                              keyword_weight: float = 0.4) -> List[Dict]:
        if abs(semantic_weight + keyword_weight - 1.0) > 1e-6:
            print(f"Warning: Weights don't sum to 1.0. Normalizing...")
            total_weight = semantic_weight + keyword_weight
            semantic_weight = semantic_weight / total_weight
            keyword_weight = keyword_weight / total_weight

        # Extract texts for processing
        slide_texts = [item['text'] for item in list_slide_summaries]
        summary_texts = [item['summary'] for item in list_summaries]

        # Calculate semantic similarity using embeddings
        print("Getting embeddings for slide summaries...")
        slide_embeddings = self.get_embeddings_batch(slide_texts)

        print("Getting embeddings for summaries...")
        summary_embeddings = self.get_embeddings_batch(summary_texts)

        # Convert to numpy arrays for similarity calculation
        slide_embeddings_np = np.array(slide_embeddings)
        summary_embeddings_np = np.array(summary_embeddings)

        # Calculate semantic similarities
        semantic_similarity = cosine_similarity(
            slide_embeddings_np, summary_embeddings_np)

        # Calculate final similarity matrix
        if use_hybrid:
            print("Calculating keyword-based similarities...")
            keyword_similarity = self.calculate_keyword_similarity(
                slide_texts, summary_texts)

            # Combine semantic and keyword similarities
            final_similarity = (semantic_weight * semantic_similarity +
                                keyword_weight * keyword_similarity)

            print(
                f"Using hybrid search (Semantic: {semantic_weight:.1f}, Keyword: {keyword_weight:.1f})")
        else:
            final_similarity = semantic_similarity
            print("Using semantic search only")

        # Create results
        results = []

        for i, slide_item in enumerate(list_slide_summaries):
            # Get similarities for this slide with all summaries
            similarities = final_similarity[i]

            # Create list of (summary_index, similarity_score, semantic_score, keyword_score) tuples
            similarity_data = []
            for j in range(len(list_summaries)):
                semantic_score = semantic_similarity[i][j]
                keyword_score = keyword_similarity[i][j] if use_hybrid else 0.0
                combined_score = similarities[j]

                similarity_data.append(
                    (j, combined_score, semantic_score, keyword_score))

            # Sort by combined similarity score (descending)
            similarity_data.sort(key=lambda x: x[1], reverse=True)

            # Limit to top_k if specified
            if top_k:
                similarity_data = similarity_data[:top_k]

            # Create match objects with detailed scoring
            for rank, (summary_idx, combined_score, semantic_score, keyword_score) in enumerate(similarity_data):
                match = {
                    'slide_number': slide_item['slide_number'],
                    'slide_text': slide_item['text'],
                    'matched_summary_id': list_summaries[summary_idx]['id'],
                    'matched_summary_text': list_summaries[summary_idx]['summary'],
                    'similarity_score': float(combined_score),
                    'semantic_score': float(semantic_score),
                    'keyword_score': float(keyword_score),
                    'search_type': 'hybrid' if use_hybrid else 'semantic',
                    'rank': rank + 1
                }
                results.append(match)

        return results
