r"""Supersphere geometry for domain routing.

Python port of the Rust supersphere project (A:\Ai\cognosphere).

This implements the core geometric components:
- N-dimensional unit sphere (Supersphere)
- Word kets as trainable vectors on the sphere
- Context construction from preceding word kets (exponential decay)
- Angular compatibility scoring (cosine similarity)

Used by DomainRouter to classify prompts and route to domain-specific SLMs.

NOT lossless. NOT part of the core ShardFit engine.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Supersphere:
    """N-dimensional Supersphere (unit sphere in R^N).

    All word kets live on this spherical surface. Angular relationships
    (cosine similarity) define geometric compatibility between words/contexts.
    """

    dimension: int

    def project(self, v: np.ndarray) -> np.ndarray:
        """Project a vector onto the unit sphere (normalize to unit length)."""
        norm = float(np.linalg.norm(v))
        if norm > 0.0:
            return v / norm
        return v.copy()

    def is_on_sphere(self, v: np.ndarray, tol: float = 1e-5) -> bool:
        """Check if a vector lies on the sphere (within tolerance)."""
        norm_sq = float(np.dot(v, v))
        return abs(norm_sq - 1.0) < tol

    def random_point(self, seed: int = 42) -> np.ndarray:
        """Generate a random point uniformly distributed on the sphere."""
        rng = np.random.RandomState(seed)
        v = rng.standard_normal(self.dimension)
        return self.project(v)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors on the sphere.

        Since both are unit vectors, this is just the dot product.
        Returns value in [-1, 1]. Higher = more similar.
        """
        return float(np.dot(a, b))

    def angular_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Angular distance (in radians) between two vectors on the sphere."""
        cos = np.clip(self.cosine_similarity(a, b), -1.0, 1.0)
        return float(math.acos(cos))

    def move_toward(self, point: np.ndarray, target: np.ndarray,
                    step_size: float) -> np.ndarray:
        """Move a point on the sphere along the geodesic toward a target.

        Implements the exponential map on the sphere.
        """
        cos_theta = np.clip(self.cosine_similarity(point, target), -1.0, 1.0)
        theta = math.acos(cos_theta)

        if theta < 1e-6:
            return point.copy()

        sin_theta = math.sin(theta)
        tangent = (target - cos_theta * point) / sin_theta

        step = min(step_size, theta)
        cos_step = math.cos(step)
        sin_step = math.sin(step)

        return cos_step * point + sin_step * tangent

    def retract(self, point: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        """Apply a tangent vector update and reproject to sphere.

        This is the retraction operation for gradient descent on the sphere.
        """
        return self.project(point + tangent)

    def project_to_tangent(self, point: np.ndarray,
                           vector: np.ndarray) -> np.ndarray:
        """Compute the tangent component of a vector at a point on the sphere.

        Projects vector onto the tangent space at point.
        """
        dot = float(np.dot(point, vector))
        return vector - dot * point


@dataclass
class WordKets:
    """Trainable word vectors on the Supersphere.

    Each vocabulary word has a ket represented by a unit vector on the sphere.
    Kets are initialized randomly and can evolve during training.
    """

    supersphere: Supersphere
    kets: Dict[str, np.ndarray] = field(default_factory=dict)

    def ket(self, word: str) -> Optional[np.ndarray]:
        """Get the ket vector for a word."""
        return self.kets.get(word)

    def add_ket(self, word: str,
                vector: Optional[np.ndarray] = None) -> None:
        """Add a word ket. If no vector given, initialize randomly."""
        if vector is None:
            seed = hash(word) % (2**32)
            vector = self.supersphere.random_point(seed)
        self.kets[word] = self.supersphere.project(np.array(vector, dtype=np.float64))

    def add_vocabulary(self, words: List[str]) -> None:
        """Add kets for multiple words at once."""
        for word in words:
            if word not in self.kets:
                self.add_ket(word)

    def cosine_similarity(self, word_a: str, word_b: str) -> Optional[float]:
        """Cosine similarity between two word kets."""
        ka = self.kets.get(word_a)
        kb = self.kets.get(word_b)
        if ka is None or kb is None:
            return None
        return self.supersphere.cosine_similarity(ka, kb)


@dataclass
class ContextConfig:
    """Configuration for context construction."""
    decay: float = 0.9
    max_length: int = 64


class ContextBuilder:
    """Build context ket from preceding word kets.

    Uses position-weighted exponential decay:
        c_t = normalize(sum_{i=1}^{L} alpha^{L-i} * k_{w_i})

    Recent words get higher weight. Result projected back to sphere.
    """

    def __init__(self, supersphere: Supersphere,
                 config: Optional[ContextConfig] = None):
        self.supersphere = supersphere
        self.config = config or ContextConfig()

    def build(self, word_ids: List[str], kets: WordKets) -> np.ndarray:
        """Build context ket from a sequence of word IDs.

        Uses the most recent max_length words with exponential decay.
        """
        if not word_ids:
            return self.supersphere.random_point()

        recent = word_ids[-self.config.max_length:]
        context = np.zeros(self.supersphere.dimension, dtype=np.float64)
        decay = self.config.decay

        for pos, word in enumerate(reversed(recent)):
            weight = decay ** pos
            ket = kets.ket(word)
            if ket is not None:
                context += weight * ket

        return self.supersphere.project(context)

    def build_incremental(self, prev_context: np.ndarray,
                          new_word: str, kets: WordKets) -> np.ndarray:
        """Build context incrementally for streaming/online inference."""
        context = prev_context.copy() * self.config.decay
        ket = kets.ket(new_word)
        if ket is not None:
            context += ket
        return self.supersphere.project(context)


class AngularScorer:
    """Angular scorer for geometric prediction.

    Given a context ket, compare it with candidate word kets using cosine
    similarity. Convert scores to probabilities via softmax with temperature.
    """

    def __init__(self, supersphere: Supersphere, temperature: float = 1.0):
        self.supersphere = supersphere
        self.temperature = temperature

    def score_all(self, context: np.ndarray,
                  kets: WordKets) -> Dict[str, float]:
        """Score all candidate words against a context ket."""
        scores: Dict[str, float] = {}
        for word, ket in kets.kets.items():
            scores[word] = self.supersphere.cosine_similarity(context, ket)
        return scores

    def softmax(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Convert scores to probabilities using softmax with temperature."""
        if not scores:
            return {}

        max_score = max(scores.values())
        exps: Dict[str, float] = {}
        total = 0.0

        for word, score in scores.items():
            exp_val = math.exp((score - max_score) / self.temperature)
            exps[word] = exp_val
            total += exp_val

        return {word: exp_val / total for word, exp_val in exps.items()}

    def predict_next(self, context: np.ndarray,
                     kets: WordKets) -> Optional[str]:
        """Get the most likely next word ID."""
        scores = self.score_all(context, kets)
        if not scores:
            return None
        return max(scores, key=scores.get)

    def top_k(self, context: np.ndarray, kets: WordKets,
              k: int) -> List[Tuple[str, float]]:
        """Get top-k predictions with probabilities."""
        scores = self.score_all(context, kets)
        probs = self.softmax(scores)
        sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]


class DomainSupersphere:
    """A supersphere model trained on a specific domain.

    Each domain has its own vocabulary, kets, and scorer. Used by DomainRouter
    to classify which domain a prompt belongs to.
    """

    def __init__(self, domain_name: str, dimension: int = 128):
        self.domain_name = domain_name
        self.dimension = dimension
        self.supersphere = Supersphere(dimension)
        self.kets = WordKets(self.supersphere)
        self.context_builder = ContextBuilder(self.supersphere)
        self.scorer = AngularScorer(self.supersphere)
        self.is_trained = False

    def add_vocabulary(self, words: List[str]) -> None:
        """Add words to this domain's vocabulary."""
        self.kets.add_vocabulary(words)

    def score_context(self, context_words: List[str]) -> float:
        """Score how well this domain fits the given context.

        Returns average cosine similarity of top-k predictions.
        Higher = better match.
        """
        context = self.context_builder.build(context_words, self.kets)
        scores = self.scorer.score_all(context, self.kets)
        if not scores:
            return 0.0
        sorted_scores = sorted(scores.values(), reverse=True)
        top_k = min(5, len(sorted_scores))
        return float(np.mean(sorted_scores[:top_k]))

    def save(self, path: str) -> None:
        """Save domain model to JSON file."""
        data = {
            "domain_name": self.domain_name,
            "dimension": self.supersphere.dimension,
            "kets": {word: ket.tolist() for word, ket in self.kets.kets.items()},
            "is_trained": self.is_trained,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                    exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DomainSupersphere":
        """Load domain model from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        domain = cls(data["domain_name"], data["dimension"])
        domain.kets.kets = {
            word: np.array(v, dtype=np.float64)
            for word, v in data["kets"].items()
        }
        domain.is_trained = data.get("is_trained", False)
        return domain


class DomainRouter:
    """Route prompts to domain-specific SLMs using supersphere geometry.

    Each domain has a supersphere model. When a prompt arrives, we build its
    context ket and score it against each domain. The domain with highest
    cosine similarity wins.

    No attention mechanism — pure geometric similarity.
    """

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.domains: Dict[str, DomainSupersphere] = {}

    def add_domain(self, domain_name: str,
                   vocabulary: List[str]) -> DomainSupersphere:
        """Add a new domain with its vocabulary."""
        domain = DomainSupersphere(domain_name, self.dimension)
        domain.add_vocabulary(vocabulary)
        self.domains[domain_name] = domain
        return domain

    def route(self, prompt: str, top_k: int = 3
              ) -> List[Tuple[str, float]]:
        """Route prompt to best matching domains.

        Returns list of (domain_name, confidence_score) sorted by score.
        """
        words = prompt.lower().split()
        if not words:
            return []

        scores: Dict[str, float] = {}
        for domain_name, domain in self.domains.items():
            scores[domain_name] = domain.score_context(words)

        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_domains[:top_k]

    def get_best_domain(self, prompt: str) -> Optional[str]:
        """Get the single best domain for a prompt."""
        routes = self.route(prompt, top_k=1)
        return routes[0][0] if routes else None

    def get_best_domain_with_confidence(self, prompt: str
                                        ) -> Tuple[Optional[str], float]:
        """Get best domain and its confidence score."""
        routes = self.route(prompt, top_k=1)
        if routes:
            return routes[0]
        return None, 0.0

    def save_all(self, directory: str) -> None:
        """Save all domain models to a directory."""
        os.makedirs(directory, exist_ok=True)
        for domain_name, domain in self.domains.items():
            domain.save(os.path.join(directory, f"{domain_name}.json"))

    @classmethod
    def load_all(cls, directory: str) -> "DomainRouter":
        """Load all domain models from a directory."""
        router = cls()
        if not os.path.isdir(directory):
            return router
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                domain = DomainSupersphere.load(
                    os.path.join(directory, filename)
                )
                router.domains[domain.domain_name] = domain
        return router
