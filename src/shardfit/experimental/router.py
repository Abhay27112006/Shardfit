"""SLM workers and supervisor LLM for the supersphere router system.

These components work with DomainRouter to create a complete
prompt-routed SLM system:

- SLMWorker: small language model for a specific domain
- SupervisorLLM: bigger LLM that reviews/corrects worker outputs
- SupersphereRouterSystem: ties router + workers + supervisor together

NOT lossless. Experimental only.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from shardfit.experimental.supersphere_geometry import DomainRouter


class SLMWorker:
    """Interface to a small language model worker for a specific domain.

    SLMs are small enough to run on consumer hardware (even Intel Celeron
    CPU). Each worker handles one domain (code, math, creative, etc.).
    """

    def __init__(self, name: str, model_path: str, backend: str = "cpu",
                 domain: str = "general"):
        self.name = name
        self.model_path = model_path
        self.backend = backend
        self.domain = domain
        self.is_loaded = False
        self.total_tokens = 0
        self.total_time_s = 0.0

    def load(self) -> None:
        """Load the SLM into memory."""
        self.is_loaded = True

    def unload(self) -> None:
        """Unload the SLM to free memory."""
        self.is_loaded = False

    def generate(self, prompt: str, max_tokens: int = 128,
                 temperature: float = 0.7) -> str:
        """Generate a response using the SLM.

        Placeholder implementation. In production, this calls the actual
        model inference (llama.cpp, transformers, etc.).
        """
        if not self.is_loaded:
            self.load()

        start = time.perf_counter()

        response = (
            f"[{self.name} domain={self.domain}] "
            f"Response to: '{prompt[:80]}...'"
        )

        elapsed = time.perf_counter() - start
        self.total_time_s += elapsed
        self.total_tokens += max_tokens

        return response

    def get_stats(self) -> dict:
        """Return worker statistics."""
        return {
            "name": self.name,
            "domain": self.domain,
            "backend": self.backend,
            "is_loaded": self.is_loaded,
            "total_tokens": self.total_tokens,
            "total_time_s": self.total_time_s,
            "avg_tok_s": (self.total_tokens / self.total_time_s
                          if self.total_time_s > 0 else 0.0),
        }


class SupervisorLLM:
    """Bigger LLM that supervises and reviews SLM outputs.

    The supervisor reviews SLM responses and can override them if
    confidence is low or the response is poor.
    """

    def __init__(self, model_path: str, backend: str = "cuda",
                 override_threshold: float = 0.5):
        self.model_path = model_path
        self.backend = backend
        self.override_threshold = override_threshold
        self.total_reviews = 0
        self.total_overrides = 0

    def review(self, prompt: str, slm_response: str,
               domain: str, confidence: float) -> str:
        """Review an SLM response and return refined output.

        Placeholder implementation. In production, calls the bigger LLM
        to review and potentially rewrite the SLM response.
        """
        self.total_reviews += 1

        if confidence < self.override_threshold:
            self.total_overrides += 1
            return (
                f"[Supervisor reviewed ({domain}, conf={confidence:.2f})] "
                f"{slm_response}"
            )
        return slm_response

    def should_override(self, prompt: str, slm_response: str,
                        confidence: float) -> bool:
        """Decide if supervisor should override the SLM response.

        Override when confidence is below threshold.
        """
        return confidence < self.override_threshold

    def get_stats(self) -> dict:
        """Return supervisor statistics."""
        return {
            "model_path": self.model_path,
            "backend": self.backend,
            "override_threshold": self.override_threshold,
            "total_reviews": self.total_reviews,
            "total_overrides": self.total_overrides,
            "override_rate": (self.total_overrides / self.total_reviews
                              if self.total_reviews > 0 else 0.0),
        }


@dataclass
class RouteResult:
    """Result of routing a prompt to an SLM."""

    prompt: str
    response: str
    domain: str
    confidence: float
    routes: List[tuple] = field(default_factory=list)
    worker_name: str = ""
    supervisor_overridden: bool = False
    lossless: bool = False
    error: Optional[str] = None


class SupersphereRouterSystem:
    """Complete system: supersphere router + SLM workers + supervisor LLM.

    Flow:
        1. Prompt arrives
        2. Supersphere router classifies prompt to domain (geometric similarity)
        3. Best domain's SLM worker generates response
        4. Optional: supervisor LLM reviews and overrides if confidence low
        5. Return result with routing metadata
    """

    def __init__(self, dimension: int = 128):
        self.router = DomainRouter(dimension)
        self.workers: Dict[str, SLMWorker] = {}
        self.supervisor: Optional[SupervisorLLM] = None
        self.use_supervisor = True
        self.total_requests = 0
        self.domain_usage: Dict[str, int] = {}

    def add_domain_worker(self, domain_name: str,
                          vocabulary: List[str],
                          slm_worker: SLMWorker) -> None:
        """Register a domain with its vocabulary and SLM worker."""
        self.router.add_domain(domain_name, vocabulary)
        self.workers[domain_name] = slm_worker

    def set_supervisor(self, supervisor: SupervisorLLM,
                       use_supervisor: bool = True) -> None:
        """Set the supervisor LLM and whether to use it."""
        self.supervisor = supervisor
        self.use_supervisor = use_supervisor

    def route_and_generate(self, prompt: str,
                           max_tokens: int = 128,
                           temperature: float = 0.7,
                           use_supervisor: Optional[bool] = None
                          ) -> RouteResult:
        """Route prompt to best SLM and generate response.

        Uses supersphere geometry (cosine similarity) to select domain,
        then generates with that domain's SLM worker.
        """
        if use_supervisor is None:
            use_supervisor = self.use_supervisor

        self.total_requests += 1

        routes = self.router.route(prompt, top_k=3)
        best_domain, best_confidence = routes[0] if routes else ("general", 0.0)

        self.domain_usage[best_domain] = (
            self.domain_usage.get(best_domain, 0) + 1
        )

        worker = self.workers.get(best_domain)
        if worker is None:
            worker = next(iter(self.workers.values())) if self.workers else None

        if worker is None:
            return RouteResult(
                prompt=prompt,
                response="",
                domain=best_domain,
                confidence=best_confidence,
                routes=routes,
                worker_name="",
                supervisor_overridden=False,
                lossless=False,
                error="No SLM workers available",
            )

        response = worker.generate(prompt, max_tokens, temperature)

        supervisor_overridden = False
        if use_supervisor and self.supervisor is not None:
            if self.supervisor.should_override(prompt, response,
                                                best_confidence):
                response = self.supervisor.review(
                    prompt, response, best_domain, best_confidence
                )
                supervisor_overridden = True

        return RouteResult(
            prompt=prompt,
            response=response,
            domain=best_domain,
            confidence=best_confidence,
            routes=routes,
            worker_name=worker.name,
            supervisor_overridden=supervisor_overridden,
            lossless=False,
        )

    def get_stats(self) -> dict:
        """Return system statistics."""
        return {
            "total_requests": self.total_requests,
            "domain_usage": self.domain_usage,
            "num_domains": len(self.router.domains),
            "num_workers": len(self.workers),
            "supervisor": self.supervisor.get_stats() if self.supervisor else None,
            "use_supervisor": self.use_supervisor,
            "workers": {name: w.get_stats() for name, w in self.workers.items()},
        }

    def save_router(self, directory: str) -> None:
        """Save router domain models to disk."""
        self.router.save_all(directory)

    @classmethod
    def load_router(cls, directory: str) -> "SupersphereRouterSystem":
        """Load router domain models from disk."""
        system = cls()
        system.router = DomainRouter.load_all(directory)
        return system
