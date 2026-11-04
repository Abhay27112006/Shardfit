"""Experimental Supersphere Router module.

This is RESEARCH/EXPERIMENTAL code — NOT part of the core ShardFit lossless
streaming engine. It does NOT satisfy the lossless principle (section 3 of the
brief). Use only for exploring prompt-aware SLM routing.

Architecture:
- Supersphere geometry (N-dimensional unit sphere) for domain classification
- Domain-specific SLM workers routed by geometric similarity
- Optional supervisor LLM that reviews/corrects worker outputs

NOT lossless. NOT a replacement for the core engine.
"""

from shardfit.experimental.supersphere_geometry import (
    Supersphere,
    WordKets,
    ContextConfig,
    ContextBuilder,
    AngularScorer,
    DomainSupersphere,
    DomainRouter,
)
from shardfit.experimental.router import (
    SLMWorker,
    SupervisorLLM,
    SupersphereRouterSystem,
)

__all__ = [
    "Supersphere",
    "WordKets",
    "ContextConfig",
    "ContextBuilder",
    "AngularScorer",
    "DomainSupersphere",
    "DomainRouter",
    "SLMWorker",
    "SupervisorLLM",
    "SupersphereRouterSystem",
]

__experimental__ = True
__lossless__ = False
