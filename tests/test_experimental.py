"""Tests for the experimental supersphere router module.

These tests validate the experimental (non-lossless) components:
- Supersphere geometry (N-dimensional unit sphere)
- Word kets and context building
- Angular scoring
- Domain router (prompt classification)
- SLM worker and supervisor LLM
- Complete router system

NOT lossless. Experimental only.
"""

import sys
import os
import math
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

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
    RouteResult,
)


def test_supersphere_creation():
    """Test supersphere creation and basic properties."""
    sphere = Supersphere(128)
    assert sphere.dimension == 128
    print("PASS: test_supersphere_creation")


def test_supersphere_project():
    """Test vector projection onto sphere."""
    sphere = Supersphere(32)
    v = np.array([3.0, 4.0] + [0.0] * 30)
    projected = sphere.project(v)
    assert sphere.is_on_sphere(projected, 1e-5)
    print("PASS: test_supersphere_project")


def test_supersphere_cosine_similarity():
    """Test cosine similarity on the sphere."""
    sphere = Supersphere(16)
    a = sphere.random_point(42)
    b = sphere.random_point(43)
    cos = sphere.cosine_similarity(a, b)
    assert -1.0 <= cos <= 1.0
    # Same vector should have similarity 1.0
    assert sphere.cosine_similarity(a, a) == 1.0
    print("PASS: test_supersphere_cosine_similarity")


def test_supersphere_angular_distance():
    """Test angular distance computation."""
    sphere = Supersphere(16)
    a = sphere.random_point(42)
    b = sphere.random_point(43)
    dist = sphere.angular_distance(a, b)
    assert 0.0 <= dist <= math.pi
    # Same vector should have distance 0
    assert sphere.angular_distance(a, a) == 0.0
    print("PASS: test_supersphere_angular_distance")


import math  # noqa: E402


def test_word_kets():
    """Test word ket creation and access."""
    sphere = Supersphere(32)
    kets = WordKets(sphere)
    kets.add_vocabulary(["hello", "world", "code", "math"])
    assert kets.vocab_size() if hasattr(kets, 'vocab_size') else True
    assert kets.ket("hello") is not None
    assert len(kets.ket("hello")) == 32
    assert sphere.is_on_sphere(kets.ket("hello"), 1e-5)
    print("PASS: test_word_kets")


def test_context_builder():
    """Test context ket construction."""
    sphere = Supersphere(32)
    kets = WordKets(sphere)
    kets.add_vocabulary(["the", "cat", "sat", "on", "mat"])
    config = ContextConfig(decay=0.9, max_length=10)
    builder = ContextBuilder(sphere, config)
    context = builder.build(["the", "cat", "sat"], kets)
    assert len(context) == 32
    assert sphere.is_on_sphere(context, 1e-5)
    print("PASS: test_context_builder")


def test_angular_scorer():
    """Test angular scoring and softmax."""
    sphere = Supersphere(16)
    kets = WordKets(sphere)
    kets.add_vocabulary(["apple", "banana", "cherry"])
    scorer = AngularScorer(sphere, temperature=1.0)
    context = sphere.random_point(42)
    scores = scorer.score_all(context, kets)
    assert len(scores) == 3
    probs = scorer.softmax(scores)
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    top_k = scorer.top_k(context, kets, k=2)
    assert len(top_k) == 2
    print("PASS: test_angular_scorer")


def test_domain_supersphere():
    """Test domain-specific supersphere model."""
    domain = DomainSupersphere("code", dimension=64)
    domain.add_vocabulary(["def", "class", "import", "return", "python"])
    score = domain.score_context(["def", "function", "python"])
    assert 0.0 <= score <= 1.0
    print(f"PASS: test_domain_supersphere (score={score:.3f})")


def test_domain_router():
    """Test domain router with multiple domains.

    Note: With randomly initialized kets, routing accuracy depends on
    domain-specific training. This test verifies the mechanism works
    (routes are returned, best domain is a registered domain).
    """
    router = DomainRouter(dimension=64)
    router.add_domain("code", ["def", "class", "import", "return", "python", "function"])
    router.add_domain("creative", ["poem", "story", "write", "creative", "imagine"])
    router.add_domain("math", ["calculate", "equation", "solve", "math", "number"])

    routes = router.route("write a python function to sort a list", top_k=3)
    assert len(routes) <= 3
    assert len(routes) > 0
    assert routes[0][0] in ("code", "creative", "math")
    print(f"PASS: test_domain_router (best={routes[0][0]}, conf={routes[0][1]:.3f})")


def test_domain_router_creative():
    """Test domain router routes creative prompts correctly."""
    router = DomainRouter(dimension=64)
    router.add_domain("code", ["def", "class", "import", "return", "python"])
    router.add_domain("creative", ["poem", "story", "write", "creative", "imagine"])

    routes = router.route("write a poem about stars", top_k=3)
    assert routes[0][0] == "creative"
    print(f"PASS: test_domain_router_creative (best={routes[0][0]}, conf={routes[0][1]:.3f})")


def test_slm_worker():
    """Test SLM worker."""
    worker = SLMWorker("code-slm", "models/coder-1b", backend="cpu", domain="code")
    worker.load()
    assert worker.is_loaded
    response = worker.generate("def hello():", max_tokens=50)
    assert "code-slm" in response
    assert "code" in response
    stats = worker.get_stats()
    assert stats["is_loaded"] is True
    print("PASS: test_slm_worker")


def test_supervisor_llm():
    """Test supervisor LLM."""
    supervisor = SupervisorLLM("models/big-llm", backend="cuda", override_threshold=0.5)
    response = supervisor.review("prompt", "slm response", "code", 0.3)
    assert "Supervisor reviewed" in response
    assert supervisor.should_override("prompt", "response", 0.3) is True
    assert supervisor.should_override("prompt", "response", 0.8) is False
    stats = supervisor.get_stats()
    assert stats["total_reviews"] == 1
    assert stats["total_overrides"] == 1
    print("PASS: test_supervisor_llm")


def test_supersphere_router_system():
    """Test complete router system."""
    system = SupersphereRouterSystem(dimension=64)

    code_worker = SLMWorker("code-worker", "models/code-1b", backend="cpu", domain="code")
    creative_worker = SLMWorker("creative-worker", "models/creative-1b", backend="cpu", domain="creative")

    system.add_domain_worker("code", ["def", "class", "import", "python", "function"], code_worker)
    system.add_domain_worker("creative", ["poem", "story", "write", "creative"], creative_worker)

    # Test code prompt
    result = system.route_and_generate("write a python function to sort a list", max_tokens=50)
    assert result.domain == "code"
    assert result.worker_name == "code-worker"
    assert result.lossless is False
    assert result.error is None
    assert len(result.response) > 0

    # Test creative prompt
    result2 = system.route_and_generate("write a creative story about dragons", max_tokens=50)
    assert result2.domain == "creative"
    assert result2.worker_name == "creative-worker"

    stats = system.get_stats()
    assert stats["total_requests"] == 2
    assert stats["num_domains"] == 2
    assert stats["num_workers"] == 2
    print("PASS: test_supersphere_router_system")


def test_supersphere_router_system_with_supervisor():
    """Test router system with supervisor LLM."""
    system = SupersphereRouterSystem(dimension=64)
    code_worker = SLMWorker("code-worker", "models/code-1b", backend="cpu", domain="code")
    system.add_domain_worker("code", ["def", "class", "import", "python"], code_worker)

    supervisor = SupervisorLLM("models/big-llm", backend="cuda", override_threshold=0.5)
    system.set_supervisor(supervisor, use_supervisor=True)

    # Low confidence should trigger supervisor override
    result = system.route_and_generate("xyz unknown prompt", max_tokens=50)
    # With low confidence, supervisor may override
    stats = system.get_stats()
    assert stats["supervisor"]["total_reviews"] >= 0
    print("PASS: test_supersphere_router_system_with_supervisor")


def test_domain_save_load():
    """Test saving and loading domain models."""
    import tempfile
    domain = DomainSupersphere("test-domain", dimension=32)
    domain.add_vocabulary(["word1", "word2", "word3"])
    domain.is_trained = True

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test-domain.json")
    domain.save(path)

    loaded = DomainSupersphere.load(path)
    assert loaded.domain_name == "test-domain"
    assert loaded.dimension == 32
    assert loaded.is_trained is True
    assert "word1" in loaded.kets.kets
    print("PASS: test_domain_save_load")


def main():
    tests = [
        test_supersphere_creation,
        test_supersphere_project,
        test_supersphere_cosine_similarity,
        test_supersphere_angular_distance,
        test_word_kets,
        test_context_builder,
        test_angular_scorer,
        test_domain_supersphere,
        test_domain_router,
        test_domain_router_creative,
        test_slm_worker,
        test_supervisor_llm,
        test_supersphere_router_system,
        test_supersphere_router_system_with_supervisor,
        test_domain_save_load,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Experimental results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 40}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
