"""
Unit tests for advanced features.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.cache import create_cache, MemoryCache
from rag.metrics import MetricsCollector
from rag.prompts import PromptOptimizer, GuardrailValidator
from rag.reranker import SimpleReranker


def test_cache():
    """Test cache functionality."""
    print("=" * 60)
    print("Testing Cache...")
    print("=" * 60)

    cache = create_cache(redis_url=None, ttl=60, enabled=True)

    cache.set(
        query="What experience does Evelyn Hamilton have?",
        top_k=5,
        response="Evelyn Hamilton is a data engineer with experience in AWS.",
        sources=[{"url": "cv://cv-01-evelyn-hamilton"}]
    )

    result = cache.get("What experience does Evelyn Hamilton have?", 5)
    assert result is not None, "Cache should return result"
    print("✅ Cache set/get works")

    result = cache.get("what experience does evelyn hamilton have?", 5)
    assert result is not None, "Cache should normalize queries"
    print("✅ Query normalization works")

    stats = cache.get_stats()
    print(f"✅ Cache stats: {stats['hits']} hits, {stats['misses']} misses")
    print(f"   Hit rate: {stats['hit_rate']:.2%}")

    print()


def test_metrics():
    """Test metrics collector."""
    print("=" * 60)
    print("Testing Metrics...")
    print("=" * 60)

    metrics = MetricsCollector(enabled=True, port=8001)

    metrics.record_cache_hit()
    metrics.record_cache_miss()
    metrics.record_retrieval(5, [0.9, 0.85, 0.8, 0.75, 0.7])
    metrics.record_token_usage(1000, 200)

    with metrics.trace_pipeline("Test query") as trace:
        with metrics.measure_stage(trace, "retrieval"):
            pass  # Simulate retrieval

        with metrics.measure_stage(trace, "generation"):
            pass  # Simulate generation

    traces = metrics.get_recent_traces(limit=1)
    assert len(traces) > 0, "Should have recorded trace"
    print(f"✅ Recorded {len(traces)} trace(s)")
    print(f"   Stages: {list(traces[0]['stages'].keys())}")

    stats = metrics.get_stats()
    print(f"✅ Metrics stats: {stats}")

    print()


def test_prompt_optimizer():
    """Test prompt optimizer."""
    print("=" * 60)
    print("Testing Prompt Optimizer...")
    print("=" * 60)

    optimizer = PromptOptimizer(
        use_few_shot=True,
        validate_output=True
    )

    prompt = optimizer.create_prompt(
        context="Evelyn Hamilton is a data engineer...",
        chat_history="",
        question="What is the profile of Evelyn Hamilton?"
    )

    assert "META-INSTRUCTIONS" in prompt[0] or "INSTRUCTIONS" in prompt[0], "Should include guardrails"
    assert "Example" in prompt[0], "Should include few-shot examples"
    print("✅ Prompt includes guardrails and few-shot")

    response = "Evelyn Hamilton is a data engineer specialized in AWS [1]."
    sources = [{"url": "cv://cv-01-evelyn-hamilton", "content": "..."}]

    validation = optimizer.validate_response(response, "context", sources)
    print(f"✅ Response validation: {validation['passed']}")
    print(f"   Score: {validation['score']:.2f}")
    print(f"   Issues: {validation['issues']}")

    validator = GuardrailValidator()
    validation = validator.validate_response(
        response="I think Caitlin Cannon makes 120000€ per year",
        context="",
        sources=[],
        language="en"
    )
    hallucination_issues = [issue for issue in validation['issues'] if
                            'Hallucination' in issue or 'hallucination' in issue]
    print(f"✅ Hallucination detection: {len(hallucination_issues)} issues found")
    for issue in hallucination_issues:
        print(f"   - {issue}")

    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RAG Advanced Features - Test Suite")
    print("=" * 60 + "\n")

    try:
        test_cache()
        test_metrics()
        test_prompt_optimizer()

        print("=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Configure .env file with new variables")
        print("3. Optional: Start Redis for production cache")
        print("4. Run application: python api.py")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
