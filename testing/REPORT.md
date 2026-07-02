# Testing Report — Sentiment Analysis Service

**Branch:** `feature/system-testing`  
**Regression Test:** `test_regression.py`  
**Environment:** FastAPI TestClient and GitHub Actions

---

## Latency and Load Testing

The API was tested using repeated and concurrent requests to evaluate its performance.

| Metric | Sequential (100 requests) | Concurrent (20 × 5 requests) |
|--------|--------------------------:|-----------------------------:|
| Average latency | 2.31 ms | 45.93 ms |
| p95 latency | 2.68 ms | 66.54 ms |
| p99 latency | 3.80 ms | — |
| Throughput | 499 requests/sec | 354.9 requests/sec |
| Error rate | 0% | 0% |

The sequential test showed very good performance with an average latency of **2.31 ms**. Under concurrent requests, latency increased but all requests completed successfully and the error rate remained **0%**.

---

## Consistency Check

The same request was sent **50 times**.

| Check | Result |
|-------|--------|
| Label | Always **Positive** |
| Confidence | Always **0.6647** |
| Consistent output | ✓ Yes |

The model produced the same prediction and confidence score for every identical request, confirming deterministic behaviour.

---

## Regression Test Results

The regression test included **9 tests**:

- 5 known sentence predictions
- 2 input validation tests
- 2 output validation tests

All tests passed successfully.

| Test | Result |
|------|--------|
| Known sentence predictions | ✓ PASSED |
| Empty input | ✓ PASSED |
| Whitespace input | ✓ PASSED |
| Confidence range | ✓ PASSED |
| Valid label check | ✓ PASSED |

The negation examples (**"This is not good at all..."** and **"Not bad, actually quite impressed..."**) are especially important because they verify that the model continues to handle negation correctly after future code or model changes.

---

## Production Readiness Assessment

The service is suitable for low-traffic internal use but is not yet ready for high-concurrency production deployment. Testing showed a **0% error rate**, consistent predictions, and successful regression tests. The main limitation is the increase in response time under concurrent requests. Although no requests failed, latency increased noticeably as more requests were processed at the same time.

The first improvement I would make is to optimize concurrent performance by using multiple Uvicorn workers or another method of processing requests more efficiently. This reflects my evaluation philosophy that a machine learning service should be assessed not only by prediction accuracy, but also by its reliability, consistency, and performance under real-world usage.
