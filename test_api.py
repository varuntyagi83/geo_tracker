#!/usr/bin/env python3
"""
Test script for GEO Tracker API.

Usage:
    1. Start the API: uvicorn api.main:app --reload --port 8000
    2. Run this script: python test_api.py
"""
import requests
import time
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("\n=== Testing Health ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_generate_queries():
    """Test query generation endpoint."""
    print("\n=== Testing Query Generation ===")
    params = {
        "industry": "supplements",
        "company_name": "Sunday Natural",
        "count": 5
    }
    response = requests.post(f"{BASE_URL}/api/queries/generate", params=params)
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Generated {data['count']} queries:")
    for q in data["queries"]:
        print(f"  - {q['question']}")
    return response.status_code == 200


def test_run_workflow():
    """Test the full run workflow."""
    print("\n=== Testing Full Run Workflow ===")
    
    # 1. Create a run
    print("\n1. Creating run...")
    run_config = {
        "company_id": "test-company",
        "brand_name": "Sunday Natural",
        "providers": ["openai"],  # Use only OpenAI for faster test
        "mode": "internal",  # Internal mode for faster response
        "queries": [
            {
                "question": "What are some popular vitamin D supplement brands?",
                "category": "product_recommendation",
                "prompt_id": "test_1"
            },
            {
                "question": "How do I choose a good magnesium supplement?",
                "category": "general_advice",
                "prompt_id": "test_2"
            }
        ],
        "market": "DE",
        "lang": "de",
        "request_timeout": 30,
        "max_retries": 0
    }
    
    response = requests.post(f"{BASE_URL}/api/runs", json=run_config)
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return False
    
    job = response.json()
    job_id = job["job_id"]
    print(f"Job created: {job_id}")
    print(f"Message: {job['message']}")
    print(f"Estimated duration: {job['estimated_duration_seconds']}s")
    
    # 2. Poll for status
    print("\n2. Polling for progress...")
    max_wait = 120  # 2 minutes max
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            print("Timeout waiting for job to complete")
            return False
        
        status_response = requests.get(f"{BASE_URL}/api/runs/{job_id}/status")
        status = status_response.json()
        
        progress = status["progress_percent"]
        current = status.get("current_query", "")[:50]
        print(f"   [{status['status']}] {progress:.0f}% complete - {current}...")
        
        if status["status"] in ["completed", "failed", "cancelled"]:
            break
        
        time.sleep(2)
    
    # 3. Get results
    print("\n3. Fetching results...")
    if status["status"] != "completed":
        print(f"Job ended with status: {status['status']}")
        return False
    
    results_response = requests.get(f"{BASE_URL}/api/runs/{job_id}/results")
    if results_response.status_code != 200:
        print(f"Error getting results: {results_response.status_code}")
        return False
    
    results = results_response.json()
    summary = results["summary"]
    
    print("\n=== Results Summary ===")
    print(f"Brand: {summary['brand_name']}")
    print(f"Overall Visibility: {summary['overall_visibility']:.1f}%")
    print(f"Average Sentiment: {summary['avg_sentiment']}")
    print(f"Total Queries: {summary['total_queries']}")
    
    if summary.get("provider_visibility"):
        print("\nVisibility by Provider:")
        for prov, vis in summary["provider_visibility"].items():
            print(f"  {prov}: {vis:.1f}%")
    
    if summary.get("competitor_visibility"):
        print("\nCompetitor Visibility:")
        for comp, vis in sorted(summary["competitor_visibility"].items(), key=lambda x: -x[1])[:5]:
            print(f"  {comp}: {vis:.1f}%")
    
    print("\n=== Individual Results ===")
    for r in results["results"]:
        brand_status = "✅" if r["brand_mentioned"] else "❌"
        print(f"\n{brand_status} {r['question'][:60]}...")
        print(f"   Provider: {r['provider']} | Presence: {r['presence']} | Sentiment: {r['sentiment']}")
        if r["other_brands_detected"]:
            print(f"   Other brands: {', '.join(r['other_brands_detected'][:5])}")
    
    return True


def main():
    print("=" * 60)
    print("GEO Tracker API Test")
    print("=" * 60)
    print(f"Target: {BASE_URL}")
    
    # Check if server is running
    try:
        requests.get(f"{BASE_URL}/health", timeout=5)
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API server.")
        print("Make sure the server is running:")
        print("  uvicorn api.main:app --reload --port 8000")
        return
    
    # Run tests
    tests = [
        ("Health Check", test_health),
        ("Query Generation", test_generate_queries),
        ("Full Run Workflow", test_run_workflow),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ {name} failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")


if __name__ == "__main__":
    main()
