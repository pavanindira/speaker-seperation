#!/usr/bin/env python3
"""
API Test Script - Tests 3-speaker separation capability
"""

import requests
import time

api_url = "http://localhost:8000"

print("="*60)
print("Speaker Separator API - 3-Speaker Test")
print("="*60)

# Submit 3-speaker separation job
print("\n1. Submitting 3-speaker separation job...")
with open("/Users/pavanchilukuri/Desktop/voice-seperator/DIALOGUE.ogg", "rb") as f:
    files = {'file': f}
    data = {
        'num_speakers': '3',
        'method': 'clustering',
        'output_format': 'wav'
    }
    
    response = requests.post(f"{api_url}/api/v1/separate", files=files, data=data)
    job = response.json()
    job_id = job["job_id"]
    print(f"   Job ID: {job_id}")
    print(f"   Speakers: {job['num_speakers']}")
    print(f"   Method: {job['method']}")
    print(f"   Format: {job['output_format']}")

# Wait for completion
print("\n2. Waiting for processing...")
while True:
    status_response = requests.get(f"{api_url}/api/v1/status/{job_id}")
    status = status_response.json()
    
    if status["status"] not in ["pending", "processing"]:
        break
    
    print(f"   Status: {status['status']}")
    time.sleep(3)

# Get results
print("\n3. Job completed! Results:")
results = requests.get(f"{api_url}/api/v1/results/{job_id}").json()
print(f"   Total speakers: {len(results['speakers'])}")
for speaker_name, info in results['speakers'].items():
    print(f"      {speaker_name}: {info['speaking_time']:.2f}s")

print("\n4. Ollama Analysis:")
if results['ollama_analysis']:
    print(f"   {results['ollama_analysis'][:200]}...")
else:
    print("   (No analysis available)")

print("\n" + "="*60)
print("Test completed successfully!")
print("="*60)
