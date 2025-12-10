"""Simple test to check FastAPI setup."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app.main import app
    print("✓ App imported successfully")
    
    from fastapi.testclient import TestClient
    print("✓ TestClient imported successfully")
    
    # Try creating the client
    client = TestClient(app)
    print("✓ TestClient created successfully")
    
    # Test health endpoint
    response = client.get("/health")
    print(f"✓ Health check: {response.status_code} - {response.json()}")
    
    # Test root endpoint
    response = client.get("/")
    print(f"✓ Root endpoint: {response.status_code} - {response.json()}")
    
    # Test graph listing
    response = client.get("/api/v1/graphs")
    print(f"✓ Graph listing: {response.status_code} - Found {len(response.json())} graphs")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()