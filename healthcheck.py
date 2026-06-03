#!/usr/bin/env python3
import sys
import requests

def check_health():
    try:
        response = requests.get("http://127.0.0.1:8000/", timeout=5)
        if response.status_code == 200 and response.json().get("status") == "running":
            print("Health check passed.")
            sys.exit(0)
        print(f"Health check failed: {response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"Health check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_health()
