
import os
import requests
from dotenv import load_dotenv

load_dotenv()

raw_keys = os.getenv("GROQ_API_KEYS", "")

keys = [
    key.strip().strip('"').strip("'")
    for key in raw_keys.split(",")
    if key.strip()
]

if not keys:
    print("No Groq API keys found in .env")
    exit(1)

url = "https://api.groq.com/openai/v1/models"

print(f"\nChecking {len(keys)} Groq API keys...\n")

for i, key in enumerate(keys, 1):
    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=15
        )

        if response.status_code == 200:
            status = "VALID"
        elif response.status_code == 401:
            status = "INVALID / UNAUTHORIZED"
        elif response.status_code == 403:
            status = "FORBIDDEN"
        elif response.status_code == 429:
            status = "RATE LIMITED"
        else:
            status = f"HTTP {response.status_code}"

        # Mask API key
        masked = key[:7] + "..." + key[-4:] if len(key) > 11 else "***"

        print(f"[{i}] {masked} -> {status}")

    except requests.RequestException as e:
        print(f"[{i}] CONNECTION ERROR: {e}")

print("\nDone!")
