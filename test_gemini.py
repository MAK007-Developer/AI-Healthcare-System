# test_gemini.py
import os
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("GOOGLE_API_KEY", "")
print(f"Key loaded: {'YES, length=' + str(len(key)) if key else 'NO - empty or missing'}")
print(f"Key starts with: {key[:10]}..." if key else "")

# Test import
try:
    import google.generativeai as genai
    print("Import: OK (google-generativeai)")
except ImportError:
    print("Import FAILED - run: pip install google-generativeai")
    exit()

# Test auth
try:
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content("Say hello.")
    print(f"Gemini response: {response.text}")
except Exception as e:
    print(f"Gemini error: {e}")