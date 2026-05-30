# diagnose_ai.py
import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

print("=" * 50)
print("STEP 1: ENV CHECK")
print("=" * 50)
key = os.getenv("GOOGLE_API_KEY", "")
print(f"GOOGLE_API_KEY set: {bool(key)}")
print(f"GOOGLE_API_KEY value: '{key[:8]}...' " if len(key) > 8 else f"GOOGLE_API_KEY value: '{key}' ← EMPTY OR MISSING!")
print(f"GEMINI_MODEL: {os.getenv('GEMINI_MODEL', 'NOT SET — will default to gemini-3.5-flash')}")

print()
print("=" * 50)
print("STEP 2: PACKAGE CHECK")
print("=" * 50)
try:
    import google.generativeai as genai
    print("✅ google-generativeai (old) — INSTALLED")
    pkg = "old"
except ImportError:
    print("❌ google-generativeai (old) — NOT INSTALLED")
    pkg = None

try:
    import google.genai as genai
    print("✅ google-genai (new) — INSTALLED")
    pkg = "new"
except ImportError:
    print("❌ google-genai (new) — NOT INSTALLED")

if not pkg:
    print()
    print("👉 FIX: run:  pip install google-generativeai")


print()
print("=" * 50)
print("STEP 4: LIVE GEMINI TEST")
print("=" * 50)
if key and pkg:
    try:
        if pkg == "old":
            import google.generativeai as genai
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-3.5-flash")
        else:
            import google.genai as genai
            model = genai.Client(api_key=key)

        async def test_gemini():
            if pkg == "old":
                resp = await asyncio.to_thread(model.generate_content, "Rate at a 1-10 scale which is the best james bond movie of the craig's era.")
                return resp.text.strip()
            else:
                resp = await asyncio.to_thread(
                    model.models.generate_content,
                    model="gemini-3.5-flash",
                    contents="Rate at a 1-10 scale which is the best james bond movie of the craig's era."
                )
                return resp.text.strip()

        result = asyncio.run(test_gemini())
        print(f"✅ Gemini works! Response: {result}")
    except Exception as e:
        print(f"❌ Gemini failed: {e}")
        print("👉 Check your API key is valid and billing/free tier is active")
else:
    print("⏭️  Skipped — key or package missing (fix steps above first)")