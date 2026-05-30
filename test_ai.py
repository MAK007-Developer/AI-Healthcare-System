# test_ai.py
import asyncio
from backend.core_ai import generate, is_available

async def test():
    available = await is_available()
    print(f"AI available: {available}")
    result = await generate("Say hello in one sentence.")
    print(f"Response: {result}")

asyncio.run(test())