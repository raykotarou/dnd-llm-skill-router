import httpx
import asyncio

async def test():
    async with httpx.AsyncClient() as client:
        try:
            # Test streaming
            async with client.stream('POST', 'http://127.0.0.1:8002/v1/chat/completions', json={'model': 'main', 'messages': [{'role': 'user', 'content': 'Hello'}], 'stream': True}) as response:
                print('Stream Status:', response.status_code)
                async for line in response.aiter_lines():
                    if line.strip():
                        print('Line:', line[:200])
        except Exception as e:
            print('Error:', e)

asyncio.run(test())