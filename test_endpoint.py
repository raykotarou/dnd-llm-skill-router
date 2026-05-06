import requests
import json

# Test non-streaming
try:
    response = requests.post('http://127.0.0.1:8000/v1/chat/completions', json={
        'model': 'dnd-skill-router',
        'stream': False,
        'messages': [{'role': 'user', 'content': 'Hello'}]
    })
    print('Non-streaming response:', response.status_code, response.text[:200])
except Exception as e:
    print('Error:', e)

# Test streaming
try:
    response = requests.post('http://127.0.0.1:8000/v1/chat/completions', json={
        'model': 'dnd-skill-router',
        'stream': True,
        'messages': [{'role': 'user', 'content': 'Hello'}]
    }, stream=True)
    print('Streaming response:', response.status_code)
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                print(line.decode('utf-8')[:100])
            break  # Just first line
except Exception as e:
    print('Streaming error:', e)