content = open('/home/runpod-setins/llm_engine.py').read()
content = content.replace('llama3.1:8b', 'llama3.2:3b')
open('/home/runpod-setins/llm_engine.py', 'w').write(content)
