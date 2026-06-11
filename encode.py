import base64

with open("connection_registry.py", "rb") as f:
    data = f.read()
with open("connection_registry.b64", "wb") as f:
    f.write(base64.b64encode(data))

with open("websocket_server.py", "rb") as f:
    data = f.read()
with open("websocket_server.b64", "wb") as f:
    f.write(base64.b64encode(data))
