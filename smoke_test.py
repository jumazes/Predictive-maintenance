from fastapi.testclient import TestClient
from app import app

client = TestClient(app)
assert client.get('/health').status_code == 200
response = client.post('/predict', json={
    'type':'L',
    'air_temperature_k':300.0,
    'process_temperature_k':307.8,
    'rotational_speed_rpm':1320,
    'torque_nm':58.0,
    'tool_wear_min':190
})
assert response.status_code == 200
payload = response.json()
assert 'risk_score' in payload and 'status' in payload
print(payload)
