from fastapi import FastAPI, HTTPException
import requests
from pydantic import BaseModel

app = FastAPI()

OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Ollama API running locally

# Define the request body model
class ChatRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "Ollama API Server is Running"}

@app.post("/chat")
def chat(request: ChatRequest):
    payload = {
        "model": "deepseek-r1",
        "prompt": request.query,
        "stream": False,
        "temperature": 0.9,
        "top_k": 100000
    }
    
    response = requests.post(OLLAMA_API_URL, json=payload)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error communicating with Ollama API")
    
    return response.json()

@app.post("/install_model")
def install_model(model: str):
    payload = {"name": model}
    response = requests.post(f"{OLLAMA_API_URL}/pull", json=payload)
    
    if response.status_code == 200:
        return {"message": f"Model '{model}' installation started."}
    else:
        raise HTTPException(status_code=response.status_code, detail=response.json())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
