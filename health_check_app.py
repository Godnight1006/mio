from fastapi import FastAPI
import uvicorn
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "ok"}

if __name__ == "__main__":
    # Hugging Face Spaces expect the app to listen on port 7860
    port = int(os.environ.get("PORT", 7860)) # Reverted default to 7860
    print(f"Attempting to start Uvicorn on host 0.0.0.0 and port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
