from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "hello"}

@app.post("/chat")
def chat(payload: dict):
    user_message = payload.get("message", "")
    return {"reply": f"你说的是：{user_message}"}