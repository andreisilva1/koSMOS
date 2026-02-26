
from fastapi import FastAPI

app = FastAPI()

@app.get("/hello_world")
async def hello_world():
    return {"message": "Hello, World!"}