# main.py - Entry point for FastAPI application

from fastapi import FastAPI
from app.routes import predict, health

# Initialize FastAPI app
app = FastAPI(title="LLaVA Medical Image Analysis API", version="1.0")

# Include API routes
app.include_router(predict.router, prefix="/api")
app.include_router(health.router, prefix="/api")

@app.get("/")
def home():
    return {"message": "Welcome to the LLaVA Medical Image Analysis API"}