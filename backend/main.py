from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

import models, database
from routers import auth, inference, chat

# Create tables
database.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="AI Medical Imaging System API")

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for demo purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files for uploaded images and heatmaps
os.makedirs("uploads/images", exist_ok=True)
os.makedirs("uploads/heatmaps", exist_ok=True)
app.mount("/static/images", StaticFiles(directory="uploads/images"), name="images")
app.mount("/static/heatmaps", StaticFiles(directory="uploads/heatmaps"), name="heatmaps")

app.include_router(auth.router)
app.include_router(inference.router)
app.include_router(chat.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Medical Imaging System API"}
