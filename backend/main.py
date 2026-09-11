from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.prediction import router as prediction_router
from backend.api.news import router as news_router

app = FastAPI(title="AI Supply Chain Intelligence API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router, prefix="/api")
app.include_router(news_router, prefix="/api")

@app.get("/")
def root():
    return {
        "message": "AI Supply Chain Intelligence API",
        "docs": "/docs",
        "model": "final_xgboost_candidate",
    }
