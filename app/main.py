from fastapi import FastAPI
from app.line_webhook import router as line_router

app = FastAPI()

app.include_router(line_router)
