
from fastapi import FastAPI
from AI_services.API.routes import router

AI_services = FastAPI()
AI_services.include_router(router)
