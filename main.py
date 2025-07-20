
from fastapi import FastAPI
from AI_services.API.routes import router
import sys
import os

# path for modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# create fastapi app and router
AI_services = FastAPI()
AI_services.include_router(router)
