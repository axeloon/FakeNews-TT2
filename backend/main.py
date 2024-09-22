from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from .routes import noticia_routes, x_routes, emol_routes
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API Backend FakeNewsTT2",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Aplicar el prefijo "/api" a todos los routers
app.include_router(x_routes.router, prefix="/api")
app.include_router(noticia_routes.router, prefix="/api")
app.include_router(emol_routes.router, prefix="/api")