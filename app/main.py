import os
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api import health, openai_proxy
from app.config.settings import AppSettings, load_settings
from app.logging.setup import setup_logging


def create_app(settings: AppSettings | None = None) -> FastAPI:
    app_settings = settings or load_settings(os.getenv("CONFIG_PATH", "./config.yaml"))
    setup_logging(app_settings.logging)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = app_settings
        logger.info(
            "Starting dnd-skill-router on {}:{}",
            app_settings.server.host,
            app_settings.server.port,
        )
        yield
        logger.info("Stopping dnd-skill-router")

    application = FastAPI(
        title="dnd-skill-router",
        version="0.1.0",
        lifespan=lifespan,
    )

    if app_settings.server.enable_cors:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    application.include_router(health.router)
    application.include_router(openai_proxy.router)
    return application


app = create_app()
