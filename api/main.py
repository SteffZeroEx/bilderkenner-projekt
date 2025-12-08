# -*- coding: utf-8 -*-
"""
FastAPI Hauptanwendung
"""

from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Bilderkenner API",
    description="REST API fuer CIFAR-10 Bildklassifikation",
    version="1.0.0"
)

app.include_router(router)


@app.get("/")
async def root():
    """Wurzel-Endpoint"""
    return {
        "message": "Bilderkenner API",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
