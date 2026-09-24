"""Aggregates all API routers so app.main can include them in one place."""

from app.routes import analysis, chat, disease, health

routers = [
    health.router,
    chat.router,
    analysis.router,
    disease.router,
]
