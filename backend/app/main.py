from fastapi import FastAPI

from fastapi.middleware.cors import (
    CORSMiddleware
)

from fastapi.staticfiles import (
    StaticFiles
)

from app.routes.analyze import (
    router as analyze_router
)

import os


# =========================
# CREATE FASTAPI APP
# =========================

app = FastAPI(

    title="AI Outfit Evaluator API",

    description=
    "Advanced AI Fashion Intelligence API",

    version="2.0.0"
)


# =========================
# CREATE OUTPUT FOLDER
# =========================

os.makedirs(
    "outputs",
    exist_ok=True
)


# =========================
# STATIC FILES
# =========================

app.mount(

    "/outputs",

    StaticFiles(directory="outputs"),

    name="outputs"
)


# =========================
# CORS CONFIGURATION
# =========================

app.add_middleware(

    CORSMiddleware,

    allow_origins=[

        "http://localhost:5173",

        "http://127.0.0.1:5173",
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# =========================
# ROUTES
# =========================

app.include_router(
    analyze_router
)


# =========================
# HOME ROUTE
# =========================

@app.get("/")
def home():

    return {

        "message":
            "AI Outfit Evaluator API Running",

        "status":
            "online",

        "version":
            "2.0.0"
    }