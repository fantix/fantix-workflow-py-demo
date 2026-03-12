from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

from vercel.workflow import start, Run
app = FastAPI(title="Backend API")

# Configure CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()


class RunRequest(BaseModel):
    id: str


class MessageResponse(BaseModel):
    status: str
    result: list[str] | None


@router.get("/")
def root():
    return {"status": "ok", "service": "backend-api"}


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/greeting")
async def get_greeting():
    from workflow import hello_world

    return {"runId": (await start(hello_world)).run_id}


@router.post("/get_greetings", response_model=MessageResponse)
async def echo_message(req: RunRequest):
    run = Run(req.id)
    status = await run.status()
    result = None
    if status == "completed":
        result = await run.return_value()

    return MessageResponse(
        status=status,
        result=result,
    )


@router.get("/items")
def get_items():
    return {
        "items": [
            {"id": 1, "name": "Widget", "price": 9.99},
            {"id": 2, "name": "Gadget", "price": 19.99},
            {"id": 3, "name": "Gizmo", "price": 29.99},
        ]
    }


app.include_router(router)
