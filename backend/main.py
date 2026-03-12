from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

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


class Message(BaseModel):
    text: str


class MessageResponse(BaseModel):
    message: str
    timestamp: str
    reversed: str


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
    from vercel.workflow.runtime import start

    await start(hello_world)

    return {
        "message": "Hello from FastAPI backend!",
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/echo", response_model=MessageResponse)
def echo_message(message: Message):
    return MessageResponse(
        message=message.text,
        timestamp=datetime.now().isoformat(),
        reversed=message.text[::-1],
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
