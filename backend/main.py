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


@router.get("/start_workflow")
async def start_workflow(token: str):
    from workflow import multi_drafter

    run = await start(multi_drafter, token)
    return {"runId": run.run_id}


@router.get("/draft_request")
async def draft_request(prompt: str, token: str):
    from workflow import DraftRequest

    await DraftRequest(prompt).resume(token)


@router.get("/finish_workflow")
async def finish_workflow(token: str):
    from workflow import DraftRequest

    await DraftRequest(None).resume(token)


@router.post("/get_result", response_model=MessageResponse)
async def get_result(req: RunRequest):
    run = Run(req.id)
    status = await run.status()
    result = None
    if status == "completed":
        result = await run.return_value()

    return MessageResponse(
        status=status,
        result=result,
    )


app.include_router(router)
