import fastapi
from fastapi.middleware.cors import CORSMiddleware
from vercel.workflow.fastapi import with_workflow
from vercel.headers import set_headers

app = with_workflow(fastapi.FastAPI())

@app.middleware("http")
async def set_vercel_headers(request: fastapi.Request, call_next):
    set_headers(request.headers)
    return await call_next(request)

# Configure CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = fastapi.APIRouter(prefix="/_/backend")



from vercel.workflow import *

@workflow
async def hello_world() -> None:
    print(await greeting("workflow"))


@step
async def greeting(name: str) -> str:
    return f"Hello, {name}!"


@router.get("/hello")
async def root():
    from vercel.workflow.runtime import start
    run = await start(hello_world)
    return run.run_id

app.include_router(router)
