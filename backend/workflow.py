import asyncio, random

from vercel.workflow import *
from vercel.workflow import runtime

runtime.workflow_entrypoint()
runtime.step_entrypoint()

@workflow
async def hello_world() -> None:
    async with asyncio.TaskGroup() as tg:
        tg.create_task(orchestrate(greeting_en, greeting_es))
        tg.create_task(orchestrate(greeting_es, greeting_en, greeting_es))


async def orchestrate(*greeting_steps):
    for (i, greeting) in enumerate(greeting_steps):
        print(await greeting(f"workflow{i}"))


@step
async def greeting_en(name: str) -> str:
    await asyncio.sleep(random.random())
    return f"Hello, {name}!"


@step
async def greeting_es(name: str) -> str:
    await asyncio.sleep(random.random())
    return f"¡Hola, {name}!"
