import asyncio, random

from vercel.workflow import *
from vercel.workflow import runtime

runtime.workflow_entrypoint()
runtime.step_entrypoint()

@workflow
async def hello_world() -> list[str]:
    async with asyncio.TaskGroup() as tg:
        t1 = tg.create_task(orchestrate(greeting_en, greeting_es))
        t2 = tg.create_task(orchestrate(greeting_es, greeting_en, greeting_es))
        return [await t1, await t2]


async def orchestrate(*greeting_steps) -> str:
    rv = []
    for (i, greeting) in enumerate(greeting_steps):
        rv.append(await greeting(f"workflow{i}"))
    return ", ".join(rv)


@step
async def greeting_en(name: str) -> str:
    await asyncio.sleep(random.random())
    return f"Hello, {name}!"


@step
async def greeting_es(name: str) -> str:
    await asyncio.sleep(random.random())
    return f"¡Hola, {name}!"
