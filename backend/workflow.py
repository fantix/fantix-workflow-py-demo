import asyncio, dataclasses, random

from vercel.workflow import *
from vercel.workflow import runtime

runtime.workflow_entrypoint()
runtime.step_entrypoint()


@dataclasses.dataclass
class DraftRequest(HookMixin):
    prompt: str | None


@workflow
async def multi_drafter(token: str) -> list[str]:
    try:
        result = []
        hook = DraftRequest.wait(token=token)
        async with asyncio.TaskGroup() as tg:
            tg.create_task(thinking_drafter(hook, result))
            tg.create_task(fast_drafter(hook, result))
        return result
    except Exception:
        import traceback
        traceback.print_exc()
        raise


async def thinking_drafter(hook: HookEvent[DraftRequest], result: list[str]) -> None:
    async for request in hook:
        if request.prompt is None:
            hook.dispose()
            break

        await asyncio.sleep(random.random() * 2)
        result.append(f"Thinking: {request.prompt}")


async def fast_drafter(hook: HookEvent[DraftRequest], result: list[str]) -> None:
    async for request in hook:
        if request.prompt is None:
            hook.dispose()
            break

        await asyncio.sleep(random.random())
        result.append(f"Fast: {request.prompt}")
