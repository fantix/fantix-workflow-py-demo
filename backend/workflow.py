import asyncio, dataclasses, random

from vercel import workflow

wf = workflow.Workflows()


@dataclasses.dataclass
class DraftRequest(workflow.BaseHook):
    prompt: str | None


@wf.workflow
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


async def thinking_drafter(hook: workflow.HookEvent[DraftRequest], result: list[str]) -> None:
    async for request in hook:
        if request.prompt is None:
            hook.dispose()
            break

        await workflow.sleep(random.random() * 2000)
        result.append(f"Thinking: {request.prompt}")


async def fast_drafter(hook: workflow.HookEvent[DraftRequest], result: list[str]) -> None:
    async for request in hook:
        if request.prompt is None:
            hook.dispose()
            break

        await workflow.sleep(random.random() * 1000)
        result.append(f"Fast: {request.prompt}")
