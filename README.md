# Multi-Drafter Workflow Demo

A demo project showcasing [Vercel Workflow](https://vercel.com/docs/workflow) with a Next.js frontend and FastAPI backend. The workflow runs two concurrent "drafters" (thinking and fast) that process prompts in parallel, demonstrating hooks, concurrent tasks, and long-running workflow orchestration.

## Project Structure

```
fantix-workflow-py-demo/
├── vercel.json              # Vercel Fluid config (services + workflow worker)
├── frontend/                # Next.js frontend
│   └── src/app/
│       ├── layout.tsx       # Root layout
│       ├── page.tsx         # Workflow interaction UI
│       └── globals.css      # Styles
├── backend/                 # FastAPI backend
│   ├── main.py              # API endpoints
│   ├── workflow.py          # Workflow definition (multi-drafter)
│   └── requirements.txt     # Python dependencies
```

## How It Works

`backend/workflow.py` defines a `multi_drafter` workflow that:

1. Waits for `DraftRequest` hooks (user prompts sent via the API)
2. Dispatches each prompt to two concurrent drafters using `asyncio.TaskGroup`:
   - **Thinking drafter** -- simulates slower, deeper processing (up to 2s)
   - **Fast drafter** -- simulates quicker responses (up to 1s)
3. Collects results from both drafters into a list
4. Completes when a `None` prompt is received as the termination signal

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/` | GET | Health check |
| `/api/start_workflow?token=` | GET | Start the multi-drafter workflow with a unique token |
| `/api/draft_request?token=&prompt=` | GET | Send a prompt to both drafters |
| `/api/finish_workflow?token=` | GET | Signal drafters to stop (sends `None` prompt) |
| `/api/get_result` | POST | Get workflow status and results by `runId` |

## Usage Flow

1. **Start** -- provide a unique token to create a workflow run (returns a `runId`)
2. **Send prompts** -- send one or more draft requests; both drafters process each prompt concurrently
3. **Finish** -- send the finish signal to stop the drafters
4. **Get result** -- poll with the `runId` until status is `completed`, then read the collected results

## Local Development

```
vercel dev
```

This starts both the frontend and backend. Open http://localhost:3000 in your browser.
