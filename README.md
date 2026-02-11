# Next.js Frontend + FastAPI Backend

A demo project showcasing frontend-backend communication using Next.js and FastAPI.

## Project Structure

```
next-frontend-backend-zc/
├── backend/              # FastAPI backend
│   ├── main.py          # API endpoints
│   └── requirements.txt # Python dependencies
├── src/app/             # Next.js frontend (App Router)
│   ├── layout.tsx       # Root layout
│   ├── page.tsx         # Main page with API calls
│   └── globals.css      # Styles
├── package.json         # Node.js dependencies
└── next.config.js       # Next.js configuration
```

## Setup

### Backend (FastAPI)

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload --port 8000
```

The backend will be available at http://localhost:8000

### Frontend (Next.js)

```bash
# From the project root directory

# Install dependencies
npm install

# Create .env.local file
echo "NEXT_PUBLIC_BACKEND_URL=http://localhost:8000" > .env.local

# Run development server
npm run dev
```

The frontend will be available at http://localhost:3000

## Environment Variables

The frontend uses `NEXT_PUBLIC_BACKEND_URL` to communicate with the backend:

- Development: `http://localhost:8000`
- Production: Set this to your deployed backend URL

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/greeting` | GET | Get a greeting message |
| `/api/echo` | POST | Echo back a message (with reversal) |
| `/api/items` | GET | Get list of items |

## Running Both Services

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
npm run dev
```

Then open http://localhost:3000 in your browser.
