# OrchestraAI Backend

Production-ready FastAPI backend for the OrchestraAI Manager-Worker RL Environment.

## Architecture

```
orchestra_backend/
├── main.py                    ← FastAPI entry point
├── config.py                  ← Configuration & environment variables
├── requirements.txt           ← Python dependencies
│
├── api/
│   ├── routes/
│   │   ├── episode.py         ← /episode endpoints
│   │   ├── metrics.py         ← /metrics endpoints
│   │   ├── model.py           ← /model endpoints
│   │   └── websocket.py       ← /ws endpoints
│   └── middleware/
│       ├── auth.py            ← JWT authentication
│       └── rate_limit.py      ← Rate limiting
│
├── services/
│   ├── episode_service.py     ← Episode business logic
│   ├── metrics_service.py     ← Metrics tracking
│   ├── budget_service.py      ← Token budget management
│   └── quality_service.py     ← Hallucination detection
│
├── models/
│   ├── episode.py             ← Episode Pydantic models
│   ├── metrics.py             ← Metrics Pydantic models
│   └── common.py              ← Common response models
│
└── db/
    ├── mongodb.py             ← MongoDB connection
    └── repositories/
        ├── episode_repo.py    ← Episode data access
        └── metrics_repo.py    ← Metrics data access
```

## Setup

### 1. Install Dependencies

```bash
cd manager-worker-env
pip install -e ".[backend]"
```

### 2. Configure Environment

Create `.env` file in `orchestra_backend/` (or omit `MONGODB_URL` to run with in-process episode state only).

```env
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=openenv-project
DEBUG=True
SERVER_PORT=8000
```

### 3. Start Server

```bash
cd manager-worker-env
python -m orchestra_backend.main
# or: uvicorn orchestra_backend.main:app --host 0.0.0.0 --port 8000
```

Server will be available at `http://localhost:8000`

## API Endpoints

### Health & Status
- `GET /health` - Health check
- `GET /status` - Server status
- `GET /` - Root endpoint

### Episodes
- `POST /episode/start` - Start new episode
- `GET /episode/{episode_id}` - Get episode state
- `POST /episode/{episode_id}/step` - Execute action
- `POST /episode/{episode_id}/end` - End episode
- `GET /episode` - List all episodes
- `GET /episode/{episode_id}/history` - Get episode history

### Metrics
- `GET /training/metrics` - Current metrics
- `GET /training/metrics/history` - Metrics history

### Model
- `GET /model/info` - Model information
- `POST /model/checkpoint` - Save checkpoint

### WebSocket
- `WS /ws/live` - Real-time updates

## Testing Endpoints

### Start Episode
```bash
curl -X POST http://localhost:8000/episode/start \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task_1",
    "difficulty": 2,
    "num_workers": 4,
    "budget": 1000.0
  }'
```

### List Episodes
```bash
curl http://localhost:8000/episode
```

### Get Metrics
```bash
curl http://localhost:8000/training/metrics
```

## Database

MongoDB collections:
- `episodes` - Episode data
- `metrics` - Training metrics
- `training_sessions` - Training session records

## Services

### EpisodeService
Manages episode lifecycle:
- Start episodes
- Execute steps
- End episodes
- Track history

### MetricsService
Tracks training metrics:
- Record episode metrics
- Calculate statistics
- Store in MongoDB

### BudgetService
Manages token budgets:
- Allocate budgets
- Track token usage
- Enforce limits

### QualityService
Detects hallucinations:
- Pattern matching
- Quality assessment
- Hallucination detection

## Development

### Code Style
- Black for formatting
- isort for imports
- mypy for type checking

### Testing
```bash
pytest tests/ -v --cov=orchestra_backend
```

### Documentation
API docs available at `http://localhost:8000/docs`

## Production Deployment

1. Set `DEBUG=False` in config
2. Use production ASGI server (Gunicorn + Uvicorn)
3. Configure proper MongoDB credentials
4. Set up SSL/TLS
5. Enable rate limiting
6. Configure CORS properly

## Troubleshooting

### MongoDB Connection Failed
- Check `MONGODB_URL` in config
- Verify network connectivity
- Check MongoDB credentials

### Port Already in Use
- Change `SERVER_PORT` in config
- Or kill process: `lsof -ti:8000 | xargs kill -9`

### Import Errors
- Ensure you're in `manager-worker-env/` (package root) so `env` and `orchestra_backend` import correctly
- Check Python path setup
- Verify all dependencies installed
