# Mili — Genesis v0.1

> **Mili** is a production-grade Agentic AI personal assistant.
> Genesis v0.1 establishes the clean architectural foundation on which all future versions build.

---

## Architecture

```
HTTP Request
    │
    ▼
[RequestID Middleware]          ← generates UUID per request; propagates to logs + response header
    │
    ▼
[FastAPI Route]                 ← validates schema; delegates to ChatService via DI
    │
    ▼
[ChatService]                   ← orchestration layer; provider-agnostic
    │
    ▼
[LLMProvider Protocol]          ← structural interface; no SDK imports
    │
    ▼
[GeminiProvider]                ← only file that imports google-genai
    │
    ▼
[google-genai SDK]
```

### Key design invariants

| Rule | Where enforced |
|---|---|
| `from google import genai` appears only in `app/llm/providers/gemini.py` | Code review + tests |
| `app/api/chat.py` imports zero symbols from `app/llm/` | Code review |
| `ChatService` depends on `LLMProvider` Protocol, not `GeminiProvider` | DI + tests |
| No secrets in source code | `pydantic-settings` + `.env` |
| Factory runs once at startup | `@lru_cache` in `dependencies.py` |

---

## Project structure

```
ProjectMILI/
├── app/
│   ├── main.py                  # FastAPI app factory
│   ├── config.py                # Pydantic Settings (env management)
│   ├── logging_config.py        # Structured logging + request-ID context
│   ├── dependencies.py          # Composition root (DI wiring)
│   ├── middleware/
│   │   └── request_id.py        # RequestID middleware
│   ├── api/
│   │   ├── router.py            # Top-level router
│   │   ├── health.py            # GET /api/health
│   │   └── chat.py              # POST /api/chat
│   ├── schemas/
│   │   ├── chat.py              # ChatRequest / ChatResponse
│   │   └── health.py            # HealthResponse
│   ├── services/
│   │   └── chat_service.py      # ChatService — orchestration layer
│   └── llm/
│       ├── base.py              # LLMProvider Protocol + Message dataclass
│       ├── exceptions.py        # LLMError / LLMProviderError / LLMConfigurationError
│       ├── factory.py           # Startup-only provider factory
│       └── providers/
│           └── gemini.py        # Google Gemini implementation (google-genai SDK)
├── tests/
│   ├── conftest.py
│   ├── test_health.py
│   ├── test_chat.py
│   ├── test_chat_service.py     # Architecture-verifying service tests
│   ├── test_llm_abstraction.py  # Architecture-verifying LLM layer tests
│   └── test_request_id.py
├── .env.example
├── .gitignore
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Setup

### 1. Clone and enter the project

```bash
git clone <repo-url>
cd ProjectMILI
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

Get a free Gemini API key at: https://aistudio.google.com/app/apikey

### 5. Run the server

```bash
uvicorn app.main:app --reload
```

The API is now available at `http://localhost:8000`.

- API docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## API Reference

### `GET /api/health`

Returns application status and version.

```bash
curl http://localhost:8000/api/health
```

```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

---

### `POST /api/chat`

Send a conversation and receive the assistant's reply.

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello Mili, who are you?"}
    ]
  }'
```

**Request body**

```json
{
  "messages": [
    {"role": "system", "content": "You are Mili, an AI assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "What can you do?"}
  ]
}
```

| Field | Type | Required |
|---|---|---|
| `messages` | array of `{role, content}` | Yes (min 1) |
| `role` | `"user"` \| `"assistant"` \| `"system"` | Yes |
| `content` | string (min 1 char) | Yes |

**Response body**

```json
{
  "reply": "I'm Mili, your AI assistant!",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

The `X-Request-ID` response header carries the same value as `request_id` in the body.
You can supply your own `X-Request-ID` request header to propagate trace IDs from upstream systems.

---

## Running tests

```bash
pytest tests/ -v
```

The test suite verifies:
- Health and chat endpoint behaviour
- Input validation (422 responses)
- Error mapping (LLMProviderError → HTTP 502)
- Request ID generation, preservation, and propagation
- ChatService isolation (works without any real SDK)
- LLMProvider Protocol structural compliance
- Factory fail-fast on unknown provider

---

## Adding a new LLM provider

1. Create `app/llm/providers/<name>.py` with a class that has:
   ```python
   async def chat(self, messages: list[Message]) -> str: ...
   ```
2. Add a `case "<name>":` branch in `app/llm/factory.py`.
3. Add the new provider's API key field to `app/config.py` and `.env.example`.
4. Set `LLM_PROVIDER=<name>` in your `.env`.

No other files need to change.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | Provider to use (`gemini` supported) |
| `GEMINI_API_KEY` | *(required)* | Google AI Studio API key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `APP_ENV` | `development` | `development` / `staging` / `production` |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `APP_PORT` | `8000` | Port (used if running via script) |

---

## What's next (future milestones)

| Milestone | Planned feature |
|---|---|
| v0.2 | Conversation memory (MongoDB) |
| v0.3 | Vector Search / RAG |
| v0.4 | Agentic tool calling |
| v0.5 | Voice / STT / TTS |
| v0.6 | Multi-provider support (OpenAI, Claude, Local) |
| v1.0 | Authentication, production deployment |
