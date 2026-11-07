# Personal Finance Recommendation System — Backend API & AI Chatbot

Backend service implementation for **Member 4: API Integration and AI Chatbot**.

This repository contains the main FastAPI application, system routing and CORS middleware, the AI personal finance chatbot service (prepared for Gemma 4 E2B/E4B with QLoRA fine-tuning), and pluggable integration endpoints for Members 1, 2, and 3.

---

## 1. Project Structure

```text
app/
├── __init__.py
├── main.py                  # FastAPI application entrypoint, CORS, exception handlers, health check
├── api/
│   ├── __init__.py
│   ├── chatbot.py           # POST /chat endpoint (Member 4)
│   └── integrations.py      # Pluggable endpoints for Members 1, 2, & 3
├── services/
│   ├── __init__.py
│   ├── llm_service.py       # Gemma 4 E2B/E4B lazy loading, QLoRA adapter handling & RAG prompt builder
│   └── context_service.py   # Financial context bridge preventing number hallucinations
└── schemas/
    ├── __init__.py
    └── chatbot.py           # Pydantic schemas: ChatRequest, ChatResponse, FinancialContext

tests/
├── __init__.py
├── test_main.py             # App startup, /health, Swagger, OpenAPI schema validation
├── test_chatbot.py          # Grounding, empty message validation, disabled mode, model error handling
└── test_integrations.py     # Team integration endpoints and pluggable handler tests

.env.example                 # Template for environment configuration
requirements.txt             # Project dependencies
```

---

## 2. API Endpoints

| Method | Endpoint | Module | Description | Status |
|---|---|---|---|---|
| `GET` | `/health` | System | Health check verification | **Implemented** |
| `POST` | `/chat` | Member 4 | AI personal finance assistant grounded in real data | **Implemented** |
| `POST` | `/upload-statement` | Member 1 | Bank statement upload (PDF, CSV, Excel) | *Integration interface ready* |
| `GET` | `/transactions` | Member 2 | Processed transaction history from SQLite | *Integration interface ready* |
| `GET` | `/analytics/summary` | Member 3 | Income, expense, and net cash flow summary | *Integration interface ready* |
| `GET` | `/analytics/categories` | Member 3 | Category-wise spending breakdown | *Integration interface ready* |
| `GET` | `/recommendations/budget` | Member 3 | Personalized budget suggestions | *Integration interface ready* |
| `GET` | `/recommendations/savings` | Member 3 | Actionable savings opportunities | *Integration interface ready* |
| `GET` | `/detection/overspending` | Member 3 | Overspending category detection | *Integration interface ready* |
| `GET` | `/detection/anomalies` | Member 3 | Suspicious transaction anomaly detection | *Integration interface ready* |

---

## 3. Installation & Setup

### Prerequisites
* Python 3.11+
* `uv` or standard Python `venv`

### Step 1: Clone and Set Up Virtual Environment

```bash
# Using standard venv
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate
# Or activate on Linux/macOS:
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

Create `.env` from the template:

```bash
cp .env.example .env
```

Default settings in `.env`:
```env
APP_NAME="Personal Finance Recommendation System"
ENVIRONMENT=development
CORS_ORIGINS="http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000"

# LLM Configuration
LLM_ENABLED=false
LLM_MODEL_PATH=
LLM_ADAPTER_PATH=
LLM_DEVICE=cpu
```

### Step 4: Run the Backend Server

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Access the interactive API documentation at:
* Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 4. Chatbot & Gemma Model Integration

The chatbot service in `app/services/llm_service.py` is designed to be fully modular and decoupled from the API routes:

1. **Singleton Lazy Loading**: The model is initialized only once upon first demand and reused across requests, avoiding expensive reload delays.
2. **Safe Fallback**: If `LLM_ENABLED=false` or no local weights are found at `LLM_MODEL_PATH`, the service runs gracefully in grounded rule-based mode without crashing.
3. **QLoRA Fine-Tuning**: Set `LLM_ADAPTER_PATH` to point to the PEFT/LoRA adapter directory once fine-tuning on Gemma 4 E2B or E4B is complete.
4. **Anti-Hallucination Guardrails**:
   * If a user asks for personal financial metrics but hasn't uploaded data, the assistant responds:
     `"Your financial data has not yet been loaded. Please upload a bank statement first..."` with `source: "system"`.
   * When data is available, exact numbers are injected into the context from backend services rather than being guessed by the LLM.

### Example Request (`POST /chat`):

```json
{
  "message": "How much did I spend this month?",
  "user_id": "demo_user"
}
```

### Example Response:

```json
{
  "response": "Your financial data has not yet been loaded. Please upload a bank statement first so I can analyze your transactions.",
  "source": "system"
}
```

---

## 5. Integration Contracts for Other Team Members

Members 1, 2, and 3 can attach their business logic functions directly to `app/api/integrations.registry` without modifying existing route declarations or schemas:

```python
from app.api.integrations import registry

# Member 1 attaches their parser function:
registry.member1_upload_handler = parse_and_save_statement

# Member 2 attaches their transaction query function:
registry.member2_transactions_handler = get_user_transactions

# Member 3 attaches their analytics and recommendation functions:
registry.member3_summary_handler = calculate_financial_summary
registry.member3_categories_handler = calculate_category_breakdown
registry.member3_budget_handler = generate_budget_recommendations
registry.member3_savings_handler = generate_savings_recommendations
registry.member3_overspending_handler = detect_overspending_categories
registry.member3_anomalies_handler = detect_transaction_anomalies
```

Until attached, endpoints return HTTP 501 with a descriptive message indicating the responsible member module.

---

## 6. Running Tests

Execute the automated test suite with pytest:

```bash
pytest -v
```

All 14 test cases cover application startup, health check, OpenAPI documentation, chatbot validation, context grounding, fallback error handling, and integration interfaces.

