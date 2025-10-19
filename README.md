# PDF to Anki Definition Extractor

FastAPI service that accepts textbook PDFs, extracts term/definition pairs through OpenAI's ChatGPT API, and returns both JSON and Anki-compatible CSV output ready for flashcard import.

## Local Setup

1. Clone the repository and open the project directory.
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the environment example:
   ```bash
   cp .env.example .env
   ```
5. Edit `.env` and set **all** required variables (see below).

## Environment Variables

- `OPENAI_API_KEY` (required): Secret key used to access OpenAI's API.
- `SYSTEM_PROMPT` (required): System prompt text for the definition extractor.
- `USER_PROMPT_TEMPLATE` (required): User prompt template; must contain `{page_text}` placeholder.
- `ALLOWED_ORIGINS` (optional): Comma-separated list of CORS origins for production deployments.

## Running Locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Upload a PDF to `POST /api/process-pdf` for JSON + CSV text, or `POST /api/download-anki` to download the CSV directly.

## Deployment (Render Example)

1. Push the code to GitHub.
2. Create a new **Web Service** on Render.
3. Connect the GitHub repository.
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add the `OPENAI_API_KEY`, `SYSTEM_PROMPT`, and `USER_PROMPT_TEMPLATE` environment variables (plus `ALLOWED_ORIGINS` if needed).
7. Deploy.

Render will provide the `PORT` environment variable automatically; the application reads it at runtime.
