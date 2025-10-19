import csv
import json
import logging
import os
import tempfile
import time
from collections import OrderedDict
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
CHUNK_SIZE = 1 * 1024 * 1024  # 1MB

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

ANKI_IMPORT_INSTRUCTIONS = (
    "To import into Anki: 1. Open Anki 2. Click 'Import File' 3. Select this CSV "
    "4. Set Field separator to 'Comma' 5. Map Front → Question and Back → Answer 6. Click Import"
)

OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT", "60"))
DEFAULT_DEV_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]


def _compute_allowed_origins() -> List[str]:
    allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
    if allowed_origins_env:
        origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
    else:
        origins = DEFAULT_DEV_ORIGINS.copy()

    if not origins:
        origins = ["*"]
        logger.warning(
            "ALLOWED_ORIGINS not set; defaulting to '*' for testing. "
            "Restrict allowed origins in production deployments."
        )

    return origins

PROMPT_IMPORT_ERROR: Optional[str] = None
try:
    from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
except Exception as exc:  # pragma: no cover - handled at runtime
    SYSTEM_PROMPT = ""
    USER_PROMPT_TEMPLATE = ""
    PROMPT_IMPORT_ERROR = str(exc)

app = FastAPI()

allowed_origins = _compute_allowed_origins()
logger.info("Configured CORS allowed origins: %s", allowed_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def healthcheck() -> Dict[str, str]:
    """
    Simple health check endpoint for deployment monitoring.
    """
    logger.debug("Health check endpoint hit.")
    return {"status": "healthy", "message": "PDF to Anki API is running"}


def extract_text_by_page(pdf_path: Path) -> List[Dict[str, str]]:
    """
    Return plain text for each PDF page using PyMuPDF's default text extractor.
    """
    pages: List[Dict[str, str]] = []

    document = fitz.open(pdf_path)
    try:
        for index, page in enumerate(document, start=1):
            page_text = page.get_text()
            if page_text:
                pages.append({"page": index, "text": page_text})
    finally:
        document.close()

    return pages


def _extract_json_segment(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        return "[]"

    if "```" in text:
        start = text.find("```")
        end = text.find("```", start + 3)
        if end != -1:
            fenced = text[start + 3 : end].strip()
            if fenced.lower().startswith("json"):
                fenced = fenced[4:].strip()
            return fenced or "[]"

    return text


def _parse_definitions_from_response(content: str) -> List[Dict[str, str]]:
    candidate = _extract_json_segment(content)

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError("Response was not valid JSON.") from exc

    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array of definitions.")

    cleaned: List[Dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        term = item.get("term")
        definition = item.get("definition")
        if isinstance(term, str) and isinstance(definition, str):
            term_clean = term.strip()
            definition_clean = definition.strip()
            if term_clean and definition_clean:
                cleaned.append({"term": term_clean, "definition": definition_clean})

    return cleaned


def extract_definitions_with_ai(page_text: str, page_num: int, client: OpenAI) -> List[Dict[str, str]]:
    user_prompt = USER_PROMPT_TEMPLATE.format(page_text=page_text.strip())

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=2000,
        )
    except Exception as exc:
        raise RuntimeError(f"OpenAI API request failed: {exc}") from exc

    if not response.choices:
        return []

    content = response.choices[0].message.content or ""

    try:
        definitions = _parse_definitions_from_response(content)
    except ValueError as exc:
        logger.warning("Failed to parse definitions on page %s: %s", page_num, exc)
        return []

    for item in definitions:
        item["page"] = page_num

    return definitions


def format_definitions_for_anki(definitions: List[Dict[str, str]]) -> str:
    """
    Convert definition records into a CSV string suitable for Anki import.
    """
    output = StringIO()
    writer = csv.writer(
        output,
        delimiter=",",
        quotechar='"',
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
    )
    writer.writerow(["Front", "Back"])

    for record in definitions:
        term = record.get("term", "")
        definition = record.get("definition", "")
        writer.writerow([term, definition])

    return output.getvalue()


def _merge_definitions(definitions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    merged: "OrderedDict[str, Dict[str, str]]" = OrderedDict()

    for definition in definitions:
        term = definition.get("term")
        if not isinstance(term, str):
            continue

        key = term.strip().lower()
        if key in merged:
            continue

        merged[key] = definition

    return list(merged.values())


def _validate_pdf_upload(upload: UploadFile) -> str:
    if not upload.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required.",
        )

    original_filename = Path(upload.filename).name

    if not original_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed.",
        )

    content_length = upload.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
            if declared_size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="File too large. Max size is 50MB.",
                )
        except ValueError:
            logger.debug("Invalid content-length header; skipping size pre-check.")

    return original_filename


def _ensure_prompts_available() -> None:
    if PROMPT_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing prompts configuration. Copy prompts.py.example to prompts.py and adjust the values. "
            f"Import error: {PROMPT_IMPORT_ERROR}"
        )


def _build_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it before calling the API."
        )
    # The OpenAI client reads the API key from the environment by default.
    return OpenAI(timeout=OPENAI_TIMEOUT_SECONDS)


async def _extract_definitions_from_upload(upload: UploadFile) -> List[Dict[str, str]]:
    _ensure_prompts_available()

    try:
        client = _build_openai_client()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    temp_path: Optional[Path] = None
    bytes_written = 0
    all_definitions: List[Dict[str, str]] = []

    logger.info("Beginning PDF processing for upload: %s", upload.filename)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_path = Path(temp_file.name)

            while True:
                chunk = await upload.read(CHUNK_SIZE)
                if not chunk:
                    break

                bytes_written += len(chunk)
                if bytes_written > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="File too large. Max size is 50MB.",
                    )

                temp_file.write(chunk)

        if bytes_written == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        pages = extract_text_by_page(temp_path)

        for page in pages:
            page_number = page["page"]
            page_text = page["text"]
            try:
                definitions = extract_definitions_with_ai(page_text, page_number, client)
            except RuntimeError as exc:
                logger.warning("API error on page %s: %s", page_number, exc)
                continue
            except Exception as exc:
                logger.warning("Unexpected error on page %s: %s", page_number, exc)
                continue

            if definitions:
                all_definitions.extend(definitions)
            else:
                logger.info("No definitions returned for page %s.", page_number)

            # Rate limiting can be enforced here if needed (e.g., token bucket per user).
            time.sleep(0.5)

    except HTTPException:
        raise
    except Exception as exc:
        if temp_path:
            temp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process the uploaded file.",
        ) from exc
    finally:
        await upload.close()
        if temp_path:
            temp_path.unlink(missing_ok=True)

    if not all_definitions:
        logger.info("No definitions extracted for upload: %s", upload.filename)
        return []

    logger.info(
        "Completed PDF processing for upload: %s; extracted %s unique definitions.",
        upload.filename,
        len(all_definitions),
    )

    return _merge_definitions(all_definitions)


@app.post("/api/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    """
    Accept a PDF upload, stream it to disk, extract page text, and request AI-powered definitions.
    """
    _validate_pdf_upload(file)
    logger.info("Received request: /api/process-pdf for file %s", file.filename)
    definitions = await _extract_definitions_from_upload(file)
    csv_data = format_definitions_for_anki(definitions)

    return {
        "status": "success",
        "count": len(definitions),
        "definitions": definitions,
        "csv_data": csv_data,
        "anki_import_instructions": ANKI_IMPORT_INSTRUCTIONS,
    }


@app.post("/api/download-anki")
async def download_anki(file: UploadFile = File(...)):
    """
    Accept a PDF upload and return an Anki-ready CSV file.
    """
    _validate_pdf_upload(file)
    logger.info("Received request: /api/download-anki for file %s", file.filename)
    definitions = await _extract_definitions_from_upload(file)
    csv_data = format_definitions_for_anki(definitions)

    headers = {
        "Content-Disposition": 'attachment; filename="textbook_definitions.csv"'
    }

    return Response(
        content=csv_data,
        media_type="text/csv; charset=utf-8",
        headers=headers,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
