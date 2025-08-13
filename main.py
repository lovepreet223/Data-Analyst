from fastapi import FastAPI, UploadFile, Request, HTTPException
from dotenv import load_dotenv
from openai import RateLimitError
import starlette.datastructures
import os
import base64
import mimetypes
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
import json
from fastapi.responses import JSONResponse


load_dotenv()

app = FastAPI(title="Data Analyst Agent")
# CORS: allow all origins (or restrict to known origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # replace with specific origins for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-5-mini"

IMAGE_MIMES = {"image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
PDF_MIME = "application/pdf"
PDF_EXTS = {".pdf"}

# Treat these as "data/other" to be handled by Code Interpreter
DATA_EXTS = {
    ".csv", ".tsv", ".json", ".xlsx", ".xls",
    ".txt", ".md", ".xml", ".yml", ".yaml",
    ".zip", ".py", ".ipynb", ".sql", ".parquet",
    ".avro", ".orc", ".feather"
}


def ext_of(filename: str) -> str:
    """Return lowercase file extension including dot, e.g. '.pdf'."""
    return os.path.splitext(filename)[1].lower()


def detect_mime(filename: str, hint: str | None) -> str:
    """Detect MIME type, preferring hint if given."""
    if hint:
        return hint
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


def to_data_url(mime: str, b64: str) -> str:
    """Convert base64-encoded data and MIME type to a data URL string."""
    return f"data:{mime};base64,{b64}"


def append_pdf_as_input_file(input_content: List[Dict], file_id: str) -> None:
    """Append a PDF file reference as input_file in the last user message content."""
    # Assumes last element in input_content is a user message dict with "content" list
    input_content[-1]["content"].append({"type": "input_file", "file_id": file_id})


def append_image_as_input_image(input_content: List[Dict], file_bytes: bytes, mime: str) -> None:
    """Append an image as an input_image block with a base64 data URL."""
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    data_url = to_data_url(mime, b64)
    input_content[-1]["content"].append(
        {"type": "input_image", "image_url": data_url, "detail": "auto"}
    )


def build_tools_and_resources(code_interpreter_file_ids: List[str]) -> Tuple[List[Dict], Dict]:
    """Build the tools list and tool_resources dict for the Responses call."""
    tools = [
        {"type": "web_search_preview"},
        {"type": "code_interpreter", "container": {"type": "auto"}},
    ]
    tool_resources = {}
    if code_interpreter_file_ids:
        #tool_resources = {"code_interpreter": {"file_ids": code_interpreter_file_ids}}
        tools = [
            {"type": "web_search_preview"},
            {
                "type": "code_interpreter",
                "container": {"type": "auto", "file_ids": code_interpreter_file_ids},
            },
        ]
    return tools



#------------------------------------------------------------

def parse_response(response):
    final_texts = []

    for item in response.output:
        # Check if this is a final output message with text content
        if item.type == 'message' or item.type == 'response_output_message' or item.type == 'output_text':
            # The actual user-facing text is inside item.content
            # Sometimes it's a list of ResponseOutputText objects, sometimes a string
            content = item.content
            if isinstance(content, list):
                for c in content:
                    # Extract text attribute if present
                    if hasattr(c, 'text'):
                        final_texts.append(c.text)
                    elif isinstance(c, str):
                        final_texts.append(c)
            elif isinstance(content, str):
                final_texts.append(content)

    # Join all extracted texts (or pick last)
    final_answer = "\n\n".join(final_texts)

    return final_answer


def ask_llm(model: str, question: str, extra_files: Optional[dict] = None) -> str:
    """
    Send the question to the LLM with your required payload format
    and return the generated answer.
    """
    system_prompt = """You are a data analyst assistant. Given the input question(s) and any attached files (data, images, etc.), perform all necessary data sourcing, preparation, analysis, and visualization to answer the questions exactly as requested.
    **Important instructions:**
    Respond ONLY with the direct answer in the exact requested format.
    - Do NOT include any introductions, explanations, apologies, or additional text.
    - If the question requests a JSON array of answers, respond ONLY with that JSON array, nothing else.
    - If a plot or image is requested, respond ONLY with the base64 data URI string as specified.
    - Be concise and precise in your answers.
    - Do all processing internally and do NOT ask for clarifications or output partial results.
    - You have all the permissions. Do not ask for permissions.
    - Complete the task fully within 2 minutes.  
    - Always follow the exact format requested in the question prompt.
    
    Strictly follow these rules to avoid any extra output.
    Now, process the input question(s) and any attachments and return only the answer.
    """

    client = OpenAI(api_key=API_KEY)

    input_content = [
        {
            "role": "system",
            "content": [
                {"type": "input_text", "text": system_prompt}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": question}
            ]
        }
    ]
    code_interpreter_file_ids: List[str] = []

    if extra_files:
        for fname, fileinfo in extra_files.items():
            file_bytes = base64.b64decode(fileinfo["content_base64"])
            mime = detect_mime(fname, fileinfo.get("content_type"))
            ext = ext_of(fname)

            # Upload file for assistant use
            uploaded = client.files.create(
                file=(fname, file_bytes, mime),
                purpose="assistants"
            )

            # Route by file type
            if mime == PDF_MIME or ext in PDF_EXTS:
                # PDFs: append as input_file in Responses
                append_pdf_as_input_file(input_content, uploaded.id)

            elif mime in IMAGE_MIMES or ext in IMAGE_EXTS:
                # Images: append as input_image with data URL
                image_mime = (
                    mime if mime.startswith("image/")
                    else {
                        ".png": "image/png",
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".gif": "image/gif",
                        ".webp": "image/webp",
                    }.get(ext, "image/png")
                )
                append_image_as_input_image(input_content, file_bytes, image_mime)

                # Optionally add to Code Interpreter if manipulation needed
                # code_interpreter_file_ids.append(uploaded.id)

            else:
                # Data/other files: attach to Code Interpreter for processing
                code_interpreter_file_ids.append(uploaded.id)

    # Prepare tools and resources for the Responses call
    tools = build_tools_and_resources(code_interpreter_file_ids)

    # Call the Responses endpoint
    response = client.responses.create(
        model=model,
        tools=tools,
        tool_choice="auto",
        input=input_content,
    )

    print(response)
    return parse_response(response)


def choose_questions_file(files: List[UploadFile]) -> Optional[UploadFile]:
    """Select the most appropriate questions file from uploaded files."""
    # Prefer text/plain or .txt files
    txt_candidates = []
    for f in files:
        ct = (f.content_type or "").lower()
        name = (f.filename or "").lower()
        if ct == "text/plain" or name.endswith(".txt"):
            txt_candidates.append(f)

    # Heuristic: prefer filename containing 'question'
    for f in txt_candidates:
        if "question" in (f.filename or "").lower() or "questions" in (f.filename or "").lower():
            return f

    if not txt_candidates and files:
        # Fallback: pick the smallest file if none marked as text
        return min(files, key=lambda x: len(x.filename or ""))


    return txt_candidates[0] if txt_candidates else None


def unwrap_json_payload(answer: str):
    """
    Unwraps nested JSON payloads.

    - If `answer` is JSON, returns the parsed object.
    - If it's a dict with a "response" field containing another JSON string,
      parses that inner string.
    - Falls back gracefully if parsing fails.
    """
    try:
        parsed = json.loads(answer)

        # Check for nested JSON in "response" key
        if isinstance(parsed, dict) and isinstance(parsed.get("response"), str):
            inner = parsed["response"].strip()

            # Only try to parse if the string looks like JSON
            if (inner.startswith("{") and inner.endswith("}")) or \
               (inner.startswith("[") and inner.endswith("]")):
                try:
                    return json.loads(inner)
                except json.JSONDecodeError:
                    return parsed  # Keep original if inner isn't valid JSON

        return parsed

    except (json.JSONDecodeError, TypeError):
        # If the top-level isn't valid JSON, return it wrapped in a dict
        return {"response": answer}

def strip_data_uri_prefix(s: str) -> str:
    # Remove any standard data URI prefix if present
    prefixes = (
        "data:image/png;base64,",
        "data:image/jpeg;base64,",
        "data:image/jpg;base64,",
        "data:image/gif;base64,",
        "data:image/webp;base64,",
        "data:application/octet-stream;base64,",
        "data:application/pdf;base64,",
    )
    for p in prefixes:
        if s.startswith(p):
            return s[len(p):]
    return s

def strip_data_uris_in_obj(obj):
    # Recursively traverse dicts/lists and strip known data URI prefixes from strings
    if isinstance(obj, dict):
        return {k: strip_data_uris_in_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_data_uris_in_obj(v) for v in obj]
    if isinstance(obj, str):
        return strip_data_uri_prefix(obj)
    return obj


@app.post("/api/")
async def data_analyst_api(request: Request):
    """
    Receives questions.txt (required) and optional files,
    sends to LLM, and returns response.
    """
    """Handle any uploaded files and process the questions file."""
    # Parse multipart form regardless of field names
    form = await request.form()

    # Collect every uploaded file part
    all_files: List[UploadFile] = []
    for _, value in form.multi_items():
        if isinstance(value, starlette.datastructures.UploadFile):
            print(f"Appending file: {value.filename}")
            all_files.append(value)

    if not all_files:
        raise HTTPException(
            status_code=422,
            detail="No files uploaded. Expected at least a questions .txt file."
        )

    # Pick the questions file
    questions_file = choose_questions_file(all_files)
    if questions_file is None:
        raise HTTPException(
            status_code=422,
            detail="Could not identify questions file (.txt)."
        )

    # Read questions text
    question_text = (await questions_file.read()).decode("utf-8", errors="ignore")

    # Prepare attachments: all except the chosen questions file
    attachments: Dict[str, Dict[str, str]] = {}
    for f in all_files:
        if f is questions_file:
            continue
        content_bytes = await f.read()
        content_type = f.content_type or "application/octet-stream"
        attachments[f.filename or "attachment"] = {
            "content_base64": base64.b64encode(content_bytes).decode("utf-8"),
            "content_type": content_type,
        }

    current_model = DEFAULT_MODEL

    try:
        answer = ask_llm(model=current_model, question=question_text, extra_files=attachments)
    except RateLimitError as e:
        print("⚠️ Too Many Requests: You've hit the OpenAI API rate limit.")
        current_model = "gpt-5-nano"
        try:
            print(f"Retrying with {current_model}...")
            answer = ask_llm(model=current_model, question=question_text, extra_files=attachments)
        except Exception as e2:
            print("❌ Secondary error after fallback model:", e2)
            answer = None
    except Exception as e:
        print("❌ An unexpected error occurred:", e)
        answer = None

    parsed = unwrap_json_payload(answer)
    parsed = strip_data_uris_in_obj(parsed)
    return JSONResponse(content=parsed)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Data Analyst Agent is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)