import os
import io
import json
import random
import requests
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pypdf

load_dotenv(Path(__file__).parent / ".env", override=True)

# Global variable to store PDF text
pdf_text = ""

# --- API Keys ---
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

print(f"[startup] Gemini key: {'Yes' if GEMINI_KEY else 'No'}")
print(f"[startup] Groq key:   {'Yes' if GROQ_KEY else 'No'}")
print(f"[startup] OpenRouter key: {'Yes' if OPENROUTER_KEY else 'No'}")

# --- Gemini setup (only if key exists) ---
gemini_model = None
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# ============================================================
# Multi-provider fallback: Gemini → Groq → OpenRouter
# ============================================================

def _try_gemini(prompt: str) -> str:
    """Try Gemini first."""
    if not gemini_model:
        raise RuntimeError("Gemini key not configured")
    print("[ai] Trying Gemini...")
    response = gemini_model.generate_content(prompt)
    answer = getattr(response, "text", str(response))
    print(f"[ai] Gemini OK: {answer[:120]}")
    return answer


def _try_groq(prompt: str) -> str:
    """Fallback to Groq (free tier with llama-3.3-70b-versatile)."""
    if not GROQ_KEY:
        raise RuntimeError("Groq key not configured")
    print("[ai] Trying Groq...")
    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    res.raise_for_status()
    answer = res.json()["choices"][0]["message"]["content"]
    print(f"[ai] Groq OK: {answer[:120]}")
    return answer


def _try_openrouter(prompt: str) -> str:
    """Backup: OpenRouter (free models available)."""
    if not OPENROUTER_KEY:
        raise RuntimeError("OpenRouter key not configured")
    print("[ai] Trying OpenRouter...")
    res = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"},
        json={
            "model": "mistralai/mistral-7b-instruct:free",
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    res.raise_for_status()
    answer = res.json()["choices"][0]["message"]["content"]
    print(f"[ai] OpenRouter OK: {answer[:120]}")
    return answer


def generate_answer(prompt: str) -> str:
    """Try each provider in order; return first success."""
    providers = [
        ("Gemini", _try_gemini),
        ("Groq", _try_groq),
        ("OpenRouter", _try_openrouter),
    ]
    errors = []
    for name, fn in providers:
        try:
            return fn(prompt)
        except Exception as e:
            print(f"[ai] {name} failed: {e}")
            errors.append(f"{name}: {e}")
    # All providers failed
    return "All free APIs exhausted, try later. Errors: " + " | ".join(errors)


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get_index():
    return FileResponse("index.html")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global pdf_text
    # Save the file locally
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
            
        # Extract text using pypdf
        pdf_text = ""
        reader = pypdf.PdfReader(io.BytesIO(content))
        for page in reader.pages:
            pdf_text += page.extract_text() or ""
            
        # Limit text length (first 5000 chars)
        pdf_text = pdf_text[:5000].strip()
        print(f"[/upload] Extracted {len(pdf_text)} characters from PDF.")
        
        if not pdf_text:
            pdf_text = "[Notice: No readable text could be extracted from the uploaded PDF. It might be a scanned document or an image.]"
        
        return {"filename": file.filename, "message": "File uploaded successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AskRequest(BaseModel):
    question: str


class QuizRequest(BaseModel):
    topic: str = ""
    count: int = 5


@app.post("/ask")
async def ask_question(req: AskRequest):
    global pdf_text
    print(f"[/ask] Received question: {req.question}")
    
    if not pdf_text:
        return {"answer": "Please upload a PDF first"}
        
    prompt = f"Based on the following content:\n{pdf_text}\n\nAnswer this question:\n{req.question}"
    
    try:
        answer = generate_answer(prompt)
        return {"answer": answer}
    except Exception as e:
        print(f"[/ask] Error: {e}")
        return {"error": str(e)}


@app.post("/quiz")
async def generate_quiz(req: QuizRequest):
    global pdf_text
    
    count = min(req.count, 15)
    
    if not pdf_text and not req.topic:
        return {"error": "Please upload a PDF or provide a topic"}
        
    context = pdf_text[:3000] if pdf_text else req.topic
    
    prompt = f"""
You are an AI tutor.

Use ONLY the following content to generate questions:

{context}

Generate {count} multiple choice questions.

STRICT RULES:
- Each question must have 4 options (A, B, C, D)
- Include correct answer
- Do NOT use outside knowledge if PDF is provided
- Keep questions clear and relevant

Output strictly in JSON format. The JSON should be an object with a "questions" key. 
Each question in the list should be a dictionary with "question" (string), "options" (list of 4 strings), and "answer" (string matching one of the options exactly).
Return ONLY the JSON object, no extra text.
"""
    print(f"[/quiz] Generating {count} questions for topic/context...")
    try:
        raw = generate_answer(prompt)
        # Clean markdown fences if present (e.g. ```json ... ```)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]  # remove first line
            cleaned = cleaned.rsplit("```", 1)[0]  # remove last fence
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"[/quiz] JSON parse failed, returning raw text")
        return {"error": "Quiz response was not valid JSON. Raw: " + raw[:500]}
    except Exception as e:
        print(f"[/quiz] Error: {e}")
        return {"error": str(e)}
