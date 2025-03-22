from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import google.generativeai as genai
import logging
import json
import random  # Import the random module
import os  # Import the os module for environment variables
from functools import lru_cache  # Import lru_cache for caching
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates and static files
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
templates = Jinja2Templates(directory=".")  # Ensure 'templates' directory exists

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
GOOGLE_AI_API_KEY = os.environ.get("GOOGLE_AI_API_KEY")
if not GOOGLE_AI_API_KEY:
    logger.error("API key not found in environment")
    raise ValueError("Missing required API configuration")

# Enhanced configuration for Gazi 2.0 persona
generation_config = {
    "temperature": 0.7,
    "top_p": 0.85,
    "top_k": 20,
    "max_output_tokens": 128,
    "response_mime_type": "text/plain",
    "stop_sequences": ["\n\n"],
}

safety_settings = {
    "HATE": "BLOCK_NONE",
    "HARASSMENT": "BLOCK_ONLY_HIGH",
    "SEXUAL": "BLOCK_NONE",
    "DANGEROUS": "BLOCK_MEDIUM_AND_ABOVE"
}

genai.configure(api_key=GOOGLE_AI_API_KEY)
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    safety_settings=safety_settings
)

conversation_history = []
MAX_HISTORY_LENGTH = 3

def load_context_from_txt(filename: str = "assets/data/ins.txt") -> str:
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        logger.warning(f"Missing context file: {filename}")
        return ""

def load_context_from_json(filename: str = "assets/data/context.json") -> str:
    def flatten_json(data):
        if isinstance(data, dict):
            return " ".join(flatten_json(value) for value in data.values())
        elif isinstance(data, list):
            return " ".join(flatten_json(item) for item in data)
        return str(data)

    try:
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
            return flatten_json(data)
    except FileNotFoundError:
        logger.warning(f"Missing context file: {filename}")
        return ""
    except json.JSONDecodeError:
        logger.error("Invalid JSON format")
        return ""

@lru_cache(maxsize=1)
def load_context() -> str:
    txt_context = load_context_from_txt()
    json_context = load_context_from_json()
    return f"{txt_context}\n{json_context}".strip()

initial_context = load_context()
SYSTEM_INSTRUCTIONS = load_context_from_txt("assets/data/ins.txt")

def get_response(question: str, context: str) -> str:
    global conversation_history

    prompt = f"""Follow these instructions: {SYSTEM_INSTRUCTIONS}

    Relevant context: {context}

    Conversation history:"""

    for q, a in conversation_history:
        prompt += f"\nUser: {q}\n{a}"  # Responses added without prefix
    prompt += f"\nUser: {question}\n"
    try:
        response = model.generate_content(prompt)
        answer = response.text

        if not answer:
            answer = "Response generation failed"

        conversation_history.append((question, answer))
        if len(conversation_history) > MAX_HISTORY_LENGTH:
            conversation_history.pop(0)

        return answer
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return "Service unavailable"


# Load JSON data for research
with open("assets/data/research.json", "r") as file:
    research_data = json.load(file)

# Route for root index page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route for partials
@app.get("/partials/{partial_name}", response_class=HTMLResponse)
async def load_partial(request: Request, partial_name: str):
    try:
        if partial_name == "research":
            return templates.TemplateResponse(
                f"partials/{partial_name}.html",
                {"request": request, "research": research_data},
            )
        else:
            return templates.TemplateResponse(f"partials/{partial_name}.html", {"request": request})
    except:
        return HTMLResponse("Partial not found", status_code=404)

@app.post("/chat", response_class=HTMLResponse)
async def chat(
    request: Request,
    message: str = Form(None),  # Allow message to be None
    reset_context: bool = Form(False)  # Default reset_context to False
):
    global conversation_history

    if reset_context:
        conversation_history = []
        return HTMLResponse(
        """<div class='message bot-message' data-clear-context='true'>
            Context reset. Wiped slate. Time to be brilliant... again. 💡
        </div>"""
    )

    if not message:
        return HTMLResponse("<div class='message bot-message'>Please enter a message. What's on your mind? 🧠</div>")

    answer = get_response(message, initial_context)
    response_html = f"""
    <!-- User Chat Bubble -->
    <div class="chat-message flex justify-end animate-fade-in">
        <div class="max-w-md bg-gradient-to-br from-indigo-600 to-violet-600 text-white p-3 rounded-2xl rounded-br-none shadow-md">
            <div class="flex items-center gap-2">
                <div class="avatar placeholder">
                    <div class="bg-violet-700 text-violet-100 rounded-full w-6">
                        <i class="fas fa-user text-sm"></i>
                    </div>
                </div>
                <span class="text-sm font-medium break-words">{message}</span>
            </div>
        </div>
    </div>

    <!-- Assistant Chat Bubble -->
    <div class="chat-message flex justify-start animate-fade-in">
        <div class="max-w-md bg-white text-slate-700 p-3 rounded-2xl rounded-bl-none shadow-md border border-slate-200">
            <div class="flex items-center gap-2">
                <div class="avatar placeholder">
                    <div class="bg-indigo-100 text-indigo-600 rounded-full w-6">
                        <i class="fas fa-robot text-sm"></i>
                    </div>
                </div>
                <span class="text-sm break-words">{answer}</span>
            </div>
        </div>
    </div>
    """
    return HTMLResponse(response_html)