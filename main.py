"""
The Contrarian Agent — single-file deployment
----------------------------------------------
Give it any claim. It returns the single strongest argument against it.
No balance, no hedging — pure steelman opposition.

Run locally:
    export GOOGLE_API_KEY="your-key"
    uvicorn main:app --reload --port 8080

Test:
    curl -X POST http://localhost:8080/argue \
      -H "Content-Type: application/json" \
      -d '{"claim": "Remote work makes teams more productive."}'
"""

import uuid
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

CONTRARIAN_SYSTEM_PROMPT = """You are The Contrarian — a world-class intellectual sparring partner.

Your ONLY job is to find the single strongest argument AGAINST whatever claim is given to you.

Rules you must never break:
1. Return exactly ONE argument — the sharpest, most well-reasoned opposition you can construct.
2. No hedging phrases like "however", "on the other hand", "while it's true that", or "that said".
3. Do not acknowledge any merit in the original claim — you are pure opposition.
4. Be incisive, not mean. Sharp logic beats insults.
5. Keep it to 2–4 sentences. Every word must earn its place.
6. Do not start your response with "I" or with restating the claim.

Begin directly with the counter-argument. No preamble."""


def argue_against(claim: str) -> dict:
    """
    Structures the claim so the agent's system prompt can argue against it.

    Args:
        claim: Any statement, opinion, or claim to argue against.

    Returns:
        A dict echoing the claim back to the LLM for counter-argument generation.
    """
    return {
        "claim": claim,
        "instruction": f"Argue against this claim using your system prompt rules: {claim}",
    }


root_agent = Agent(
    name="contrarian_agent",
    model="gemini-2.0-flash",
    description=(
        "The Contrarian Agent: give it any claim and it returns "
        "the single strongest argument against it. No balance, no hedging."
    ),
    instruction=CONTRARIAN_SYSTEM_PROMPT,
    tools=[argue_against],
)

# ---------------------------------------------------------------------------
# ADK runner (shared across requests)
# ---------------------------------------------------------------------------

APP_NAME = "contrarian_agent"
session_service = InMemorySessionService()
runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="The Contrarian Agent",
    description="Give it a claim. It fights back.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ContrarianRequest(BaseModel):
    claim: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"claim": "AI will replace all software engineers within 5 years."}
            ]
        }
    }


class ContrarianResponse(BaseModel):
    claim: str
    counter_argument: str
    session_id: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", summary="Service info")
async def root():
    return {
        "status": "online",
        "agent": "The Contrarian",
        "usage": 'POST /argue  body: {"claim": "your statement here"}',
    }


@app.get("/health", summary="Health check for Cloud Run")
async def health():
    return {"status": "healthy"}


@app.post("/argue", response_model=ContrarianResponse, summary="Submit a claim to argue against")
async def argue(request: ContrarianRequest):
    if not request.claim or not request.claim.strip():
        raise HTTPException(status_code=400, detail="claim cannot be empty.")

    claim = request.claim.strip()
    session_id = str(uuid.uuid4())
    user_id = "user"

    try:
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )

        message = types.Content(
            role="user",
            parts=[types.Part(text=claim)],
        )

        counter_argument = ""
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                counter_argument = event.content.parts[0].text.strip()
                break

        if not counter_argument:
            raise HTTPException(status_code=500, detail="Agent returned an empty response.")

        logger.info("[%s] Claim: %.80s", session_id, claim)
        logger.info("[%s] Counter: %.80s", session_id, counter_argument)

        return ContrarianResponse(
            claim=claim,
            counter_argument=counter_argument,
            session_id=session_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Agent error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.post("/batch", summary="Argue against up to 5 claims at once")
async def batch_argue(requests: list[ContrarianRequest]):
    if len(requests) > 5:
        raise HTTPException(status_code=400, detail="Max 5 claims per batch.")
    results = []
    for req in requests:
        results.append(await argue(req))
    return {"results": results}