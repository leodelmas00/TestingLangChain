from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Frustration API", version="1.0.0")

emotion_responses = {
    "anger": "Respond with calm empathy. Acknowledge the frustration and offer a clear solution.",
    "disgust": "Respond with empathy. Acknowledge the negative reaction and help carefully.",
    "fear": "Respond with reassurance. Be clear, calm, and supportive.",
    "joy": "Respond warmly and positively.",
    "neutral": "Respond normally and helpfully.",
    "sadness": "Respond gently and supportively.",
    "surprise": "Respond clearly and adapt to the unexpected tone."
}

classifier = pipeline(
    "sentiment-analysis",
    model="j-hartmann/emotion-english-distilroberta-base"
)

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

class ChatRequest(BaseModel):
    message: str
    conversation_history: list = []

class FrustrationRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {
        "status": "ok",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }

@app.post("/detect-frustration")
def detect_frustration(req: FrustrationRequest):
    result = classifier(req.text)[0]
    label = result["label"]
    score = result["score"]
    return {
        "label": label,
        "score": score,
        "strategy": emotion_responses.get(label.lower(), "Respond helpfully and with empathy.")
    }

@app.post("/chat")
def chat(req: ChatRequest):
    result = classifier(req.message)[0]
    label = result["label"]
    strategy = emotion_responses.get(label.lower(), "Respond helpfully and with empathy.")

    messages = [SystemMessage(content=f"You are a helpful assistant. {strategy}")]

    for msg in req.conversation_history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))

    messages.append(HumanMessage(content=req.message))

    response = llm.invoke(messages).content

    return {
        "emotion": label,
        "strategy": strategy,
        "response": response
    }