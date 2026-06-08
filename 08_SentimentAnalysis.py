import os
from dotenv import load_dotenv
from transformers import pipeline
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Diccionario de respuesta segun emocion

emotion_responses = {
    "anger": "Respond more Angry than the user message",
    "disgust": "Respond more Disgust than the user message",
    "fear": "Respond more Fear than the user message",
    "joy": "Respond more Joy than the user message",
    "neutral": "Respond more Neutral than the user message",
    "sadness": "Respond more Sad than the user message",
    "surprise": "Respond more Surprise than the user message"
}

classifier = pipeline("sentiment-analysis", model="j-hartmann/emotion-english-distilroberta-base")

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

while True:
    user_prompt = input("Tú: ")
    
    if user_prompt.lower() == "chau":
        break
    
    emotion_detected = classifier(user_prompt)
    robot_response = emotion_responses[emotion_detected[0]["label"]]
    
    print(f"\nEmoción detectada: {emotion_detected[0]['label']}")
    
    messages = [
        HumanMessage(content=user_prompt),
        SystemMessage(content=f"You are a helpful assistant, {robot_response}")
    ]
    
    response = llm.invoke(messages).content
    print(f"llm: {response}\n")