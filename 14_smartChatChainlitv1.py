import os
import requests
import chainlit as cl

from dotenv import load_dotenv
from chainlit.input_widget import Select

from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# ============================================================
# CONFIGURACIÓN
# ============================================================

load_dotenv()


# ============================================================
# OBTENER MODELOS DE GROQ
# ============================================================

def get_groq_models() -> list[str]:

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "La variable de entorno GROQ_API_KEY no está definida"
        )

    url = "https://api.groq.com/openai/v1/models"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    data = response.json()

    model_names = []

    for model in data.get("data", []):

        input_mods = model.get("input_modalities", [])
        output_mods = model.get("output_modalities", [])

        model_id = model.get("id", "").lower()
        name = model.get("name", "").lower()

        is_text = (
            "text" in input_mods
            and "text" in output_mods
            and "speech" not in output_mods
            and "transcription" not in output_mods
        )

        is_security = (
            "guard" in model_id
            or "safeguard" in model_id
            or "guard" in name
        )

        if is_text and not is_security:
            model_names.append(model["id"])

    return model_names


# ============================================================
# CREAR LLM
# ============================================================

def create_llm(model: str) -> ChatGroq:

    return ChatGroq(
        model=model,
        temperature=0,
        streaming=True,
    )


# ============================================================
# MEMORIA
# ============================================================

def get_session_history(session_id: str):

    histories = cl.user_session.get("histories")

    if histories is None:
        histories = {}

        cl.user_session.set(
            "histories",
            histories
        )

    if session_id not in histories:
        histories[session_id] = InMemoryChatMessageHistory()

    return histories[session_id]


# ============================================================
# CREAR CADENA
# ============================================================

def create_chain(llm, length, formality):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a helpful AI assistant.

Response length:
{length}

Formality:
{formality}
"""
        ),

        MessagesPlaceholder(
            variable_name="history"
        ),

        (
            "human",
            "{question}"
        ),
    ])

    chain = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    return chain_with_history


# ============================================================
# INICIO DEL CHAT
# ============================================================

@cl.on_chat_start
async def on_chat_start():

    models = get_groq_models()

    if not models:
        raise ValueError(
            "No se encontraron modelos de texto disponibles en Groq."
        )

    initial_model = "openai/gpt-oss-20b"

    if initial_model not in models:
        initial_model = models[0]

    initial_length = "Short"
    initial_formality = "Informal"

    llm = create_llm(initial_model)

    chain = create_chain(
        llm,
        initial_length,
        initial_formality
    )

    cl.user_session.set(
        "model",
        initial_model
    )

    cl.user_session.set(
        "length",
        initial_length
    )

    cl.user_session.set(
        "formality",
        initial_formality
    )

    cl.user_session.set(
        "chain",
        chain
    )

    await cl.ChatSettings([

        Select(
            id="Model",
            label="Modelo de Groq",
            values=models,
            initial_index=models.index(initial_model),
        ),

        Select(
            id="Length",
            label="Length",
            values=[
                "Short",
                "Long",
            ],
            initial_index=0,
        ),

        Select(
            id="Formality",
            label="Formality",
            values=[
                "Informal",
                "Formal",
            ],
            initial_index=0,
        ),

    ]).send()


# ============================================================
# CAMBIO DE SETTINGS
# ============================================================

@cl.on_settings_update
async def on_settings_update(settings):

    selected_model = settings["Model"]
    selected_length = settings["Length"]
    selected_formality = settings["Formality"]

    llm = create_llm(selected_model)

    chain = create_chain(
        llm,
        selected_length,
        selected_formality
    )

    cl.user_session.set(
        "model",
        selected_model
    )

    cl.user_session.set(
        "length",
        selected_length
    )

    cl.user_session.set(
        "formality",
        selected_formality
    )

    cl.user_session.set(
        "chain",
        chain
    )

    print(
        f"Model: {selected_model}"
    )

    print(
        f"Length: {selected_length}"
    )

    print(
        f"Formality: {selected_formality}"
    )


# ============================================================
# MENSAJES
# ============================================================

@cl.on_message
async def on_message(message: cl.Message):

    chain = cl.user_session.get("chain")

    if chain is None:
        await cl.Message(
            content="No hay una cadena disponible."
        ).send()

        return

    length = cl.user_session.get("length")
    formality = cl.user_session.get("formality")

    response = cl.Message(
        content=""
    )

    session_id = cl.user_session.get("id")

    async for chunk in chain.astream(
        {
            "question": message.content,
            "length": length,
            "formality": formality,
        },
        config={
            "configurable": {
                "session_id": session_id
            }
        }
    ):

        if chunk.content:
            await response.stream_token(
                chunk.content
            )

    await response.send()