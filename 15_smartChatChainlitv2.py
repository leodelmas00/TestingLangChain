import os
import requests
import chainlit as cl

from dotenv import load_dotenv
from chainlit.input_widget import Select

from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

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

PDF_PATH = "data/SkyRouteTravelAgency.pdf"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

CHROMA_PATH = "chroma_db"

COLLECTION_NAME = "skyroute_travel_agency"


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
# CREAR VECTOR STORE
# ============================================================

def create_vectorstore():

    print("Cargando PDF...")

    loader = PyPDFLoader(PDF_PATH)

    documents = loader.load()

    print(
        f"PDF cargado: {len(documents)} páginas"
    )


    # --------------------------------------------------------
    # Dividir documentos
    # --------------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        f"Chunks creados: {len(chunks)}"
    )


    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    print(
        f"Cargando embeddings: {EMBEDDING_MODEL}"
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


    # --------------------------------------------------------
    # Chroma
    # --------------------------------------------------------

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
    )

    print("Vector store creado.")

    return vectorstore


# ============================================================
# CREAR RETRIEVER
# ============================================================

def create_retriever(vectorstore):

    return vectorstore.as_retriever(
        search_kwargs={
            "k": 4
        }
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

        histories[session_id] = (
            InMemoryChatMessageHistory()
        )

    return histories[session_id]


# ============================================================
# CREAR CADENA RAG
# ============================================================

def create_chain(
    llm,
    retriever,
    length,
    formality
):

    prompt = ChatPromptTemplate.from_messages([

        (
            "system",
            """
You are a helpful AI assistant for SkyRoute Travel Agency.

Use the retrieved context below to answer the user's
question.

You should primarily use the information contained
in the retrieved context.

If the retrieved context does not contain enough
information to answer the question, clearly say that
you do not have enough information.

Do not invent information that is not present in
the retrieved context.

Response length:
{length}

Formality:
{formality}

Retrieved context:
{context}
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
# FORMATEAR DOCUMENTOS
# ============================================================

def format_docs(docs):

    return "\n\n".join(
        document.page_content
        for document in docs
    )


# ============================================================
# INICIO DEL CHAT
# ============================================================

@cl.on_chat_start
async def on_chat_start():

    # --------------------------------------------------------
    # Modelos de Groq
    # --------------------------------------------------------

    models = get_groq_models()

    if not models:

        raise ValueError(
            "No se encontraron modelos de texto disponibles en Groq."
        )


    initial_model = "openai/gpt-oss-20b"

    if initial_model not in models:

        initial_model = models[0]


    # --------------------------------------------------------
    # Settings iniciales
    # --------------------------------------------------------

    initial_length = "Short"

    initial_formality = "Informal"


    # --------------------------------------------------------
    # Crear / cargar Vector Store
    # --------------------------------------------------------

    vectorstore = create_vectorstore()

    retriever = create_retriever(
        vectorstore
    )


    # --------------------------------------------------------
    # Crear LLM
    # --------------------------------------------------------

    llm = create_llm(
        initial_model
    )


    # --------------------------------------------------------
    # Crear cadena RAG
    # --------------------------------------------------------

    chain = create_chain(
        llm,
        retriever,
        initial_length,
        initial_formality
    )


    # --------------------------------------------------------
    # Guardar información en la sesión
    # --------------------------------------------------------

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
        "vectorstore",
        vectorstore
    )

    cl.user_session.set(
        "retriever",
        retriever
    )

    cl.user_session.set(
        "chain",
        chain
    )


    # --------------------------------------------------------
    # Settings de Chainlit
    # --------------------------------------------------------

    await cl.ChatSettings([

        Select(
            id="Model",
            label="Modelo de Groq",
            values=models,
            initial_index=models.index(
                initial_model
            ),
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


    # --------------------------------------------------------
    # Recuperar retriever existente
    # --------------------------------------------------------

    retriever = cl.user_session.get(
        "retriever"
    )

    if retriever is None:

        raise ValueError(
            "No existe un retriever disponible."
        )


    # --------------------------------------------------------
    # Crear nuevo LLM
    # --------------------------------------------------------

    llm = create_llm(
        selected_model
    )


    # --------------------------------------------------------
    # Crear nueva cadena
    # --------------------------------------------------------

    chain = create_chain(
        llm,
        retriever,
        selected_length,
        selected_formality
    )


    # --------------------------------------------------------
    # Actualizar sesión
    # --------------------------------------------------------

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

    chain = cl.user_session.get(
        "chain"
    )

    retriever = cl.user_session.get(
        "retriever"
    )


    if chain is None:

        await cl.Message(
            content="No hay una cadena disponible."
        ).send()

        return


    if retriever is None:

        await cl.Message(
            content="No hay un retriever disponible."
        ).send()

        return


    # --------------------------------------------------------
    # Settings actuales
    # --------------------------------------------------------

    length = cl.user_session.get(
        "length"
    )

    formality = cl.user_session.get(
        "formality"
    )


    # --------------------------------------------------------
    # RAG: recuperar documentos relevantes
    # --------------------------------------------------------

    retrieved_docs = await retriever.ainvoke(
        message.content
    )


    context = format_docs(
        retrieved_docs
    )


    # --------------------------------------------------------
    # Mensaje de respuesta
    # --------------------------------------------------------

    response = cl.Message(
        content=""
    )


    # --------------------------------------------------------
    # ID de sesión
    # --------------------------------------------------------

    session_id = cl.user_session.get(
        "id"
    )


    # --------------------------------------------------------
    # Ejecutar cadena
    # --------------------------------------------------------

    async for chunk in chain.astream(
        {
            "question": message.content,
            "context": context,
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


    # --------------------------------------------------------
    # Enviar respuesta
    # --------------------------------------------------------

    await response.send()