import json
import os
from dotenv import load_dotenv

import gradio as gr

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

loader = PyPDFLoader("data/SkyRouteTravelAgency.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

splits = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Given a chat history and the latest user question,
rewrite the latest question so it can be understood without
the previous conversation.

Do NOT answer the question.

Return ONLY the rewritten question."""
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# NOTA: se añade el placeholder {preferences} para que el asistente
# adapte su estilo de respuesta a las preferencias de largo plazo del usuario.
qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a chatbot assistant.

Answer ONLY using the retrieved context.
If the answer is not contained in the context,
say you don't have enough information.

Adapt your response style to these known user preferences
(apply them only when they don't conflict with answering correctly):
{preferences}

Context:
{context}"""
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])


# ---------------------------------------------------------------------------
# Sistema de memoria de largo plazo de preferencias conversacionales
# ---------------------------------------------------------------------------

PREFERENCES_FILE = "data/user_preferences.json"

DEFAULT_PREFERENCES = {
    "preferredResponseLength": "medium",   # short | medium | long
    "preferredFormat": "mixed",            # prose | list | mixed
    "examplePreference": "when_helpful",   # never | when_helpful | often
    "clarificationPreference": "balanced", # ask_first | balanced | assume_when_reasonable
}

# Valores válidos por campo, usados para validar lo que devuelve la LLM.
VALID_VALUES = {
    "preferredResponseLength": {"short", "medium", "long"},
    "preferredFormat": {"prose", "list", "mixed"},
    "examplePreference": {"never", "when_helpful", "often"},
    "clarificationPreference": {"ask_first", "balanced", "assume_when_reasonable"},
}

# Umbrales de confianza para decidir qué hacer con cada propuesta.
CONFIDENCE_APPLY_DIRECTLY = 0.9
CONFIDENCE_NEEDS_CORROBORATION = 0.6

# Cuántas veces debe repetirse una propuesta "media confianza" (entre los dos
# umbrales) con el MISMO valor antes de aplicarla.
CORROBORATION_COUNT_REQUIRED = 2

_DEFAULT_STORE = {
    "preferences": DEFAULT_PREFERENCES.copy(),
    "pending": {},  # field -> {"value": ..., "count": int, "confidence": float}
}


def _load_store():
    if os.path.exists(PREFERENCES_FILE):
        try:
            with open(PREFERENCES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("preferences", DEFAULT_PREFERENCES.copy())
            data.setdefault("pending", {})
            for field, value in DEFAULT_PREFERENCES.items():
                data["preferences"].setdefault(field, value)
            return data
        except (json.JSONDecodeError, OSError):
            pass
    return json.loads(json.dumps(_DEFAULT_STORE))


def _save_store(store):
    os.makedirs(os.path.dirname(PREFERENCES_FILE), exist_ok=True)
    with open(PREFERENCES_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def get_current_preferences():
    return _load_store()["preferences"]


def format_preferences_for_prompt(preferences):
    return "\n".join(f"- {k}: {v}" for k, v in preferences.items())


preference_analysis_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Eres un analista que revisa una conversación completa entre un
usuario y un asistente para inferir preferencias de largo plazo del usuario.

Debes evaluar EXCLUSIVAMENTE estos campos, y solo proponer un cambio cuando
haya evidencia observable en la conversación (peticiones explícitas del
usuario, reacciones positivas/negativas, correcciones, quejas, etc.). No
inventes evidencia ni propongas cambios especulativos:

- preferredResponseLength: short | medium | long
- preferredFormat: prose | list | mixed
- examplePreference: never | when_helpful | often
- clarificationPreference: ask_first | balanced | assume_when_reasonable

Devuelve ÚNICAMENTE un JSON válido (sin texto adicional, sin markdown, sin
```), con esta forma exacta:

{{
  "updates": [
    {{
      "field": "<uno de los campos anteriores>",
      "value": "<uno de los valores permitidos para ese campo>",
      "confidence": <número entre 0 y 1>,
      "reason": "<breve justificación basada solo en la conversación>"
    }}
  ]
}}

Si no hay evidencia suficiente para ningún campo, devuelve {{"updates": []}}.
No propongas más de un update por campo."""
    ),
    ("human", "Conversación a analizar:\n\n{conversation}")
])


def _conversation_to_text(chat_history):
    lines = []
    for msg in chat_history:
        role = "Usuario" if isinstance(msg, HumanMessage) else "Asistente"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def analyze_conversation_for_preferences(chat_history):
    """Llama a la LLM para proponer actualizaciones de preferencias.

    Devuelve una lista de dicts: {field, value, confidence, reason}.
    Si la LLM falla o responde algo no parseable, devuelve [].
    """
    if not chat_history:
        return []

    conversation_text = _conversation_to_text(chat_history)

    messages = preference_analysis_prompt.format_messages(
        conversation=conversation_text
    )
    result = llm.invoke(messages)
    raw = result.content.strip()

    # Por si el modelo envuelve la respuesta en ```json ... ```
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json\n", "", 1).replace("json", "", 1)
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
        updates = parsed.get("updates", [])
    except (json.JSONDecodeError, AttributeError):
        return []

    valid_updates = []
    for u in updates:
        field = u.get("field")
        value = u.get("value")
        confidence = u.get("confidence")
        reason = u.get("reason", "")

        if field not in VALID_VALUES:
            continue
        if value not in VALID_VALUES[field]:
            continue
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            continue

        valid_updates.append({
            "field": field,
            "value": value,
            "confidence": confidence,
            "reason": reason,
        })

    return valid_updates


def apply_preference_updates(updates):
    """Aplica la lógica de umbrales sobre las propuestas de la LLM.

    - confidence > 0.9  -> se actualiza la preferencia inmediatamente.
    - 0.6 <= confidence <= 0.9 -> se guarda como pendiente; se necesita
      corroboración (misma propuesta) en otra conversación para aplicarse.
    - confidence < 0.6 -> se ignora.

    Devuelve un resumen legible de lo ocurrido, útil para mostrar en la UI.
    """
    store = _load_store()
    preferences = store["preferences"]
    pending = store["pending"]

    summary_lines = []

    for u in updates:
        field, value, confidence, reason = (
            u["field"], u["value"], u["confidence"], u["reason"]
        )

        if confidence > CONFIDENCE_APPLY_DIRECTLY:
            preferences[field] = value
            pending.pop(field, None)
            summary_lines.append(
                f"✅ Actualizado directamente: {field} = {value} "
                f"(confianza {confidence:.2f}). Motivo: {reason}"
            )

        elif confidence >= CONFIDENCE_NEEDS_CORROBORATION:
            current_pending = pending.get(field)
            if current_pending and current_pending["value"] == value:
                current_pending["count"] += 1
                current_pending["confidence"] = confidence
                if current_pending["count"] >= CORROBORATION_COUNT_REQUIRED:
                    preferences[field] = value
                    pending.pop(field, None)
                    summary_lines.append(
                        f"✅ Actualizado tras corroboración: {field} = {value} "
                        f"(confianza {confidence:.2f}). Motivo: {reason}"
                    )
                else:
                    summary_lines.append(
                        f"⏳ Evidencia repetida pero aún insuficiente: "
                        f"{field} = {value} "
                        f"({current_pending['count']}/{CORROBORATION_COUNT_REQUIRED})."
                    )
            else:
                pending[field] = {
                    "value": value,
                    "count": 1,
                    "confidence": confidence,
                }
                summary_lines.append(
                    f"⏳ Propuesta en espera de corroboración: {field} = {value} "
                    f"(confianza {confidence:.2f}). Motivo: {reason}"
                )

        else:
            summary_lines.append(
                f"❌ Ignorado por baja confianza: {field} = {value} "
                f"(confianza {confidence:.2f})."
            )

    store["preferences"] = preferences
    store["pending"] = pending
    _save_store(store)

    if not summary_lines:
        summary_lines.append("No se detectaron propuestas de actualización.")

    return "\n".join(summary_lines)


def end_conversation_and_update_preferences(history):
    """Handler para el botón de la UI: analiza la conversación actual
    y actualiza (o pospone) las preferencias según la confianza."""
    chat_history = normalize_history(history)
    updates = analyze_conversation_for_preferences(chat_history)
    summary = apply_preference_updates(updates)

    current_prefs = get_current_preferences()
    prefs_text = format_preferences_for_prompt(current_prefs)

    return (
        f"Preferencias actuales:\n{prefs_text}\n\n"
        f"Resultado del análisis de esta conversación:\n{summary}"
    )


# ---------------------------------------------------------------------------


def normalize_history(history):
    messages = []
    for item in history or []:
        if isinstance(item, tuple) and len(item) == 2:
            user_msg, bot_msg = item
            messages.append(HumanMessage(content=str(user_msg)))
            messages.append(AIMessage(content=str(bot_msg)))
        elif isinstance(item, dict):
            role = item.get("role")
            content = item.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=str(content)))
            elif role == "assistant":
                messages.append(AIMessage(content=str(content)))
    return messages


def chat_fn(message, history):
    chat_history = normalize_history(history)

    rewrite_result = llm.invoke(
        contextualize_q_prompt.format_messages(
            chat_history=chat_history,
            input=message
        )
    )
    standalone_question = rewrite_result.content.strip()

    print("\nStandalone question:", standalone_question)

    retrieved_docs = retriever.invoke(standalone_question)

    print("\nRetrieved documents:")
    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"\n--- Document {i} ---")
        print(doc.page_content)
        if doc.metadata:
            print("Metadata:", doc.metadata)

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Inyectamos las preferencias de largo plazo aprendidas hasta ahora.
    preferences_text = format_preferences_for_prompt(get_current_preferences())

    final_messages = qa_prompt.format_messages(
        chat_history=chat_history,
        input=message,
        context=context,
        preferences=preferences_text,
    )

    answer_result = llm.invoke(final_messages)
    answer = answer_result.content.strip()

    return answer


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RAG Chatbot")
    gr.Markdown(
        "Chatbot con memoria conversacional y recuperación de contexto desde PDF. "
        "Incluye memoria de largo plazo de preferencias del usuario."
    )

    chat_interface = gr.ChatInterface(
        fn=chat_fn,
        examples=[
            "Hi, do you know who Marcelo is?",
            "What is the price of the Europe Explorer package?"
        ],
    )

    with gr.Accordion("Memoria de preferencias del usuario", open=False):
        end_conversation_btn = gr.Button(
            "Finalizar conversación y analizar preferencias"
        )
        preferences_output = gr.Textbox(
            label="Estado de preferencias",
            lines=10,
            interactive=False,
        )
        end_conversation_btn.click(
            fn=end_conversation_and_update_preferences,
            inputs=[chat_interface.chatbot],
            outputs=[preferences_output],
        )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())