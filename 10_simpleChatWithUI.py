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
    model="llama-3.1-8b-instant",
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

qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a chatbot assistant.

Answer ONLY using the retrieved context.
If the answer is not contained in the context,
say you don't have enough information.

Context:
{context}"""
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])


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

    final_messages = qa_prompt.format_messages(
        chat_history=chat_history,
        input=message,
        context=context
    )

    answer_result = llm.invoke(final_messages)
    answer = answer_result.content.strip()

    return answer


demo = gr.ChatInterface(
    fn=chat_fn,
    title="RAG Chatbot",
    description="Chatbot con memoria conversacional y recuperación de contexto desde PDF.",
    examples=[
        "Hi, do you know who Marcelo is?",
        "What is the price of the Europe Explorer package?"
    ],
)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())