from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.messages import (SystemMessage,HumanMessage)

load_dotenv()

# === LLM ===

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# === Cargar PDF ===

loader = PyPDFLoader("data/Telefononica.pdf")

documents = loader.load()

# === Dividir documento ===

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = text_splitter.split_documents(documents)

# === Embeddings ===

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# === Vector DB ===

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

while True:

    question = input("\nPregunta: ")

    if question.lower() == "chau":
        break

    # Buscar contexto relevante
    docs = retriever.invoke(question)

    # Unir chunks encontrados
    context = "\n\n".join([doc.page_content for doc in docs])

    messages = [
        SystemMessage(
            content="""You are a helpful assistant.
            Use ONLY the retrieved context to answer.
            If the answer is not present, say you don't know."""),
            SystemMessage(content=f"""Retrieved Context:{context}"""),
            HumanMessage(content=question)
        ]

    # Llamar LLM
    response = llm.invoke(messages)

    print("\nAI:", response.content)



"""
El programa implementa un pipeline RAG básico donde un PDF se divide en fragmentos,
se convierte en embeddings y se almacena en una base vectorial para permitir
búsqueda semántica.

Cuando el usuario hace una pregunta, el sistema recupera los fragmentos más relevantes
del documento y los inyecta en el contexto antes de enviar la consulta a la LLM para
generar la respuesta.

Basicamente:

1. El PDF se carga.
2. Se divide en fragmentos.
3. Cada fragmento se convierte en embeddings.
5. Chroma almacena esos embeddings.
5. Cuando el usuario pregunta:
    i. se busca similitud semántica,
    ii. se recuperan chunks relevantes,
    iii. esos chunks se inyectan en el prompt.
6. El LLM responde usando ese contexto.

Random, pero probe hacerlo tanto con PrompTemplate pelado
como con ChatPromptTemplate y este ultimo esta mas perdido,
no genera buenas respuestas, no se si sera por el modelo
o por la naturaleza de mandar los mensajes separados
(en System,Human,etc), anda a saber, pero bueno, por ahora 
PromptTemplate comun es mas copado con las respuestas, pone
menos 'peros'.
"""