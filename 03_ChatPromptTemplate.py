from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# Prompt estructurado
prompt = ChatPromptTemplate.from_messages([
    ("system", "You must ALWAYS end every response with the word 'miau'."),
    ("human", "{question}")
])

# Crear mensajes
messages = prompt.invoke({
    "question": "Hi, how are you?"
})

# Llamar al modelo
response = llm.invoke(messages)

print(response.content)