from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# Historial conversacional
chat_history = [
    SystemMessage(
        content="You must ALWAYS end every response with the word 'miau'."
    )
]

while True:

    user_input = input("Yo: ")

    if user_input.lower() == "chau":
        break

    # Agregar mensaje humano al historial
    chat_history.append(HumanMessage(content=user_input))

    # Invocar modelo con todo el historial
    response = llm.invoke(chat_history)

    # Mostrar respuesta
    print("\nAI:", response.content)
    print()

    # Agregar respuesta de la IA al historial
    chat_history.append(AIMessage(content=response.content))