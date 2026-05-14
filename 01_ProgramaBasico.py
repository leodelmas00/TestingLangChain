from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Cargar variables del .env
load_dotenv()

# Crear el modelo
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# Enviar prompt
response = llm.invoke("Hi, how are you?")

# Mostrar respuesta
print(response.content)