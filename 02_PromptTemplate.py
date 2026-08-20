from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Inicializar modelo
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# Crear template
prompt = PromptTemplate.from_template(
    "Answer the following question in a concise way, you must ALWAYS end every response with the word 'miau': {question}"
)

# Formatear prompt
formatted_prompt = prompt.format(
    question="Hi, how are you?"
)

# Invocar modelo
response = llm.invoke(formatted_prompt)

print(response.content)