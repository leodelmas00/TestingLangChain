from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

load_dotenv()

# Definimos la Tool
@tool
def calculator(expression: str) -> str:
    """Performs arithmetic calculations."""
    return str(eval(expression) + 1) #Devuelve un resultado erroneo A PROPOSITO para verificar si la llm efectivamente usa la tool.

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# Vincular tools
llm_with_tools = llm.bind_tools([calculator])

messages = [
    HumanMessage(content="Who date is your model??"),
    SystemMessage(content="""You are a helpful assistant.
                  If a tool was used, explicitly mention it. 
                  If no tool was used, explicitly say so.
                  """)
]

# Primera respuesta IA
response = llm_with_tools.invoke(messages)

# Verificar si hubo tool call
if response.tool_calls:

    print("La IA usó una tool\n")

    # Obtener tool call
    tool_call = response.tool_calls[0]

    # Ejecutar tool
    tool_result = calculator.invoke(tool_call["args"])

    print("Resultado tool: "+tool_result)

    # Agregar mensajes al historial
    messages.append(response)

    messages.append(
        ToolMessage(
            content=tool_result,
            tool_call_id=tool_call["id"]
        )
    )

    # Segunda llamada al modelo
    final_response = llm_with_tools.invoke(messages)

    print("\nRespuesta final IA:")
    print(final_response.content)

else:
    print("La IA NO usó tools\n")
    print(response.content)