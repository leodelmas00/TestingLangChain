from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_exa import ExaSearchResults
from langchain_core.messages import HumanMessage, ToolMessage

# Cargar variables
load_dotenv()

# Tool búsqueda web
search_tool = ExaSearchResults()

# Modelo
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# Vincular tool
llm_with_tools = llm.bind_tools([search_tool])

# Mensaje usuario
messages = [
    HumanMessage(
        content="Search the web: Did Akira Toriyama die?, and then tell me if akira toriyama is still alive"
    )
]

# Primera respuesta
response = llm_with_tools.invoke(messages)

# Verificar tool call
if response.tool_calls:

    tool_call = response.tool_calls[0]

    tool_result = search_tool.invoke(tool_call["args"])
    #print(tool_result)

    # Recortar texto
    tool_result = str(tool_result)[:1000]

    # Agregar historial
    messages.append(response)

    messages.append(
        ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call["id"]
        )
    )

    # Respuesta final
    final_response = llm_with_tools.invoke(messages)

    print(final_response.content)

else:
    print(response.content)


"""
El conocimiento del modelo que estoy usando alcanza hasta diciembre de 2023, por lo tanto,
usando PrebuiltTools de Search online vamos a tratar de que encuentre informacion sobre el fallecimiento
de Akira Toriyama en marzo de 2024.

Vamos a ver si sale con Exa Search.

Efectivamente funciona, le preguntamos a la llm: Search the web: Did Akira Toriyama die?, and then tell me if akira toriyama is still alive

Y esta respondio: Based on the search results, it appears that Akira Toriyama has passed away. , Algo que no
diria si no fuese por el uso de la Tool.

Pequeño dato, pero se tuvo que recortar la cantidad de informacion dada a la llm luego de la busqueda de ~40k tokens a
1000, ya que mi limite era de 6000 tokens.
"""