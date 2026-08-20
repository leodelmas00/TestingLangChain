from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# Memoria de corto plazo: el checkpointer guarda el historial
# de la conversación, asociado a un thread_id
checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="You must ALWAYS end every response with the word 'miau'.",
    checkpointer=checkpointer,
)

# Todos los turnos de esta sesión comparten el mismo thread_id,
# así el agente recuerda lo que se dijo antes
config = {"configurable": {"thread_id": "1"}}

while True:

    user_input = input("Yo: ")

    if user_input.lower() == "chau":
        break

    # El agente ya mantiene el historial internamente vía el checkpointer,
    # solo hay que mandar el mensaje nuevo
    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
    )

    print("\nAI:", response["messages"][-1].content)
    print()