## Qué hace el programa
Un chat por consola con un modelo de Groq (`openai/gpt-oss-20b`). El usuario escribe, la IA responde, y así hasta escribir `chau`.

## Cómo funciona la memoria
- `create_agent()` arma un agente cuyo estado interno guarda automáticamente los mensajes (`messages`).
- `InMemorySaver()` es el **checkpointer**: persiste ese estado en memoria RAM mientras corre el programa.
- `thread_id` identifica la conversación. Todas las invocaciones con el mismo `thread_id` comparten historial; uno distinto arranca una charla nueva desde cero.
- En cada turno solo se manda el **mensaje nuevo** del usuario, el agente recupera el historial del `thread_id` vía el checkpointer y lo agrega antes de llamar al modelo.

Basicamente el checkpointer + `thread_id` reemplazan a la lista manual de mensajes: LangGraph guarda y recupera el historial por vos, en vez de que lo hagas a mano.