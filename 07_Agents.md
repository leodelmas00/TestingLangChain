Un agente es un modelo que llama a herramientas en un bucle hasta que se completa una tarea.

Agent = Model + Harness

Un arnés(Harness) es todo lo que rodea ese bucle: el modelo, su prompt, sus herramientas y cualquier middleware que moldee su comportamiento.

Con un agente:
- No se necesita manejar manualmente tool_calls
- El agente decide cuándo usar tools,
- Ejecuta la tool automáticamente,
- Y genera la respuesta final.