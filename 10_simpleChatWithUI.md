## Qué hace
Este código crea un chatbot con interfaz web que responde usando información de un PDF mediante RAG, o sea, busca contexto relevante y luego genera la respuesta con un LLM.

Además, guarda el historial de la conversación para que las preguntas posteriores se entiendan mejor.

## Cómo funciona
Primero convierte el PDF en fragmentos, después los transforma en vectores con embeddings y los guarda en una base vectorial para poder buscarlos por similitud.

Cuando el usuario pregunta algo, el sistema reescribe la pregunta usando el historial, recupera los documentos más relevantes con `retriever.invoke()`, arma el prompt final y genera la respuesta.

## Errores y soluciones
1. get_relevant_documents() dio error porque en la versión actual del retriever se usa invoke() para recuperar documentos.
2. theme="soft" en gr.ChatInterface falló porque ese parámetro no va ahí; se corrigió moviendo el tema a launch().
3. El historial dio ValueError porque Gradio no siempre lo entrega como lista de tuplas; se resolvió normalizándolo para aceptar tuplas o diccionarios con role y content.
4. La advertencia de HuggingFaceEmbeddings se solucionó usando langchain_huggingface en lugar de langchain_community.