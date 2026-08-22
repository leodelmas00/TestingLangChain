Chatbot con Gradio, LangChain y Groq

Este programa implementa una interfaz de chatbot utilizando Gradio, LangChain y Groq. El usuario puede seleccionar un modelo de Groq antes de iniciar la conversación y configurar la longitud y formalidad de las respuestas.

El chatbot utiliza RAG (Retrieval-Augmented Generation) para consultar información contenida en data/SkyRouteTravelAgency.pdf. El documento se divide en fragmentos, se generan embeddings utilizando BAAI/bge-small-en-v1.5 y se almacenan en una base de datos Chroma persistente. Ante cada pregunta, se recuperan los fragmentos más relevantes y se incorporan al prompt enviado al modelo.

Funcionamiento

Al iniciar el programa, se consultan los modelos disponibles en la API de Groq y se filtran aquellos que corresponden a modelos de texto. Al mismo tiempo, se inicializa la base de datos Chroma. Si todavía no existe, se procesa el PDF y se generan los embeddings; si ya existe, se reutiliza la información almacenada.

Antes de comenzar el chat, el usuario selecciona el modelo que desea utilizar. Al confirmar, se crea una única instancia de ChatGroq, que se mantiene durante toda la conversación.

Durante el chat, cada pregunta pasa por el retriever de Chroma para obtener información relevante del PDF. Luego se construye el prompt utilizando el contexto recuperado, el historial de conversación y las configuraciones de longitud y formalidad.

Gradio

Gradio se utiliza como capa de interfaz web. gr.Blocks permite construir la estructura de la aplicación mediante componentes como Row y Column. ChatInterface proporciona la interfaz de conversación y administra el historial del chat.

Los componentes Dropdown y Radio permiten seleccionar el modelo, la longitud y la formalidad. Los eventos, como button.click(), conectan las acciones del usuario con funciones Python.

Finalmente, gr.State permite mantener datos internos entre eventos, como la instancia de ChatGroq seleccionada al comenzar la conversación. De esta manera, Gradio se encarga de la interfaz y la interacción con el usuario, mientras LangChain gestiona el RAG y la comunicación con el modelo.

Instalar: pip install requests