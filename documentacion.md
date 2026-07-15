# TestingLangChain - Documentación Completa

## Descripción General

Proyecto de aprendizaje y experimentación con **LangChain**, un framework para construir aplicaciones basadas en modelos de lenguaje (LLMs). El proyecto explora diferentes funcionalidades de LangChain, desde conexiones básicas con proveedores hasta la implementación de RAG, tools personalizadas, agentes y análisis de sentimiento.

## Estructura del Proyecto

```
TestingLangChain/
├── 01_ProgramaBasico.py          # Conexión básica con LLM
├── 02_PromptTemplate.py          # Uso de templates de prompt
├── 03_ChatPromptTemplate.py      # Templates para chat con roles
├── 04_ShortTermMemory.py         # Memoria de corto plazo en conversaciones
├── 05_IntentoDeRAG.py            # Implementación de RAG con PDF
├── 06_CustomTool.py              # Creación de tools personalizadas
├── 06_PrebuiltTools.py           # Uso de tools preconstruidas (búsqueda web)
├── 07_Agent.py                   # Implementación de agente con herramientas
├── 08_SentimentAnalysis.py       # Análisis de sentimiento y respuesta adaptativa
├── server.py                     # API REST con FastAPI
├── data/                         # Directorio para documentos PDF
├── requirements.txt              # Dependencias del proyecto
├── .env                          # Variables de entorno (API keys)
└── *.md                          # Documentación adicional
```

## Dependencias

### Archivo `requirements.txt`
```
fastapi
uvicorn
pydantic
transformers
langchain-groq
langchain-core
python-dotenv
```

### Instalación Completa

```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias base
pip install langchain langchain-groq python-dotenv

# Para RAG (script 05)
pip install langchain-community chromadb pypdf sentence-transformers

# Para búsqueda web (script 06)
pip install langchain-exa

# Para análisis de sentimiento (script 08)
pip install transformers

# Para API REST (server.py)
pip install fastapi uvicorn pydantic
```

### Variables de Entorno

El archivo `.env` debe contener:
```
GROQ_API_KEY="tu_api_key_de_groq"
EXA_API_KEY="tu_api_key_de_exa"
```

---

## Scripts y Funcionalidades

### 01_ProgramaBasico.py

**Propósito:** Conexión básica con un proveedor de LLM (Groq) usando LangChain.

**Qué hace:**
1. Carga las variables de entorno desde `.env`
2. Inicializa el modelo `llama-3.1-8b-instant` con temperatura 0
3. Envía un prompt simple: "Hi, how are you?"
4. Muestra la respuesta del modelo

**Cómo ejecutar:**
```bash
python 01_ProgramaBasico.py
```

**Resultado esperado:** Respuesta del modelo LLM a la pregunta "Hi, how are you?"

**Conceptos clave:**
- Uso de `ChatGroq` como wrapper de LangChain para Groq
- Carga de variables de entorno con `dotenv`
- El modelo está acoplado al proveedor Groq (cambiar proveedor requiere cambiar import, clase y API key)
- Parámetros configurables: `temperature`, `max_tokens`, `top_p`, `timeout`, `max_retries`, `streaming`, `seed`

**Nota del autor:** La abstracción de LangChain no es total; al cambiar de proveedor se debe modificar el import, la clase del modelo, posiblemente algunos parámetros y la API key. La lógica general (`invoke`) se mantiene igual.

---

### 02_PromptTemplate.py

**Propósito:** Demonstración de `PromptTemplate` para construir prompts reutilizables.

**Qué hace:**
1. Crea un template de texto plano con una variable `{question}`
2. Formatea el template con una pregunta específica
3. Envía el prompt formateado al LLM
4. El modelo debe terminar cada respuesta con la palabra "miau"

**Cómo ejecutar:**
```bash
python 02_PromptTemplate.py
```

**Resultado esperado:** Respuesta del modelo que termina con la palabra "miau"

**Conceptos clave:**
- `PromptTemplate.from_template()` construye texto plano
- No hay separación entre instrucciones del sistema y mensaje del usuario
- El resultado es un único string concatenado
- Útil para prompts simples sin necesidad de roles diferenciados

---

### 03_ChatPromptTemplate.py

**Propósito:** Demonstración de `ChatPromptTemplate` para construir conversaciones estructuradas con roles.

**Qué hace:**
1. Crea un template con mensajes estructurados (system y human)
2. El mensaje de sistema instruye al modelo para que termine con "miau"
3. Invoca el template con una pregunta
4. Envía los mensajes estructurados al LLM

**Cómo ejecutar:**
```bash
python 03_ChatPromptTemplate.py
```

**Resultado esperado:** Respuesta del modelo que termina con "miau"

**Conceptos clave:**
- `ChatPromptTemplate.from_messages()` crea mensajes con roles definidos
- Genera objetos `SystemMessage` y `HumanMessage` internamente
- Permite separación explícita entre instrucciones del sistema y mensajes del usuario
- Más potente y controlable para chatbots

**Comparación con PromptTemplate:**
- `PromptTemplate`: texto plano, sin roles
- `ChatPromptTemplate`: mensajes estructurados con roles (system, human, assistant)

**Observaciones del autor:** En pruebas, `PromptTemplate` obedecía mejor la regla de terminar con "miau" que `ChatPromptTemplate`.

---

### 04_ShortTermMemory.py

**Propósito:** Implementación de memoria de corto plazo en conversaciones con el LLM.

**Qué hace:**
1. Inicializa una lista `chat_history` con un mensaje de sistema
2. En un bucle continuo:
   - Recibe input del usuario
   - Agrega el mensaje humano al historial
   - Invoca al LLM con todo el historial completo
   - Muestra la respuesta y la agrega al historial
3. El bucle termina cuando el usuario escribe "chau"

**Cómo ejecutar:**
```bash
python 04_ShortTermMemory.py
```

**Resultado esperado:** Conversación interactiva donde el modelo recuerda el contexto anterior

**Conceptos clave:**
- Uso de `HumanMessage`, `AIMessage` y `SystemMessage`
- El historial se envía completo en cada invocación
- No hay persistencia de memoria entre ejecuciones
- Limitación: el contexto crece indefinidamente (hasta el límite del modelo)

---

### 05_IntentoDeRAG.py

**Propósito:** Implementación de un pipeline RAG (Retrieval-Augmented Generation) básico con documentos PDF.

**Qué hace:**
1. Carga un PDF usando `PyPDFLoader`
2. Divide el documento en fragmentos (chunks) de 1000 caracteres con overlap de 200
3. Genera embeddings usando HuggingFace (`all-MiniLM-L6-v2`)
4. Almacena los embeddings en Chroma (base de datos vectorial)
5. Crea un retriever que busca los 3 chunks más relevantes
6. En un bucle interactivo:
   - Recibe una pregunta del usuario
   - Busca chunks relevantes por similitud semántica
   - Inyecta el contexto encontrado en el prompt
   - Envía la consulta al LLM
   - Muestra la respuesta

**Cómo ejecutar:**
```bash
python 05_IntentoDeRAG.py
```

**Resultado esperado:** Respuestas del modelo basadas en el contenido del PDF cargado

**Documentos requeridos:**
- Colocar PDFs en la carpeta `data/`
- El script usa `data/Telefononica.pdf` como ejemplo

**Conceptos clave:**
- **Document Loading**: Carga de documentos PDF con `PyPDFLoader`
- **Text Splitting**: División en chunks con `RecursiveCharacterTextSplitter`
- **Embeddings**: Conversión de texto a vectores numéricos con HuggingFace
- **Vector Store**: Almacenamiento y búsqueda semántica con Chroma
- **Retrieval**: Recuperación de documentos relevantes
- **Context Injection**: Inyección de contexto recuperado en el prompt

**Pipeline RAG:**
1. PDF → Carga del documento
2. Documento → División en chunks
3. Chunks → Generación de embeddings
4. Embeddings → Almacenamiento en Chroma
5. Pregunta del usuario → Búsqueda semántica
6. Chunks relevantes → Inyección en prompt
7. Prompt + Contexto → Respuesta del LLM

**Observaciones del autor:** El script funciona pero las respuestas pueden variar según el modelo y la estructura del PDF.

---

### 06_CustomTool.py

**Propósito:** Demonstración de cómo crear y usar tools personalizadas con LangChain.

**Qué hace:**
1. Define una tool personalizada `calculator` usando el decorador `@tool`
2. La tool ejecuta una expresión matemática y retorna el resultado + 1 (error intencional para verificar uso)
3. Vincula la tool al modelo con `llm.bind_tools()`
4. Envía un mensaje al LLM
5. Verifica si el modelo decidió usar la tool
6. Si la usa, ejecuta la tool, agrega el resultado al historial y llama al modelo nuevamente
7. Si no la usa, muestra la respuesta directa

**Cómo ejecutar:**
```bash
python 06_CustomTool.py
```

**Resultado esperado:** El modelo decide si necesita usar la calculadora y responde accordingly

**Conceptos clave:**
- Decorador `@tool` para definir tools personalizadas
- El docstring de la función se convierte en la descripción de la tool
- Los type hints son obligatorios (definen el esquema de entrada)
- `llm.bind_tools()` vincula las tools al modelo
- `tool_calls` en la respuesta indica si el modelo usó alguna tool
- `ToolMessage` agrega el resultado de la tool al historial

**Flujo de ejecución:**
1. Usuario envía mensaje
2. LLM analiza si necesita usar tools
3. Si necesita tool → retorna `tool_calls`
4. Se ejecuta la tool manualmente
5. Se agrega resultado al historial
6. Se hace segunda llamada al LLM con el resultado
7. LLM genera respuesta final

---

### 06_PrebuiltTools.py

**Propósito:** Demonstración del uso de tools preconstruidas (búsqueda web con Exa).

**Qué hace:**
1. Inicializa la tool de búsqueda web `ExaSearchResults`
2. Vincula la tool al modelo
3. Envía una pregunta que requiere información actualizada
4. Ejecuta la búsqueda web si el modelo la solicita
5. Recorta los resultados a 1000 caracteres (límite de tokens)
6. Envía los resultados al LLM para generar una respuesta

**Cómo ejecutar:**
```bash
python 06_PrebuiltTools.py
```

**Resultado esperado:** Respuesta del modelo con información actualizada de la web

**Conceptos clave:**
- `ExaSearchResults` como tool preconstruida para búsqueda web
- Necesita API key de Exa en `.env`
- Los resultados de búsqueda pueden ser muy grandes (se recortan)
- El modelo puede acceder a información fuera de su fecha de entrenamiento

**Observaciones del autor:** El conocimiento del modelo alcanza hasta diciembre de 2023. Con la tool de búsqueda web, el modelo pudo encontrar información sobre el fallecimiento de Akira Toriyama en marzo de 2024, algo que no sabía sin la tool.

---

### 07_Agent.py

**Propósito:** Implementación de un agente simple que usa herramientas de búsqueda.

**Qué hace:**
1. Inicializa el modelo y la tool de búsqueda `ExaSearchRetriever`
2. Realiza una búsqueda con Exa directamente
3. Recorta los resultados a 1500 caracteres
4. Construye un prompt con los resultados de búsqueda
5. Envía el prompt al LLM para generar una respuesta

**Cómo ejecutar:**
```bash
python 07_Agent.py
```

**Resultado esperado:** Respuesta del modelo basada en resultados de búsqueda web

**Conceptos clave:**
- Diferencia entre agentes y uso manual de tools
- **Agente**: Modelo + Harness (bucle automático de tool calls)
- **Uso manual**: El desarrollador maneja los tool_calls explícitamente
- Los agentes deciden cuándo usar tools, las ejecutan y generan la respuesta final

**Nota del autor:** Este script es una implementación simplificada. Un agente completo usaría un bucle automático con `create_react_agent` o similar.

---

### 08_SentimentAnalysis.py

**Propósito:** Análisis de sentimiento del usuario y adaptación de la respuesta del LLM.

**Qué hace:**
1. Carga un modelo de análisis de sentimiento de HuggingFace
2. Define un diccionario de estrategias de respuesta para cada emoción
3. En un bucle interactivo:
   - Recibe input del usuario
   - Detecta la emoción del texto
   - Selecciona la estrategia de respuesta correspondiente
   - Envía el mensaje con la estrategia al LLM
   - Muestra la emoción detectada y la respuesta

**Cómo ejecutar:**
```bash
python 08_SentimentAnalysis.py
```

**Resultado esperado:** El LLM responde adaptándose emocionalmente al tono del usuario

**Modelo de análisis:** `j-hartmann/emotion-english-distilroberta-base`

**Emociones detectadas:**
1. Anger (Enojo)
2. Disgust (Asco)
3. Fear (Miedo)
4. Joy (Alegría)
5. Neutral (Neutral)
6. Sadness (Tristeza)
7. Surprise (Sorpresa)

**Conceptos clave:**
- Pipeline de `transformers` para análisis de sentimiento
- El diccionario `emotion_responses` mapea emociones a instrucciones
- La instrucción se inyecta en el `SystemMessage`
- El modelo adapta su tono según la emoción detectada

---

### server.py

**Propósito:** API REST con FastAPI que combina análisis de sentimiento con chat adaptativo.

**Qué hace:**
1. Define dos endpoints:
   - `POST /detect-frustration`: Detecta la emoción de un texto
   - `POST /chat`: Envía un mensaje con análisis de sentimiento y genera una respuesta
2. Usa el mismo modelo de análisis de sentimiento que el script 08
3. Incluye historial de conversación en el endpoint de chat
4. Responde con la emoción detectada, la estrategia aplicada y la respuesta del LLM

**Cómo ejecutar:**
```bash
uvicorn server:app --reload
```

**Endpoints:**

#### `GET /`
```json
{
  "status": "ok",
  "docs": "/docs",
  "openapi": "/openapi.json"
}
```

#### `POST /detect-frustration`
Request:
```json
{
  "text": "I'm so angry about this!"
}
```
Response:
```json
{
  "label": "anger",
  "score": 0.85,
  "strategy": "Respond with calm empathy. Acknowledge the frustration and offer a clear solution."
}
```

#### `POST /chat`
Request:
```json
{
  "message": "I'm feeling sad today",
  "conversation_history": [
    {"role": "user", "content": "Hello"}
  ]
}
```
Response:
```json
{
  "emotion": "sadness",
  "strategy": "Respond gently and supportively.",
  "response": "I'm sorry to hear you're feeling sad..."
}
```

**Conceptos clave:**
- FastAPI como framework web
- Pydantic para validación de modelos de datos
- Integración de análisis de sentimiento con LLM
- Historial de conversación en la API
- Documentación automática con Swagger (`/docs`)

---

## Conceptos Generales de LangChain

### Proveedores de LLM
LangChain abstrae diferentes proveedores bajo una interfaz uniforme. Para cambiar de proveedor se necesita:
- Cambiar el import (ej: `langchain_groq` → `langchain_openai`)
- Cambiar la clase del modelo (ej: `ChatGroq` → `ChatOpenAI`)
- Configurar la API key correspondiente

### Templates
- **PromptTemplate**: Texto plano sin roles
- **ChatPromptTemplate**: Mensajes estructurados con roles (system, human, assistant)

### Memoria
- La memoria de corto plazo se implementa manteniendo una lista de mensajes
- El historial se envía completo en cada invocación
- No hay persistencia automática entre sesiones

### RAG (Retrieval-Augmented Generation)
Pipeline para responder preguntas basándose en documentos propios:
1. Carga de documentos
2. División en chunks
3. Generación de embeddings
4. Almacenamiento en base vectorial
5. Búsqueda semántica
6. Inyección de contexto en el prompt

### Tools
- Funciones ejecutables que el modelo puede invocar
- Se definen con el decorador `@tool`
- El modelo decide cuándo usarlas según el contexto
- Se vinculan al modelo con `bind_tools()`

### Agentes
- Modelos que ejecutan tools en un bucle hasta completar una tarea
- Componen: Modelo + Harness (bucle, tools, middleware)
- Deciden automáticamente cuándo y qué tools usar

---

## Observaciones y Notas

### Sobre PromptTemplate vs ChatPromptTemplate
En pruebas realizadas, `PromptTemplate` obedecía mejor la instrucción de terminar con "miau" que `ChatPromptTemplate`. Esto puede deberse a la naturaleza de enviar mensajes separados en roles.

### Sobre RAG
El pipeline RAG funciona pero las respuestas pueden variar. Con `PromptTemplate` las respuestas son más directas, mientras que con `ChatPromptTemplate` el modelo puede ser menos preciso.

### Sobre el conocimiento del modelo
El modelo `llama-3.1-8b-instant` tiene conocimiento hasta diciembre de 2023. Para información más reciente se necesitan tools de búsqueda web.

### Sobre límites de tokens
Al usar tools de búsqueda web, los resultados pueden ser muy grandes (~40k tokens). Es necesario recortarlos para evitar exceder los límites del modelo (~6000 tokens).

---

## Fuentes y Referencias

- Documentación General: https://docs.langchain.com/oss/python/langchain/overview
- Mensajes: https://docs.langchain.com/oss/python/langchain/messages
- Tools: https://docs.langchain.com/oss/python/langchain/tools
- Integraciones de Tools: https://docs.langchain.com/oss/python/integrations/tools
- Exa Search: https://docs.langchain.com/oss/python/integrations/tools/exa_search
- HuggingFace Course: https://huggingface.co/learn/llm-course/es/chapter1/3
- Modelo de Sentimiento: https://huggingface.co/j-hartmann/emotion-english-distilroberta-base