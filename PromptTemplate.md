**PromptTemplate**: Construye un texto plano.
**ChatPromptTemplate**: Construye una conversación estructurada con roles (system, human, assistant).

Internamente, ChatPromptTemplate genera algo parecido a:
```
SystemMessage(content="You must ALWAYS end every response with the word 'miau'."),
HumanMessage(content="Hi, how are you?")
```
y llm.invoke(messages) envía ambos mensajes al modelo.

Se hace una distincion explicita entre las instrucciones del sistema y el mensaje del usuario.

Mientras que con PromptTemplate, normalmente se hace:
```
prompt = PromptTemplate.from_template(
    "Answer the following question in a concise way, you must ALWAYS end every response with the word 'miau': {question}"
)
```

Eso termina siendo un único string: `Answer the following question in a concise way, you must ALWAYS end every response with the word 'miau': Hi, how are you?`

No existe separación entre instrucciones del sistema y mensaje del usuario.

Por eso ChatPromptTemplate es más potente y más controlable para lo que tiene que ver con chatbots.