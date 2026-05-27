## Que son las Tools?

Las tools amplían lo que los agentes pueden hacer, permitiéndoles obtener datos en tiempo real, ejecutar código, consultar bases de datos externas y realizar acciones en el mundo real.

Internamente, las tools son funciones ejecutables con entradas y salidas bien definidas que se pasan al modelo de chat.

**El modelo decide cuándo invocar una tool según el contexto de la conversación y qué argumentos proporcionar.**

## Crear Tools

La forma más sencilla de crear una tool es usando el decorador @tool. Por defecto, **el docstring de la función se convierte en la descripción de la tool, ayudando al modelo a entender cuándo debe utilizarla.**

Los type hints son obligatorios porque definen el esquema de entrada de la tool. El docstring debe ser informativo y conciso para ayudar al modelo a comprender el propósito de la tool.

Algunos modelos de chat incluyen tools integradas (como búsquedas web o intérpretes de código) que se ejecutan del lado del servidor.

Por defecto, el nombre de la tool proviene del nombre de la función. Puedes sobrescribirlo cuando necesites algo más descriptivo:

`@tool("web_search")`

También puedes sobrescribir la descripción generada automáticamente para darle instrucciones más claras al modelo:

`@tool("calculator", description="Performs arithmetic calculations. Use this for any math problems.")`


---

## Fuentes
26/5/2026: https://docs.langchain.com/oss/python/langchain/tools#server-side-tool-use
26/5/2026: https://docs.langchain.com/oss/python/integrations/tools
26/5/2026: https://docs.langchain.com/oss/python/integrations/tools/exa_search