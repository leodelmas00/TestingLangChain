Ejecutar con: uvicorn 09_server:app --reload --port 8000

Una API web que detecta emociones en texto y sugiere estrategias de respuesta construida con FastAPI que:

Analiza el tono/emoción de un texto en inglés.
Identifica la emoción dominante usando un modelo de inteligencia artificial.
Devuelve una estrategia recomendada de cómo responder según la emoción detectada.

Tecnologías utilizadas

FastAPI: Para crear la API.
Hugging Face Transformers: Usa el modelo j-hartmann/emotion-english-distilroberta-base, que clasifica texto en 7 emociones: ira, asco, miedo, alegría, neutral, tristeza y sorpresa

Endpoints
GET/ : Muestra estado básico de la API
POST/detect-frustration : Analiza el texto y devuelve emoción + estrategia

Ejemplo de uso
Request:
JSON{
  "text": "This service is terrible, I'm furious!"
}

Response:
JSON{
  "label": "anger",
  "score": 0.987,
  "strategy": "Respond with calm empathy. Acknowledge the frustration and offer a clear solution."
}

Se creo para usar como Tool en OpenWebUI