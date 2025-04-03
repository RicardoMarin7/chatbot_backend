import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class Message(BaseModel):
    text: str

nltk.download("punkt_tab")
nltk.download("stopwords")

def preprocesar_texto(texto):
    stop_words = set(stopwords.words("spanish"))
    tokens = word_tokenize(texto.lower())
    tokens_filtrados = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens_filtrados)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Base de datos de preguntas y respuestas
preguntas_respuestas = {
    "horario atención hora": "Nuestro horario es de 9 AM a 6 PM.",
    "ubicación tienda donde": "Nos encontramos en Av. Reforma #123, Ciudad de México.",
    "contacto soporte ayuda": "Para ayuda escribirnos a soporte@empresa.com.",
    "productos disponibles venden": "Tenemos laptops, celulares y accesorios electrónicos.",
    "métodos pago tarjeta": "Aceptamos tarjetas de crédito, débito y pagos en efectivo."
}

# Preprocesamos las preguntas
preguntas_procesadas = [preprocesar_texto(pregunta) for pregunta in preguntas_respuestas.keys()]

# Vectorizamos las preguntas usando TF-IDF
vectorizer = TfidfVectorizer()
matriz_tfidf = vectorizer.fit_transform(preguntas_procesadas)

def chatbot_pln(pregunta):
    pregunta_proc = preprocesar_texto(pregunta)
    print(pregunta_proc)
    vector_pregunta = vectorizer.transform([pregunta_proc])
    print(vector_pregunta)

    # Calculamos la similitud de coseno con todas las preguntas almacenadas
    similidades = cosine_similarity(vector_pregunta, matriz_tfidf)
    
    # Buscamos la pregunta con mayor similitud
    indice_mejor_respuesta = similidades.argmax()
    mejor_pregunta = list(preguntas_respuestas.keys())[indice_mejor_respuesta]

    if similidades[0, indice_mejor_respuesta] > 0.5:
        return preguntas_respuestas[mejor_pregunta]
    else:
        return "Lo siento, no entiendo la pregunta."


@app.post("/chat")
async def chat_response(message: Message):
    response_text = chatbot_pln(message.text)
    return {"response": response_text}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)