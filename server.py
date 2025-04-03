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
    "que es un dragon": "Un dragón es una criatura mitológica que aparece en muchas culturas, usualmente descrita como un reptil gigante que escupe fuego y puede volar.",
    "cuantos tipos de dragones existen": "Existen muchos tipos según la mitología: dragones europeos, chinos, japoneses, wyverns, entre otros.",
    "los dragones existen": "Los dragones son criaturas mitológicas, no existen en la realidad, aunque algunas leyendas se inspiraron en fósiles de dinosaurios.",
    "de donde vienen los dragones": "Los dragones aparecen en mitologías de Europa, Asia, Medio Oriente y Mesoamérica. Su origen varía según la cultura.",
    "que comen los dragones": "En las leyendas, los dragones comen ganado, personas o incluso piedras y metales preciosos, según la versión del mito.",
    "como se comunican los dragones": "Algunos mitos dicen que los dragones pueden hablar, otros solo rugen o se comunican telepáticamente.",
    "los dragones pueden volar": "Sí, en la mayoría de las leyendas los dragones tienen alas y pueden volar grandes distancias.",
    "que poderes tiene un dragon": "Los dragones suelen tener aliento de fuego, fuerza descomunal, vuelo, y a veces poderes mágicos o inmortalidad.",
    "cuanto vive un dragon": "En la mitología, los dragones pueden vivir cientos o incluso miles de años.",
    "donde viven los dragones": "Según las historias, habitan montañas, cuevas profundas, bosques mágicos o incluso en otras dimensiones.",
    "que tamano tienen los dragones": "Su tamaño varía, pero normalmente se los representa como enormes, de varios metros de largo y con alas inmensas.",
    "los dragones son buenos o malos": "Depende de la cultura. En Europa suelen ser malvados, mientras que en Asia son símbolos de sabiduría y buena fortuna.",
    "como se mata un dragon": "En las leyendas, los héroes usan espadas mágicas o arpones y atacan puntos débiles como el corazón o debajo de las alas.",
    "hay dragones en harry potter": "Sí, en Harry Potter hay varias especies de dragones, como el Colacuerno Húngaro y el Galés Verde Común.",
    "los dragones escupen fuego": "Sí, en muchas leyendas los dragones pueden lanzar fuego por la boca como arma ofensiva.",
    "que representa un dragon": "Los dragones pueden simbolizar poder, sabiduría, protección o destrucción, según el contexto cultural.",
    "que es un wyvern": "Un wyvern es una criatura parecida a un dragón, pero con solo dos patas y dos alas. Es común en la heráldica europea.",
    "cual es el dragon mas famoso": "Uno de los más famosos es Smaug, de 'El Hobbit' de J.R.R. Tolkien.",
    "hay dragones en la biblia": "Sí, se mencionan criaturas parecidas a dragones en Apocalipsis y otros libros bíblicos, aunque de forma simbólica.",
    "cual es la diferencia entre un dragon oriental y uno occidental": "El dragón oriental es largo, sin alas, sabio y protector. El occidental es alado, escupe fuego y suele ser hostil.",
    "que significa sonar con dragones": "Soñar con dragones puede representar desafíos, poder oculto o transformación, dependiendo del contexto del sueño.",
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