import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Importa el middleware CORS
from pydantic import BaseModel
from chat2_llm import iniciar_llm_chat  # Importa tu función principal

# Inicializa la aplicación FastAPI
app = FastAPI()

# Configura CORS : Permite o bloquea solicitudes de diferentes orígenes (ejemplo del Frontend)
app.add_middleware(
    CORSMiddleware,
    #allow_origins=["http://localhost:3000"],  # Permite solicitudes desde este origen
    #allow_origins=["http://localhost:3000", "http://192.168.14.24:3000"],  # Agrega aquí todas las IPs necesarias
    allow_origins=["http://localhost:3000","http://frontend:3000"],  # Nombre del servicio en Docker
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los headers
)

# Define el modelo de entrada para la API
class QueryRequest(BaseModel):
    question: str

# Inicializa el modelo RAG (esto se ejecuta una vez al iniciar la API)
ruta_files = [
    #"pdf_solos/ASIS_REPORTE_EJECUTIVO.pdf",  # Ruta a un archivo PDF
    "ppt_solos",  # Ruta a una carpeta de PPTs
    "word_solos",  # Ruta a una carpeta de Words
    "pdf_carpetas/pdfs_varios"  # Ruta a una carpeta de PDFs
]
cadena_rag = iniciar_llm_chat(ruta_files)

# Define RAG sin memoria, el endpoint para recibir preguntas
@app.post("/respuesta_llm")
def ask_question(query: QueryRequest):
    try:
        # Obtén la respuesta del modelo RAG
        respuesta = cadena_rag.invoke({"query": query.question})
        return {
            "response": respuesta["result"],
            "source_documents": [
                {
                    "page": doc.metadata["page"],
                    "source": doc.metadata["source"]
                }
                for doc in respuesta["source_documents"]
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Ejecuta la API con Uvicorn
if __name__ == "__main__":
    #host="127.0.0.1" → La API solo es accesible desde la misma máquina (localhost).
    #host="0.0.0.0" → La API es accesible desde cualquier dispositivo en la red: Util cuando : 
    # 1.Quieres acceder a la API desde otro equipo en la misma red
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
