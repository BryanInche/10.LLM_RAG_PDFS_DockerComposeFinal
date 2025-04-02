from langchain_community.llms import Ollama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
#from cargar_solodata import cargar_documentos, crear_vectorstore
from cargar_all_type_data import cargar_documentos, crear_vectorstore
#from langchain_community.vectorstores import PGVector

from langchain.memory import ConversationBufferMemory # Agregar Memoria a tu Chatbot RAG

def iniciar_llm_chat(ruta_files):
    # Inicializar el modelo de lenguaje (Mistral), Generar Texto
    llm_ms4m = Ollama(model="mistral", 
                      base_url="http://ollama:11434",  # Nombre del servicio en Docker
                      temperature=0) #baja (o 0) para respuestas más deterministas 
    
    # Tarea de búsqueda semántica
    # Inicializar el modelo de embeddings (Hugging Face)
    # Modelo multilingüe. Esto significa que puede generar embeddings para textos en inglés y español
    embeding_modelo = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Intentar cargar el vector store existente
    try:
        vs_ms4m = Chroma(
            embedding_function=embeding_modelo,
            persist_directory="4chroma_ms4m_bd_allche",
            collection_name="4data_ms4m_allche"
        )
    except Exception as e:
        print(f"No se pudo cargar el vector store existente: {e}")
        print("Creando un nuevo vector store...")
        # Cargar documentos y crear el vector store
        documentos = cargar_documentos(ruta_files)
        vs_ms4m = crear_vectorstore(documentos)
    
    # Crear el retriever para buscar en el vector store
    retriver_ms4m = vs_ms4m.as_retriever(search_kwargs={"k": 3})
    
    # Definir el prompt personalizado para Responder al usuario
    prompt_template_ms4m = """
    Eres un experto ingeniero de minas de minería subterránea y tajo abierto. Tu tarea es responder preguntas utilizando únicamente la información proporcionada en el contexto. 
    Sigue las siguientes instrucciones:

    1. Si la pregunta está en español y el contexto que obtienes está en inglés, traduce la información del contexto al español y luego genera la respuesta en español.
    2. Si no sabes la respuesta o la información no está en el contexto, responde: "No tengo información suficiente para responder a esa pregunta". No alucines porfavor.
    3. No inventes respuestas. Solo utiliza la información proporcionada en el contexto.

    Contexto: {context}
    Pregunta: {question}

    Responde únicamente en español y de manera profesional, concisa y precisa.
    Respuesta útil:
    """


    
    prompt_1 = PromptTemplate(
        template=prompt_template_ms4m,
        input_variables=["context", "question"]
    )

    # Crear la cadena de RetrievalQA
    cadena_rag = RetrievalQA.from_chain_type(
        llm=llm_ms4m,                      # Modelo de lenguaje (Mistral)
        chain_type="stuff",                # Tipo de cadena ("stuff")
        retriever=retriver_ms4m,           # Retriever para buscar documentos
        return_source_documents=True,      # Devuelve los documentos fuente
        chain_type_kwargs={"prompt": prompt_1}  # Prompt personalizado
    )

    
    return cadena_rag
