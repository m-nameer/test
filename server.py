from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import FastAPI
import numpy as np
import faiss
import uvicorn
import threading
import fitz
import numpy as np



from openai import OpenAI



# embedding_model = SentenceTransformer("BAAI/bge-m3", device="cpu")


def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text += page.get_text()
    return text

# Function to split text into chunks with overlap
def split_text_into_chunks_with_overlap(text, chunk_size=200, overlap=50):
    words = text.split()

    # total_words = len(words)

    # chunk_size = total_words // 10
    # overlap = chunk_size // 4


    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks




def make_embeddings(corpus):

    # doc_response = client.embeddings.create(
    # input=corpus,
    # model="text-embedding-3-small"
    # )
    # doc_embeddings = [data.embedding for data in doc_response.data]

    test = np.array([[1,2,3],[4,5,6],[7,8,9]])
    print(test.shape[1])
    print(test)
    index = faiss.IndexFlatL2(test.shape[1])
    index.add(test)

    # corpus_embeddings_np = np.array(doc_embeddings)
    # index = faiss.IndexFlatL2(corpus_embeddings_np.shape[1])
    # index.add(corpus_embeddings_np)




    # corpus_embeddings = embedding_model.encode(corpus, convert_to_tensor=True)

    # corpus_embeddings_np = corpus_embeddings.cpu().numpy()

    # index = faiss.IndexFlatL2(corpus_embeddings_np.shape[1])
    # index.add(corpus_embeddings_np)

    return index


pdf_path = "./product.pdf"  # Update with actual PDF path
pdf_text = extract_text_from_pdf(pdf_path)
text_chunks = split_text_into_chunks_with_overlap(pdf_text, chunk_size=200, overlap=50)



def perform_rag(query, index, top_k, text_chunks):

    # query_response = client.embeddings.create(
    # input=[query],
    # model='text-embedding-3-small'
    # )
    # query_embeddings = [data.embedding for data in query_response.data]
    query_embeddings = np.array([[1,2,3],[4,5,6],[7,8,9]])
    distances, indices = index.search(query_embeddings, k=top_k)
    print(distances,indices)
    top_matches = [text_chunks[idx] for idx in indices[0]]





    # query_embedding = embedding_model.encode([query], convert_to_tensor=True).cpu().numpy()
  
    # distances, indices = index.search(query_embedding, k=top_k)
    # top_matches = [text_chunks[idx] for idx in indices[0]]

    return top_matches


vectorStore = make_embeddings(text_chunks)

print(perform_rag("hello", vectorStore, 3, text_chunks))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_chroma(request: QueryRequest):
    # Perform the query on your ChromaDB collection
    results = perform_rag(request.query, vectorStore, 3, text_chunks)
    
    return {"results": results}


def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Run the FastAPI app in a background thread
thread = threading.Thread(target=run_api)
thread.start()