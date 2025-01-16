from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.services.embeddings import HFEmbeddings
from app.services.loaders import PDFLoader
from app.services.chunkers import RTChunker
from app.services.vectordbs import FAISSVectorStore
from app.utils.variables import VECTOR_DB_PATH
from utils.helpers import handle_temp_dir
import os

#todo: comment and black style

app = FastAPI()


class PDFUploadResponse(BaseModel):
    message: str
    #similarity_results: Optional[list] = None


@app.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):

    temp_file_path = handle_temp_dir(file)

    try:
        # save uploded file temproarily
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())

        # Process the PDF
        pdf_loader = PDFLoader(doc_path=temp_file_path)
        doc_chunker = RTChunker(docs=pdf_loader.get_docs())
        docs = doc_chunker.split_docs()

        # Init embeddings
        embeddings = HFEmbeddings().get_embeddings()

        # load vector store
        db = FAISSVectorStore(
            docs=docs,
            embeddings=embeddings,
            vector_db_path=VECTOR_DB_PATH,
        )

        #todo : handle indexes with langchain.indexes library using custom IndexManager -> Docs,SQL's
        try:
            # Try to load existing index
            get_db = db.get_vdb()
        except Exception as e:
            # if fails, create new index
            print(f"Creating new FAISS index at {VECTOR_DB_PATH}")
            db.add_docs_to_vector_db()
            get_db = db.get_vdb()


        """
        similarity_results = get_db.similarity_search("vectors", k=5)
        print(similarity_results)"""

        return PDFUploadResponse(
            message="PDF uploaded and processed successfully.",
            #similarity_results=similarity_results,
        )

    ## add extensions
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)