from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
VECTOR_DB_PATH = "docs/out_data"
SQL_MANAGER_NAMESPACE = f"FAISS/chatpdf"
SQLITE_DB_URL = "sqlite:///chatpdf_sqlmanager.sql"
