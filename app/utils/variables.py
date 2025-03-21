import os

from dotenv import load_dotenv

load_dotenv()

### Local Storage Variables and Path configurations ######

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

SQL_MANAGER_NAMESPACE = f"FAISS/chatpdf"

SQLITE_DB_URL = "sqlite:///chatpdf_sqlmanager.sql"
VECTOR_DB_PATH = "docs/out_data"
