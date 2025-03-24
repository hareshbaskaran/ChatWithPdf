import os

from dotenv import load_dotenv

load_dotenv()

### Local Storage Variables and Path configurations ######

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SQL_MANAGER_NAMESPACE = f"PDFChat"

SQLITE_DB_URL = os.getenv("SQLITE_DB_URL", "sqlite:////data/sqlite/chatpdf_sqlmanager.sql")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "/data/vector_store")