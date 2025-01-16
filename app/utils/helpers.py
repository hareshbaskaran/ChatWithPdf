import os
from app.utils.variables import VECTOR_DB_PATH
from fastapi import File
def handle_temp_dir(file):
    os.makedirs("docs", exist_ok=True)
    os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)

    return f"docs/{file.filename}"