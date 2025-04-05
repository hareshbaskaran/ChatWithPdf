import os
import asyncio
import aiohttp
from pathlib import Path

API_URL = "http://fastapi:8000/upload"  # Change to your actual API endpoint


async def upload_file(pdf_path: Path, bib_path: Path, domain: str):
    """Uploads a PDF and its corresponding BibTeX file to the FastAPI API."""
    try:
        async with aiohttp.ClientSession() as session:
            with pdf_path.open("rb") as pdf_file, bib_path.open("rb") as bib_file:
                files = {
                    "file": (pdf_path.name, pdf_file, "application/pdf"),
                    "bib_file": (bib_path.name, bib_file, "application/x-bibtex"),
                }
                data = {"domain": domain}

                async with session.post(API_URL, data=data, files=files) as response:
                    response_data = await response.json()
                    print(f"✅ Uploaded {pdf_path.name}: {response_data}")

    except Exception as e:
        print(f"❌ Error uploading {pdf_path.name}: {e}")


async def upload_all_files(root_dir: str):
    """Finds and uploads all PDF & BibTeX pairs to the API."""
    root_path = Path(root_dir)
    tasks = []

    for domain_dir in root_path.iterdir():
        if domain_dir.is_dir():  # Each subdirectory is a domain
            domain = domain_dir.name
            pdf_files = list(domain_dir.glob("*.pdf"))
            bib_files = {bib.stem: bib for bib in domain_dir.glob("*.bib")}  # Match by filename

            for pdf_path in pdf_files:
                bib_path = bib_files.get(pdf_path.stem)
                if bib_path:
                    tasks.append(upload_file(pdf_path, bib_path, domain))
                else:
                    print(f"⚠️ Skipping {pdf_path.name}: No matching .bib file.")

    if tasks:
        await asyncio.gather(*tasks)
    else:
        print("No valid PDF-BibTeX pairs found.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python upload_documents.py <root_directory>")
        sys.exit(1)

    root_directory = sys.argv[1]
    asyncio.run(upload_all_files(root_directory))
