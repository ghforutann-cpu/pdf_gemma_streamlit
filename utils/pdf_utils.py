from pypdf import PdfReader
from typing import List

def extract_pages_from_pdf(uploaded_pdf) -> List[dict]:
    """Return list of dicts: { 'text': ..., 'page_number': i+1 }"""
    reader = PdfReader(uploaded_pdf)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"text": text.strip(), "page_number": i+1})
    return pages
