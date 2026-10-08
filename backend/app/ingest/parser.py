"""
PDF Ingestion Engine for Veritas Legal.
Extracts pages and paragraphs with page numbers and character offsets (char_start, char_end).
Supports PyMuPDF (fitz), pdfplumber, and pytesseract OCR fallback.
"""

import os
import pymupdf  # PyMuPDF (fitz import name is deprecated)
import pdfplumber
from typing import List, Dict, Any, Optional
from ..schemas import Chunk, DocType


class PDFParser:
    def __init__(self, doc_id: str, doc_type: DocType):
        self.doc_id = doc_id
        self.doc_type = doc_type

    def parse_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parses a PDF file into structured paragraph elements with character offsets.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        paragraphs = []
        para_counter = 0

        doc = pymupdf.open(file_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_index = page_num + 1
            page_text = page.get_text("text")

            # Fallback to pdfplumber / OCR if page text is empty
            if not page_text or len(page_text.strip()) < 10:
                page_text = self._extract_ocr_fallback(file_path, page_num)

            if not page_text or not page_text.strip():
                continue

            # Split into paragraphs by blank lines or double newlines
            raw_blocks = page.get_text("blocks") if hasattr(page, "get_text") else []
            
            if raw_blocks:
                char_offset = 0
                for block in raw_blocks:
                    # block format: (x0, y0, x1, y1, text, block_no, block_type)
                    if len(block) >= 5 and isinstance(block[4], str):
                        block_text = block[4].strip()
                        if not block_text:
                            continue
                        
                        para_counter += 1
                        char_start = char_offset
                        char_end = char_start + len(block_text)
                        char_offset = char_end + 1  # count spacing

                        paragraphs.append({
                            "doc_id": self.doc_id,
                            "doc_type": self.doc_type,
                            "page": page_index,
                            "para_id": para_counter,
                            "char_start": char_start,
                            "char_end": char_end,
                            "text": block_text
                        })
            else:
                # Fallback paragraph splitting by double newlines
                blocks = [b.strip() for b in page_text.split("\n\n") if b.strip()]
                char_offset = 0
                for block_text in blocks:
                    para_counter += 1
                    char_start = char_offset
                    char_end = char_start + len(block_text)
                    char_offset = char_end + 2

                    paragraphs.append({
                        "doc_id": self.doc_id,
                        "doc_type": self.doc_type,
                        "page": page_index,
                        "para_id": para_counter,
                        "char_start": char_start,
                        "char_end": char_end,
                        "text": block_text
                    })

        doc.close()
        return paragraphs

    def _extract_ocr_fallback(self, file_path: str, page_num: int) -> str:
        """Fallback text extraction using pdfplumber and pytesseract OCR if available."""
        try:
            with pdfplumber.open(file_path) as pdf:
                if page_num < len(pdf.pages):
                    text = pdf.pages[page_num].extract_text()
                    if text and len(text.strip()) >= 10:
                        return text
        except Exception:
            pass

        # Pytesseract OCR fallback for scanned images
        try:
            import pytesseract
            from PIL import Image

            doc = pymupdf.open(file_path)
            page = doc[page_num]
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            
            import io
            image = Image.open(io.BytesIO(img_bytes))
            ocr_text = pytesseract.image_to_string(image)
            doc.close()
            return ocr_text.strip()
        except Exception:
            return ""
