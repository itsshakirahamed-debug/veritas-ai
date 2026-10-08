"""
Clause & Paragraph aware Chunking Engine for Veritas Legal.
Combines small paragraphs or splits large legal clauses into chunks of 300-500 tokens (1000-2000 chars).
Preserves page numbers, paragraph IDs, and exact character start/end offsets.
"""

import uuid
from typing import List, Dict, Any
from ..schemas import Chunk, DocType


class ClauseChunker:
    def __init__(self, target_min_chars: int = 800, target_max_chars: int = 2000):
        self.target_min_chars = target_min_chars
        self.target_max_chars = target_max_chars

    def create_chunks(self, parsed_paragraphs: List[Dict[str, Any]]) -> List[Chunk]:
        """
        Takes raw parsed paragraphs and produces structured Chunk objects.
        """
        chunks: List[Chunk] = []

        if not parsed_paragraphs:
            return chunks

        current_text = ""
        current_para_ids = []
        current_page = parsed_paragraphs[0]["page"]
        doc_id = parsed_paragraphs[0]["doc_id"]
        doc_type = parsed_paragraphs[0]["doc_type"]
        start_char = parsed_paragraphs[0]["char_start"]
        end_char = parsed_paragraphs[0]["char_end"]

        for idx, para in enumerate(parsed_paragraphs):
            para_text = para["text"]
            para_len = len(para_text)

            # If a single paragraph is larger than target_max_chars, split along clause boundaries
            if para_len > self.target_max_chars:
                # Flush existing accumulated chunk first
                if current_text:
                    chunks.append(self._build_chunk(
                        doc_id=doc_id,
                        doc_type=doc_type,
                        page=current_page,
                        para_id=current_para_ids[0] if len(current_para_ids) == 1 else f"{current_para_ids[0]}-{current_para_ids[-1]}",
                        char_start=start_char,
                        char_end=end_char,
                        text=current_text.strip()
                    ))
                    current_text = ""
                    current_para_ids = []

                # Split large clause into sub-chunks
                sub_chunks = self._split_clause(para)
                chunks.extend(sub_chunks)

                # Reset accumulator for next items
                if idx + 1 < len(parsed_paragraphs):
                    next_p = parsed_paragraphs[idx + 1]
                    current_page = next_p["page"]
                    start_char = next_p["char_start"]
                    end_char = next_p["char_end"]
                continue

            # Check page change or length limit
            if (len(current_text) + para_len > self.target_max_chars) or (para["page"] != current_page and len(current_text) >= self.target_min_chars):
                if current_text:
                    chunks.append(self._build_chunk(
                        doc_id=doc_id,
                        doc_type=doc_type,
                        page=current_page,
                        para_id=current_para_ids[0] if len(current_para_ids) == 1 else f"{current_para_ids[0]}-{current_para_ids[-1]}",
                        char_start=start_char,
                        char_end=end_char,
                        text=current_text.strip()
                    ))
                current_text = para_text
                current_para_ids = [para["para_id"]]
                current_page = para["page"]
                start_char = para["char_start"]
                end_char = para["char_end"]
            else:
                if not current_text:
                    start_char = para["char_start"]
                    current_page = para["page"]
                current_text += ("\n\n" + para_text) if current_text else para_text
                current_para_ids.append(para["para_id"])
                end_char = para["char_end"]

        # Flush trailing accumulated chunk
        if current_text:
            chunks.append(self._build_chunk(
                doc_id=doc_id,
                doc_type=doc_type,
                page=current_page,
                para_id=current_para_ids[0] if len(current_para_ids) == 1 else f"{current_para_ids[0]}-{current_para_ids[-1]}",
                char_start=start_char,
                char_end=end_char,
                text=current_text.strip()
            ))

        return chunks

    def _split_clause(self, para: Dict[str, Any]) -> List[Chunk]:
        """Splits a large paragraph along sentence/clause boundaries."""
        sub_chunks = []
        text = para["text"]
        
        # Split on sentence boundaries or legal clause dividers
        sentences = [s.strip() for s in text.replace("; ", ".\n").split(". ") if s.strip()]
        
        cur_sub_text = ""
        sub_start = para["char_start"]
        
        for sent in sentences:
            if len(cur_sub_text) + len(sent) > self.target_max_chars and cur_sub_text:
                sub_end = sub_start + len(cur_sub_text)
                sub_chunks.append(self._build_chunk(
                    doc_id=para["doc_id"],
                    doc_type=para["doc_type"],
                    page=para["page"],
                    para_id=f"{para['para_id']}_sub{len(sub_chunks)+1}",
                    char_start=sub_start,
                    char_end=sub_end,
                    text=cur_sub_text.strip()
                ))
                sub_start = sub_end + 1
                cur_sub_text = sent
            else:
                cur_sub_text += (". " + sent) if cur_sub_text else sent

        if cur_sub_text:
            sub_chunks.append(self._build_chunk(
                doc_id=para["doc_id"],
                doc_type=para["doc_type"],
                page=para["page"],
                para_id=f"{para['para_id']}_sub{len(sub_chunks)+1}",
                char_start=sub_start,
                char_end=para["char_end"],
                text=cur_sub_text.strip()
            ))

        return sub_chunks

    def _build_chunk(
        self,
        doc_id: str,
        doc_type: DocType,
        page: int,
        para_id: Any,
        char_start: int,
        char_end: int,
        text: str
    ) -> Chunk:
        unique_id = f"chk_{doc_id}_{page}_{uuid.uuid4().hex[:8]}"
        return Chunk(
            chunk_id=unique_id,
            doc_id=doc_id,
            doc_type=doc_type,
            page=page,
            para_id=para_id,
            char_start=char_start,
            char_end=char_end,
            text=text
        )
