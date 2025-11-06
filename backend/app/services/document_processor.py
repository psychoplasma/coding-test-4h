"""
Document processing service using Docling

Implements PDF document processing with:
1. Parse PDF documents using Docling
2. Extract text, images, and tables
3. Store extracted content in database
4. Generate embeddings for text chunks
"""
from datetime import datetime, timezone
from io import BytesIO
import logging
import os
from pathlib import Path
import re
import time
from typing import Dict, Any, List, Literal, Tuple

from docling.document_converter import DocumentConverter
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy.orm import Session

from app.db.session import scoped_session
from app.core.config import settings
from app.models.document import Document, DocumentImage, DocumentTable
from app.services.vector_store import VectorStore


logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Process PDF documents and extract multimodal content.
    
    Uses Docling for PDF parsing and supports text, image, and table extraction.
    Integrates with VectorStore for embedding generation.
    """
    
    def __init__(self):
        self.vector_store = VectorStore()
        self._ensure_upload_dir()
    
    def _ensure_upload_dir(self):
        """Ensure upload directory exists."""
        upload_dir = settings.UPLOAD_DIR
        Path(upload_dir).mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        for subdir in ["images", "tables"]:
            Path(os.path.join(upload_dir, subdir)).mkdir(parents=True, exist_ok=True)
    
    async def process_document(self, file_path: str, document_id: int) -> Dict[str, Any]:
        """
        Process a PDF document using Docling.

        Steps:
        1. Update document status to 'processing'
        2. Use Docling to parse the PDF
        3. Extract and save text chunks
        4. Extract and save images
        5. Extract and save tables
        6. Generate embeddings for text chunks
        7. Update document status to 'completed'
        8. Handle errors appropriately
        
        Args:
            file_path: Path to the uploaded PDF file
            document_id: Database ID of the document
            
        Returns:
            {
                "status": "success" or "error",
                "text_chunks": <count>,
                "images": <count>,
                "tables": <count>,
                "processing_time": <seconds>,
                "error": <error_message if failed>
            }
        """
        logger.info(f"Starting to process document {document_id}")

        start_time = time.time()
        all_chunks = []
        text_chunks_count = 0
        images_count = 0
        tables_count = 0

        with scoped_session() as session:
            # Get document record
            document = self._get_document_by_id(session, document_id)

            # Update status to processing
            document.processing_status = "processing"
            session.commit()

            # Parse PDF with Docling
            docling_result = self._parse_pdf_file(file_path)

            # Extract and save images first
            images_count, image_ids_by_page = await self._extract_and_save_image_metadata(
                session,
                document_id,
                docling_result,
            )

            # Extract and save tables
            tables_count, table_ids_by_page = await self._extract_and_save_table_metadata(
                session,
                document_id,
                docling_result,
            )

            # Extract and chunk texts
            all_chunks = self._extract_text_and_create_chunks(
                docling_result,
                image_ids_by_page,
                table_ids_by_page,
            )
            text_chunks_count = len(all_chunks)

            document.total_pages = len(docling_result.pages)
            document.text_chunks_count = text_chunks_count
            document.images_count = images_count
            document.tables_count = tables_count

            # Save chunks with embeddings in VectorStore
            await self._save_text_chunks(session,all_chunks, document_id)

            # Get document record
            document = self._get_document_by_id(session, document_id)
            # Update status to completed
            document.processing_status = "completed"

            processing_time = time.time() - start_time
            logger.info(
                f"Document {document_id} processed successfully. "
                f"Chunks: {text_chunks_count}, Images: {images_count}, Tables: {tables_count}, "
                f"Time: {processing_time:.2f}s"
            )

            return {
                "status": "success",
                "text_chunks": text_chunks_count,
                "images": images_count,
                "tables": tables_count,
                "processing_time": processing_time
            }
    
    def _parse_pdf_file(self, file_path: str) -> Any:
        """
        Parse PDF using Docling and return the document object.
        
        Args:
            file_path: Path to the PDF file
        """
        logger.info(f"Parsing PDF: {file_path}")

        converter = DocumentConverter()
        docling_result = converter.convert(file_path)

        logger.info(f"PDF parsed successfully. Pages: {len(docling_result.pages)}")

        return docling_result

    async def _extract_and_save_image_metadata(
        self,
        session: Session,
        document_id: int,
        docling_result: Any,
    ) -> Tuple[int, Dict[int, List[int]]]:
        """
        Extract metadata from an image object.
        
        Args:
            document_id: Document ID
            docling_result: Parsed Docling document object

        Returns:
            tuple(int, dict): (images_count, image_ids_by_page)
        """
        images_count = 0
        # Track images per page
        image_ids_by_page = {}

        # Extract images
        logger.info("Extracting images...")
        for page_num, page_view in enumerate(docling_result.pages, 1):
            image_ids_by_page[page_num] = []

            # Extract images from page
            if hasattr(page_view, 'images'):
                for image in page_view.images:
                    img_record = await self._save_image(
                        session=session,
                        image_data=image.to_bytes() if hasattr(image, 'to_bytes') else image.image,
                        document_id=document_id,
                        page_number=page_num,
                        metadata={
                            "caption": image.caption if hasattr(image, 'caption') else None,
                            "source": "docling_extraction"
                        }
                    )
                    image_ids_by_page[page_num].append(img_record.id)
                    images_count += 1
                    logger.debug(f"Saved image on page {page_num}: {img_record.id}")

        logger.info(f"Extracted {images_count} images from document {document_id}")
        return images_count, image_ids_by_page

    async def _save_image(
        self,
        session: Session,
        image_data: bytes, 
        document_id: int, 
        page_number: int,
        metadata: Dict[str, Any]
    ) -> DocumentImage:
        """
        Save an extracted image to disk and database.
        
        Implementation:
        - Save image file to disk in UPLOAD_DIR/images/
        - Create DocumentImage record
        - Extract caption if available
        
        Args:
            image_data: Image bytes
            document_id: Document ID
            page_number: Page number
            metadata: Image metadata (caption, source, etc.)
            
        Returns:
            DocumentImage database record
            
        Raises:
            RuntimeError: If image saving fails
        """
        image_path = self._generate_unique_filepath("table", document_id, page_number)
        
        # Convert image_data to PIL Image
        width, height = self._convert_image_to_png_and_save(image_path, image_data)

        # Create database record
        doc_image = DocumentImage(
            document_id=document_id,
            file_path=image_path,
            page_number=page_number,
            caption=metadata.get("caption"),
            width=width,
            height=height,
            metadata=metadata,
        )

        session.add(doc_image)
        session.flush()

        logger.info(f"Created DocumentImage record {doc_image.id}")
        return doc_image
    
    def _convert_image_to_png_and_save(self, image_path: str, image_data: bytes):
        width, height = None, None
        try:
            if isinstance(image_data, bytes):
                pil_image = Image.open(BytesIO(image_data))
            else:
                pil_image = image_data
            
            # Ensure RGB mode for PNG
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            
            # Save image
            pil_image.save(image_path, format="PNG")

            width, height = pil_image.width, pil_image.height

            logger.debug(f"Saved image to {image_path}")
        except Exception as e:
            logger.warning(f"Failed to process image as Pillow image, saving raw bytes instead: {e}")
            # Save raw bytes
            with open(image_path, 'wb') as f:
                f.write(image_data)

        return width, height

    async def _extract_and_save_table_metadata(
        self,
        session: Session,
        document_id: int,
        docling_result: Any,
    ) -> Tuple[int, Dict[int, List[int]]]:
        """
        Extract table metadata from the Docling result.
        
        Args:
            document_id: Document ID
            docling_result: Parsed Docling document object

        Returns:
            tuple(int, dict): (tables_count, table_ids_by_page)
        """
        tables_count = 0
        # Track tables per page
        table_ids_by_page = {}

        # Extract tables
        logger.info("Extracting tables...")
        for page_num, page_view in enumerate(docling_result.pages, 1):
            table_ids_by_page[page_num] = []
            
            # Extract tables from page
            if hasattr(page_view, 'tables'):
                for table in page_view.tables:
                    tbl_record = await self._save_table(
                        session=session,
                        table_data=table,
                        document_id=document_id,
                        page_number=page_num,
                        metadata={
                            "caption": table.caption if hasattr(table, 'caption') else None,
                            "source": "docling_extraction"
                        }
                    )
                    table_ids_by_page[page_num].append(tbl_record.id)
                    tables_count += 1
                    logger.debug(f"Saved table on page {page_num}: {tbl_record.id}")

    async def _save_table(
        self,
        session: Session,
        table_data: Any,
        document_id: int,
        page_number: int,
        metadata: Dict[str, Any]
    ) -> DocumentTable:
        """
        Save an extracted table as image and structured data.
        
        Implementation:
        - Render table as image
        - Store structured data as JSON
        - Create DocumentTable record
        
        Args:
            table_data: Table object from Docling
            document_id: Document ID
            page_number: Page number
            metadata: Table metadata (caption, source, etc.)
            
        Returns:
            DocumentTable database record
            
        Raises:
            RuntimeError: If table saving fails
        """
        # Generate unique table filename
        table_image_path = self._generate_unique_filepath("table", document_id, page_number)

        # Extract table data
        table_dict, rows, columns = self._extract_table_data_as_dict(table_data)

        # Render table as image
        self._render_table_as_image_and_save(table_dict, table_data, table_image_path)

        # Create database record
        doc_table = DocumentTable(
            document_id=document_id,
            image_path=table_image_path,
            page_number=page_number,
            caption=metadata.get("caption"),
            rows=rows,
            columns=columns,
            data=table_dict,
            metadata=metadata
        )

        session.add(doc_table)
        session.commit()
        session.refresh(doc_table)

        logger.info(f"Created DocumentTable record {doc_table.id}")
        return doc_table
    
    def _extract_table_data_as_dict(table_data: Any) -> Tuple[Dict[str, Any], int, int]:
        """
        Extract table data as dictionary along with dimensions.
        """
        # Extract table data
        table_dict = {}
        rows = 0
        columns = 0

        # Try to extract structured data from table
        if hasattr(table_data, 'to_dict'):
            table_dict = table_data.to_dict()
        elif hasattr(table_data, 'data'):
            table_dict = table_data.data
        else:
            # Fallback: convert to string representation
            table_dict = {"content": str(table_data)}

        # Try to get dimensions
        if isinstance(table_dict, dict):
            if "rows" in table_dict:
                rows = len(table_dict.get("rows", []))
            elif "data" in table_dict:
                rows = len(table_dict["data"])
            
            if "columns" in table_dict:
                columns = len(table_dict["columns"])
            elif rows > 0 and isinstance(table_dict.get("data", [None])[0], list):
                columns = len(table_dict["data"][0])

        return table_dict, rows, columns

    def _render_table_as_image_and_save(
        self,
        table_dict: Dict[str, Any],
        table_data: Any,
        table_image_path: str
    ) -> None:
        """
        Render table as an image for display.
        
        Args:
            table_dict: Table data as dictionary
            table_data: Original table object
            
        Returns:
            PIL Image with rendered table
        """
        try:
            # Try to use table object's rendering if available
            if hasattr(table_data, 'to_image'):
                return table_data.to_image()
        except:
            pass
        
        # Fallback: create simple table visualization
        cell_width = 150
        cell_height = 40
        margin = 10
        
        # Extract rows and columns
        rows = []
        if isinstance(table_dict, dict):
            if "rows" in table_dict:
                rows = table_dict["rows"]
            elif "data" in table_dict:
                rows = table_dict["data"]
            else:
                # Create a simple text representation
                rows = [[str(table_dict)]]
        
        if not rows:
            rows = [[str(table_dict)]]
        
        # Limit table size for rendering
        rows = rows[:20]  # Max 20 rows
        cols = max([len(row) if isinstance(row, list) else 1 for row in rows], 1)
        cols = min(cols, 6)  # Max 6 columns
        
        # Calculate image size
        width = cols * cell_width + margin * 2
        height = len(rows) * cell_height + margin * 2
        
        # Create image
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        try:
            # Try to use a better font
            font = ImageFont.truetype("/System/Library/Fonts/Monaco.dfont", 10)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Draw cells
        for row_idx, row in enumerate(rows):
            for col_idx in range(min(cols, len(row) if isinstance(row, list) else 1)):
                x = margin + col_idx * cell_width
                y = margin + row_idx * cell_height
                
                # Draw cell border
                draw.rectangle(
                    [x, y, x + cell_width, y + cell_height],
                    outline='black'
                )
                
                # Draw cell text
                try:
                    if isinstance(row, list) and col_idx < len(row):
                        cell_text = str(row[col_idx])[:20]
                    else:
                        cell_text = str(row)[:20]
                    
                    text_x = x + 5
                    text_y = y + 5
                    draw.text((text_x, text_y), cell_text, fill='black', font=font)
                except Exception as e:
                    logger.debug(f"Failed to draw cell text: {e}")

        # Save table image
        try:
            image.save(table_image_path, format="PNG")
            logger.debug(f"Rendered table image to {table_image_path}")
        except Exception as e:
            logger.warning(f"Failed to render table image: {e}. Creating a placeholder instead.")
            # Create a minimal placeholder image
            placeholder = Image.new('RGB', (800, 600), color='white')
            placeholder.save(table_image_path, format="PNG")

    def _extract_text_and_create_chunks(
        self,
        docling_result: Any,
        image_ids_by_page: Dict[int, List[int]],
        table_ids_by_page: Dict[int, List[int]],
    ) -> List[Dict[str, Any]]:
        """
        Extract and chunk text from the Docling result.

        Args:
            document_id: Document ID
            docling_result: Parsed Docling document object
            image_ids_by_page: Mapping of page numbers to image IDs
            table_ids_by_page: Mapping of page numbers to table IDs

        Returns:
            List of text chunk dictionaries
        """
        logger.info("Extracting and chunking text...")
        all_chunks = []
        chunk_index = 0

        for page_num, page_view in enumerate(docling_result.pages, 1):
            # Extract text from page
            text_content = page_view.text if hasattr(page_view, 'text') else ""
            
            if not text_content.strip():
                logger.debug(f"No text content on page {page_num}")
                continue

            # Chunk text
            chunks = self._chunk_text(text_content, page_num)

            # Add related media IDs to chunks
            for chunk in chunks:
                chunk["related_images"] = image_ids_by_page.get(page_num, [])
                chunk["related_tables"] = table_ids_by_page.get(page_num, [])
                chunk["chunk_index"] = chunk_index
                chunk_index += 1

            all_chunks.extend(chunks)
            logger.debug(f"Chunked page {page_num} into {len(chunks)} chunks")
        
        return all_chunks

    def _chunk_text(self, text: str, page_number: int) -> List[Dict[str, Any]]:
        """
        Split text into chunks for vector storage.
        
        Implementation:
        - Split by sentences, then by size with overlap
        - Maintain context with overlap
        - Keep metadata (page number, position, etc.)
        
        Args:
            text: Text to chunk
            document_id: Document ID (for reference)
            page_number: Page number where text appears
            
        Returns:
            List of chunk dictionaries with:
            - content: chunk text
            - page_number: page where chunk appears
            - start_pos: start position in page text
            - related_images: image IDs (added by caller)
            - related_tables: table IDs (added by caller)
        """
        chunks = []
        
        if not text or not text.strip():
            return chunks
        
        chunk_size = settings.CHUNK_SIZE
        overlap = settings.CHUNK_OVERLAP
        
        # Simple chunking strategy: split by sentences, then by size with overlap
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_chunk_start = 0
        
        for i, sentence in enumerate(sentences):
            # Add sentence to current chunk
            if current_chunk:
                test_chunk = current_chunk + " " + sentence
            else:
                test_chunk = sentence
            
            # Check if adding this sentence would exceed chunk size
            if len(test_chunk) > chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    "content": current_chunk.strip(),
                    "page_number": page_number,
                    "start_pos": current_chunk_start,
                    "related_images": [],
                    "related_tables": []
                })
                
                # Start new chunk with overlap
                # Include last few sentences for context
                overlap_text = " ".join(sentences[max(0, i-2):i])
                current_chunk = overlap_text + " " + sentence
                current_chunk_start = len(text) - len(test_chunk)
            else:
                current_chunk = test_chunk
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "page_number": page_number,
                "start_pos": current_chunk_start,
                "related_images": [],
                "related_tables": []
            })
        
        logger.debug(f"Created {len(chunks)} chunks from page {page_number}")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for better chunking.
        
        Simple heuristic-based splitting:
        - Split on '. ', '! ', '? '
        - Preserve sentence boundaries for context
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Split on sentence boundaries but keep delimiters
        sentences = re.split(r'([.!?])\s+', text)
        
        # Reconstruct sentences with delimiters
        result = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and sentences[i + 1] in '.!?':
                # Combine text with its delimiter
                result.append(sentences[i] + sentences[i + 1])
                i += 2
            elif sentences[i].strip():
                result.append(sentences[i])
                i += 1
            else:
                i += 1
        
        return [s.strip() for s in result if s.strip()]

    async def _save_text_chunks(
        self,
        session: Session,
        chunks: List[Dict[str, Any]],
        document_id: int,
    ) -> None:
        """
        Save text chunks to database with embeddings.
        
        Implementation:
        - Generate embeddings using VectorStore
        - Store in database
        - Link related images/tables in metadata
        
        Args:
            chunks: List of chunk dictionaries
            document_id: Document ID
        """
        logger.info(f"Saving {len(chunks)} text chunks with embeddings")
        
        saved_count = 0
        for i, chunk in enumerate(chunks):
            try:
                # Prepare metadata with related content
                metadata = {
                    "related_images": chunk.get("related_images", []),
                    "related_tables": chunk.get("related_tables", []),
                    "start_pos": chunk.get("start_pos", 0)
                }

                # Store chunk with embeddings
                await self.vector_store.store_chunk(
                    session=session,
                    content=chunk["content"],
                    document_id=document_id,
                    page_number=chunk["page_number"],
                    chunk_index=chunk.get("chunk_index", i),
                    metadata=metadata
                )

                saved_count += 1

                if (i + 1) % 10 == 0:
                    logger.debug(f"Saved {i + 1}/{len(chunks)} chunks")

            except Exception as e:
                logger.error(f"Failed to save chunk {i}: {e}")
                # Continue processing other chunks
                continue

        logger.info(f"Successfully saved {saved_count}/{len(chunks)} chunks")

    def _generate_unique_filepath(
        self,
        document_type: Literal["table", "image"],
        document_id: int,
        page_number: int,
    ) -> str:
        """Generate a unique filepath from document ID, type, and page number."""
        timestamp = datetime.now(timezone.utc).timestamp()
        filename = f"doc_{document_id}_{document_type}_{page_number}_{int(timestamp)}.png"
        return os.path.join(settings.UPLOAD_DIR, f"{document_type}s", filename)

    def _get_document_by_id(self, session: Session, document_id: int) -> Document:
        """Fetch document record from the database."""
        document = session.get(Document, document_id)
        if not document:
            raise ValueError(f"Document with ID {document_id} not found")
        return document
