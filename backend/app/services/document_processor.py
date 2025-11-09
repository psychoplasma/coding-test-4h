"""
Document processing service using Docling

Implements PDF document processing with:
1. Parse PDF documents using Docling
2. Extract text, images, and tables
3. Store extracted content in database
4. Generate embeddings for text chunks
"""
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import time
from typing import Dict, Any, List, Literal, Tuple

from docling_core.types.doc import PictureItem, TableItem
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument
from docling.datamodel.pipeline_options import TableFormerMode, PdfPipelineOptions
from PIL import Image
from sqlalchemy.orm import Session

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
    
    async def process_document(self, session: Session, file_path: str, document_id: int) -> None:
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
        images_count = 0
        tables_count = 0

        # Get document record
        document = self._get_document_by_id(session, document_id)

        # Update status to processing
        document.processing_status = "processing"
        session.flush()

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

        # Chunk document and save embeddings to Vector Store
        num_chunks = await self._chunk_document(
            session,
            document_id,
            docling_result.document,
            image_ids_by_page,
            table_ids_by_page,
        )

        processing_time = time.time() - start_time

        document.total_pages = len(docling_result.pages)
        document.text_chunks_count = num_chunks
        document.images_count = images_count
        document.tables_count = tables_count
        # Update status to completed
        document.processing_status = "completed"
        document.processing_time = processing_time
        session.flush()
        
        logger.info(
            f"Document {document_id} processed successfully. "
            f"Chunks: {num_chunks}, Images: {images_count}, Tables: {tables_count}, "
            f"Time: {processing_time:.2f}s"
        )
    
    def _parse_pdf_file(self, file_path: str) -> ConversionResult:
        """
        Parse PDF using Docling and return the document object.
        
        Args:
            file_path: Path to the PDF file
        """
        logger.info(f"Parsing PDF: {file_path}")

        pipeline_options = PdfPipelineOptions(do_table_structure=True)
        pipeline_options.table_structure_options.do_cell_matching = False
        pipeline_options.do_formula_enrichment = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            },
        )
        docling_result = converter.convert(file_path)

        logger.info(f"PDF parsed successfully. Pages: {len(docling_result.pages)}")

        return docling_result

    async def _extract_and_save_image_metadata(
        self,
        session: Session,
        document_id: int,
        docling_result: ConversionResult,
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
        for element, _ in docling_result.document.iterate_items():
            if isinstance(element, PictureItem):
                images_count += 1
                image = element.get_image(docling_result.document)
                caption = element.caption_text(docling_result.document)
                page_num = element.prov[0].page_no  # Use the first page this image appears on

                img_record = await self._save_image(
                    session=session,
                    image=image,
                    document_id=document_id,
                    page_number=page_num,
                    metadata={
                        "caption": caption,
                        "source": "docling_extraction" #TODO: use file source
                    }
                )
                if image_ids_by_page.get(page_num) is None:
                    image_ids_by_page[page_num] = []
                image_ids_by_page[page_num].append(img_record.id)
                logger.info(f"Saved image on page {page_num}: {img_record.id}")

        logger.info(f"Extracted {images_count} images from document {document_id}")
        return images_count, image_ids_by_page

    async def _save_image(
        self,
        session: Session,
        image: Image.Image, 
        document_id: int, 
        page_number: int,
        metadata: Dict[str, Any]
    ) -> DocumentImage:
        """
        Save an extracted image to disk and database.

        - Save image file to disk in UPLOAD_DIR/images/
        - Create DocumentImage record
        - Extract caption if available

        Args:
            session: Scoped database session
            image: PIL Image object
            document_id: Document ID
            page_number: Page number
            metadata: Image metadata (caption, source, etc.)

        Returns:
            DocumentImage database record
        """
        image_path = self._save_to_disk("image", document_id, page_number, image)

        # Create database record
        doc_image = DocumentImage(
            document_id=document_id,
            file_path=image_path,
            page_number=page_number,
            caption=metadata.get("caption"),
            width=image.width,
            height=image.height,
            metadata=metadata,
        )

        session.add(doc_image)
        session.flush()

        logger.info(f"Created DocumentImage record {doc_image.id}")
        return doc_image

    async def _extract_and_save_table_metadata(
        self,
        session: Session,
        document_id: int,
        docling_result: ConversionResult,
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
        for element, _ in docling_result.document.iterate_items():
            if isinstance(element, TableItem):
                tables_count += 1
                page_num = element.prov[0].page_no  # Use the first page this table appears on
                table_image = element.get_image(docling_result.document)
                table_caption = element.caption_text(docling_result.document)

                tbl_record = self._save_table(
                    session=session,
                    table=element,
                    table_image=table_image,
                    document_id=document_id,
                    page_number=page_num,
                    metadata={
                        "caption": table_caption,
                        "source": "docling_extraction" #TODO: use file source
                    }
                )
                if table_ids_by_page.get(page_num) is None:
                    table_ids_by_page[page_num] = []
                table_ids_by_page[page_num].append(tbl_record.id)
                logger.info(f"Saved table on page {page_num}: {tbl_record.id}")
        
        logger.info(f"Extracted {tables_count} tables from document {document_id}")
        return tables_count, table_ids_by_page

    def _save_table(
        self,
        session: Session,
        table: TableItem,
        table_image: Image.Image,
        document_id: int,
        page_number: int,
        metadata: Dict[str, Any]
    ) -> DocumentTable:
        """
        Save a table as image and structured data.

        - Save table image to disk
        - Store structured data as JSON
        - Create DocumentTable record

        Args:
            session: Scoped database session
            table: Table object
            table_image: Table image (PIL Image)
            document_id: Document ID
            page_number: Page number
            metadata: Table metadata (caption, source, etc.)

        Returns:
            DocumentTable database record
        """
        # Save table image
        image_path = self._save_to_disk("table", document_id, page_number, table_image)
        # Create database record
        doc_table = DocumentTable(
            document_id=document_id,
            image_path=image_path,
            page_number=page_number,
            caption=metadata.get("caption"),
            rows=table.data.num_rows,
            columns=table.data.num_cols,
            data=table.data.model_dump(), # FIXME: Does this serialize table cells correctly?
            metadata=metadata
        )

        session.add(doc_table)
        session.flush()

        logger.info(f"Created DocumentTable record {doc_table.id}")
        return doc_table

    def _save_to_disk(
        self,
        media_type: Literal["table", "image"],
        document_id: int,
        page_number: int,
        image: Image.Image,
    ) -> str:
        """
        Save image to disk, or create a placeholder if saving fails.

        Args:
            media_type: Type of media ("table" or "image")
            document_id: Document ID of the image
            page_number: Page number of the image
            image: PIL Image object

        Returns:
            Path to the saved image file
        """
        # Generate unique image filename
        image_path = self._generate_unique_filepath(media_type, document_id, page_number)

        # Save image
        try:
            image.save(image_path, format="PNG")
            logger.debug(f"Saved image to {image_path}")
        except Exception as e:
            logger.warning(f"Failed to save image to '{image_path}': {e}. Creating a placeholder instead.")
            # Create a minimal placeholder image
            placeholder = Image.new('RGB', (800, 600), color='white')
            placeholder.save(image_path, format="PNG")

        return image_path

    async def _chunk_document(
        self,
        session: Session,
        document_id: int,
        document: DoclingDocument,
        image_ids_by_page: Dict[int, List[int]],
        table_ids_by_page: Dict[int, List[int]],
    ) -> int:
        """
        Chunk document text and save them in Vector Store with embeddings

        Args:
            session: Scoped database session
            document_id: Corresponding document ID
            document: Parsed Docling document object
            image_ids_by_page: Related images by page
            table_ids_by_page: Related tables by page
        Returns:
            List of text chunk dictionaries
        """
        from docling.chunking import HybridChunker
        from docling_core.transforms.chunker.hierarchical_chunker import DocChunk

        chunker = HybridChunker(tokenizer=self.vector_store.tokenizer)
        chunk_iter = list(chunker.chunk(document))

        logger.info(f"Generated {len(chunk_iter)} chunks.")

        num_chunks = 0
        for i, chunk in enumerate(chunk_iter):
            doc_chunk = DocChunk.model_validate(chunk)
            page_num = doc_chunk.meta.doc_items[0].prov[0].page_no # Get the number the page where this chunk first appears in
            text = chunker.contextualize(chunk)
            try:
                await self.vector_store.store_chunk(
                    session=session,
                    content=text,
                    document_id=document_id,
                    page_number=page_num,
                    chunk_index=i,
                    metadata={
                        "related_images": image_ids_by_page.get(page_num, []),
                        "related_tables": table_ids_by_page.get(page_num, []),
                        "start_pos": 0, # TODO: Is this applicable for HybridChunker?  
                    },
                )
                num_chunks += 1
            except Exception as e:
                logger.error(f"Failed to save chunk with index {i}, page {page_num}: {e}")
                # Continue processing chunks

        logger.info(f"Saved {num_chunks} chunks to Vector Store")

        return num_chunks

    def _generate_unique_filepath(
        self,
        media_type: Literal["table", "image"],
        document_id: int,
        page_number: int,
    ) -> str:
        """Generate a unique filepath from document ID, media type, and page number."""
        timestamp = datetime.now(timezone.utc).timestamp()
        filename = f"doc_{document_id}_{media_type}_{page_number}_{int(timestamp)}.png"
        return os.path.join(settings.UPLOAD_DIR, f"{media_type}s", filename)

    def _get_document_by_id(self, session: Session, document_id: int) -> Document:
        """Fetch document record from the database."""
        document = session.get(Document, document_id)
        if not document:
            raise ValueError(f"Document with ID {document_id} not found")
        return document
