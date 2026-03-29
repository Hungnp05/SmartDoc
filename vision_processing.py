"""
Layer A: Vision Processing Layer
─────────────────────────────────
Handles PDF/image ingestion with multi-modal understanding:
  1. Layout Analysis  → detect text / title / table / figure regions
  2. Text Extraction  → OCR on text regions
  3. Table Extraction → structure-preserving Markdown conversion
  4. Chart Description → LLaVA-7B generates textual descriptions of figures
"""

import io
import base64
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    x1: int; y1: int; x2: int; y2: int

    def to_tuple(self):
        return (self.x1, self.y1, self.x2, self.y2)

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y1:self.y2, self.x1:self.x2]


@dataclass
class DocumentRegion:
    region_type: str          # text | title | table | figure | reference
    bbox: BoundingBox
    confidence: float
    page_num: int
    content: str = ""         # Extracted content (text, markdown, description)
    raw_image: Optional[np.ndarray] = None   # For figures


@dataclass
class ProcessedPage:
    page_num: int
    page_summary: str = ""
    regions: list[DocumentRegion] = field(default_factory=list)
    full_text: str = ""


@dataclass
class ProcessedDocument:
    source_file: str
    total_pages: int
    pages: list[ProcessedPage] = field(default_factory=list)

    def all_regions(self) -> list[DocumentRegion]:
        return [r for page in self.pages for r in page.regions]


# ── Vision Processing Layer ───────────────────────────────────────────────────

class VisionProcessingLayer:
    """
    Layer A: Orchestrates all vision-based document understanding.
    """

    def __init__(self, config, ollama_client):
        self.cfg = config.vision
        self.ollama = ollama_client
        self._ocr = None      # Lazy-load PaddleOCR
        self._layout = None   # Lazy-load layout model

    # ── Public API ────────────────────────────────────────────────────────────

    def process_document(self, file_path: str | Path, progress_callback=None) -> ProcessedDocument:
        """
        Main entry point. Processes a PDF or image file into structured regions.
        """
        file_path = Path(file_path)
        logger.info(f"Processing document: {file_path.name}")

        if file_path.suffix.lower() == ".pdf":
            pages_images = self._pdf_to_images(file_path)
        else:
            pages_images = [(0, self._load_image(file_path))]

        doc = ProcessedDocument(
            source_file=file_path.name,
            total_pages=len(pages_images)
        )

        for idx, (page_num, page_img) in enumerate(pages_images):
            if progress_callback:
                progress_callback(idx, len(pages_images), f"Analyzing page {page_num + 1}...")

            processed_page = self._process_page(page_img, page_num)
            doc.pages.append(processed_page)

        logger.info(f"Document processed: {len(doc.pages)} pages, "
                    f"{len(doc.all_regions())} regions detected")
        return doc

    # ── PDF Handling ──────────────────────────────────────────────────────────

    def _pdf_to_images(self, pdf_path: Path) -> list[tuple[int, np.ndarray]]:
        """Convert each PDF page to a high-resolution numpy image."""
        from src.config import config
        dpi = config.vision.pdf_dpi
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        result = []
        with fitz.open(str(pdf_path)) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8)
                img = img_array.reshape(pix.height, pix.width, 3)
                result.append((page_num, img))

        return result

    def _load_image(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        return np.array(img)

    # ── Page Processing ───────────────────────────────────────────────────────

    def _process_page(self, page_img: np.ndarray, page_num: int) -> ProcessedPage:
        """
        Full pipeline for one page:
        1. Layout detection → regions
        2. Per-region content extraction
        3. Generate page summary
        """
        processed = ProcessedPage(page_num=page_num)

        # Step 1: Detect layout regions
        raw_regions = self._detect_layout(page_img, page_num)

        # Step 2: Extract content per region type
        for region in raw_regions:
            self._extract_region_content(region, page_img)
            if region.content:  # Only keep regions with content
                processed.regions.append(region)

        # Step 3: Build page summary for metadata enrichment
        processed.full_text = self._build_page_text(processed.regions)
        processed.page_summary = self._generate_page_summary(processed.regions)

        return processed

    # ── Layout Detection ──────────────────────────────────────────────────────

    def _detect_layout(self, page_img: np.ndarray, page_num: int) -> list[DocumentRegion]:
        """
        Use PaddleOCR Layout to classify regions.
        Falls back to full-page text extraction if layout model unavailable.
        """
        try:
            return self._paddle_layout_detect(page_img, page_num)
        except Exception as e:
            logger.warning(f"Layout detection failed ({e}), using fallback OCR")
            return self._fallback_full_page_ocr(page_img, page_num)

    def _paddle_layout_detect(self, page_img: np.ndarray, page_num: int) -> list[DocumentRegion]:
        """PaddleOCR-based layout analysis."""
        from paddleocr import PPStructure

        if self._layout is None:
            self._layout = PPStructure(
                show_log=False,
                enable_mkldnn=True,
                lang=self.cfg.ocr_lang,
            )

        result = self._layout(page_img)
        regions = []

        for item in result:
            region_type = item.get("type", "text").lower()
            bbox_raw = item.get("bbox", [0, 0, 100, 100])
            bbox = BoundingBox(
                x1=int(bbox_raw[0]), y1=int(bbox_raw[1]),
                x2=int(bbox_raw[2]), y2=int(bbox_raw[3])
            )

            # Map PaddleOCR types to our schema
            type_map = {
                "text": "text", "title": "title",
                "table": "table", "figure": "figure",
                "figure_caption": "text", "reference": "reference",
                "equation": "text",
            }
            mapped_type = type_map.get(region_type, "text")

            region = DocumentRegion(
                region_type=mapped_type,
                bbox=bbox,
                confidence=item.get("score", 1.0),
                page_num=page_num,
                raw_image=bbox.crop(page_img),
            )
            regions.append(region)

        return sorted(regions, key=lambda r: (r.bbox.y1, r.bbox.x1))

    def _fallback_full_page_ocr(self, page_img: np.ndarray, page_num: int) -> list[DocumentRegion]:
        """Simple fallback: treat entire page as one text region."""
        text = self._run_ocr(page_img)
        h, w = page_img.shape[:2]
        return [DocumentRegion(
            region_type="text",
            bbox=BoundingBox(0, 0, w, h),
            confidence=1.0,
            page_num=page_num,
            content=text,
            raw_image=page_img,
        )]

    # ── Content Extraction per Region ─────────────────────────────────────────

    def _extract_region_content(self, region: DocumentRegion, page_img: np.ndarray):
        """Dispatch to the right extractor based on region type."""
        crop = region.bbox.crop(page_img)

        if region.region_type in ("text", "title", "reference"):
            region.content = self._run_ocr(crop)

        elif region.region_type == "table":
            region.content = self._extract_table_as_markdown(crop, region)

        elif region.region_type == "figure":
            region.content = self._describe_figure(crop, region)

    # ── OCR ───────────────────────────────────────────────────────────────────

    def _run_ocr(self, image: np.ndarray) -> str:
        """Run PaddleOCR on an image region and return plain text."""
        try:
            from paddleocr import PaddleOCR
            if self._ocr is None:
                self._ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.cfg.ocr_lang,
                    use_gpu=self.cfg.ocr_use_gpu,
                    show_log=False,
                )

            result = self._ocr.ocr(image, cls=True)
            if not result or not result[0]:
                return ""

            lines = []
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    lines.append(text)

            return "\n".join(lines)
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""

    # ── Table Extraction ──────────────────────────────────────────────────────

    def _extract_table_as_markdown(self, table_img: np.ndarray, region: DocumentRegion) -> str:
        """
        Convert a table image to Markdown format.
        Strategy:
          1. Try PPStructure table recognition (PaddleOCR)
          2. Fallback: grid-line detection + OCR per cell
          3. Final fallback: raw OCR
        """
        try:
            return self._paddle_table_to_markdown(table_img)
        except Exception as e:
            logger.warning(f"Paddle table extraction failed: {e}, trying grid method")

        try:
            return self._grid_table_to_markdown(table_img)
        except Exception as e:
            logger.warning(f"Grid table extraction failed: {e}, falling back to OCR")

        return f"[TABLE - OCR Fallback]\n{self._run_ocr(table_img)}"

    def _paddle_table_to_markdown(self, table_img: np.ndarray) -> str:
        """Use PPStructure's table recognition to get HTML → convert to Markdown."""
        from paddleocr import PPStructure
        from src.utils.table_extractor import html_table_to_markdown

        engine = PPStructure(table=True, ocr=True, show_log=False, lang=self.cfg.ocr_lang)
        result = engine(table_img)

        for item in result:
            if item.get("type") == "table":
                html = item.get("res", {}).get("html", "")
                if html:
                    return html_table_to_markdown(html)

        raise ValueError("No table found in PPStructure result")

    def _grid_table_to_markdown(self, table_img: np.ndarray) -> str:
        """
        Detect table grid lines → segment cells → OCR each cell → build Markdown.
        """
        from src.utils.table_extractor import GridTableExtractor
        extractor = GridTableExtractor()
        return extractor.extract(table_img, ocr_fn=self._run_ocr)

    # ── Figure/Chart Description ──────────────────────────────────────────────

    def _describe_figure(self, figure_img: np.ndarray, region: DocumentRegion) -> str:
        """
        Send chart/figure image to LLaVA-7B via Ollama for textual description.
        """
        try:
            from src.config import CHART_DESCRIPTION_PROMPT
            img_b64 = self._image_to_base64(figure_img)

            description = self.ollama.vision_query(
                prompt=CHART_DESCRIPTION_PROMPT,
                image_base64=img_b64
            )

            return f"[FIGURE DESCRIPTION - Page {region.page_num + 1}]\n{description}"

        except Exception as e:
            logger.error(f"Figure description failed: {e}")
            return f"[FIGURE - Page {region.page_num + 1}] (Description unavailable)"

    @staticmethod
    def _image_to_base64(img: np.ndarray) -> str:
        pil_img = Image.fromarray(img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    # ── Page Summary Generation ───────────────────────────────────────────────

    def _build_page_text(self, regions: list[DocumentRegion]) -> str:
        """Concatenate all region content for a page."""
        parts = []
        for r in regions:
            if r.content:
                prefix = {
                    "title": "# ",
                    "table": "",
                    "figure": "",
                }.get(r.region_type, "")
                parts.append(f"{prefix}{r.content}")
        return "\n\n".join(parts)

    def _generate_page_summary(self, regions: list[DocumentRegion]) -> str:
        """
        Generate a brief summary of the page content for metadata enrichment.
        Uses Llama3 for a concise 1-2 sentence summary.
        """
        text_sample = self._build_page_text(regions)[:1500]
        if not text_sample.strip():
            return "Empty page"

        region_types = list({r.region_type for r in regions})

        try:
            summary = self.ollama.query(
                prompt=f"""Tóm tắt nội dung trang tài liệu sau trong 1-2 câu ngắn gọn.
Loại nội dung: {', '.join(region_types)}

Nội dung:
{text_sample}

Tóm tắt:""",
                system="Bạn là trợ lý tóm tắt tài liệu. Chỉ trả lời bằng 1-2 câu súc tích.",
            )
            return summary.strip()
        except Exception:
            # Fallback: use first 200 chars
            return text_sample[:200].replace("\n", " ") + "..."
