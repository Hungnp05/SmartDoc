"""
Table Extractor Utilities
──────────────────────────
Converts detected table regions to clean Markdown format.
Two strategies:
  1. HTML → Markdown (from PPStructure output)
  2. Grid-line detection → cell segmentation → OCR (fallback)
"""

import re
import logging
import numpy as np

logger = logging.getLogger(__name__)


# HTML to Markdown

def html_table_to_markdown(html: str) -> str:
    """
    Convert an HTML table string to clean Markdown format.
    Handles rowspan, colspan, and nested content.
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            return html

        rows = table.find_all("tr")
        if not rows:
            return html

        # Build a 2D grid to handle rowspan/colspan
        grid = {}
        row_idx = 0

        for row in rows:
            col_idx = 0
            for cell in row.find_all(["td", "th"]):
                # Skip occupied cells
                while (row_idx, col_idx) in grid:
                    col_idx += 1

                text = cell.get_text(separator=" ").strip()
                text = re.sub(r'\s+', ' ', text)

                rowspan = int(cell.get("rowspan", 1))
                colspan = int(cell.get("colspan", 1))

                # Fill grid
                for r in range(rowspan):
                    for c in range(colspan):
                        grid[(row_idx + r, col_idx + c)] = text if r == 0 and c == 0 else ""

                col_idx += colspan
            row_idx += 1

        # Convert grid to markdown
        if not grid:
            return html

        max_row = max(r for r, _ in grid.keys()) + 1
        max_col = max(c for _, c in grid.keys()) + 1

        md_rows = []
        for r in range(max_row):
            cells = [grid.get((r, c), "") for c in range(max_col)]
            md_rows.append("| " + " | ".join(cells) + " |")

            # Add separator after header row
            if r == 0:
                separator = "|" + "|".join([" --- " for _ in range(max_col)]) + "|"
                md_rows.append(separator)

        return "\n".join(md_rows)

    except ImportError:
        logger.warning("BeautifulSoup not available, using regex HTML parser")
        return _regex_html_to_markdown(html)
    except Exception as e:
        logger.error(f"HTML table conversion failed: {e}")
        return html


def _regex_html_to_markdown(html: str) -> str:
    """Simple regex-based HTML table to Markdown fallback."""
    html = re.sub(r'<br\s*/?>', ' ', html, flags=re.IGNORECASE)
    html = re.sub(r'<[^>]+>', ' ', html)
    html = re.sub(r'\s+', ' ', html).strip()
    return html


# Grid-Line Table Extractor

class GridTableExtractor:
    """
    Extracts table structure from an image by:
    1. Detecting horizontal and vertical lines (table borders)
    2. Finding cell intersections
    3. OCR-ing each cell
    4. Assembling into Markdown
    """

    def __init__(
        self,
        min_line_length_ratio: float = 0.3,
        line_threshold: int = 15,
    ):
        self.min_line_length_ratio = min_line_length_ratio
        self.line_threshold = line_threshold

    def extract(self, table_img: np.ndarray, ocr_fn) -> str:
        """
        Main extraction method.
        ocr_fn: callable(image_array) -> str
        """
        import cv2

        gray = cv2.cvtColor(table_img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Detect horizontal lines
        h_kernel_len = max(int(w * self.min_line_length_ratio), 20)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)

        # Detect vertical lines
        v_kernel_len = max(int(h * self.min_line_length_ratio), 20)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)

        # Combine
        grid = cv2.add(h_lines, v_lines)

        # Find contours of cells
        contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 4:
            # Not enough structure detected — fall back to raw OCR
            logger.warning("Grid detection found insufficient structure")
            return f"[TABLE]\n{ocr_fn(table_img)}"

        # Get bounding boxes for cells
        cells = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # Filter out very small regions (noise) and the full table bounding box
            if cw > 20 and ch > 10 and cw < w * 0.98 and ch < h * 0.98:
                cells.append((x, y, x + cw, y + ch))

        if not cells:
            return f"[TABLE]\n{ocr_fn(table_img)}"

        # Organize cells into rows/columns
        return self._cells_to_markdown(cells, table_img, ocr_fn)

    def _cells_to_markdown(self, cells, table_img, ocr_fn) -> str:
        """Group cells by row, OCR each, build Markdown table."""
        # Sort cells by y then x
        cells_sorted = sorted(cells, key=lambda c: (c[1], c[0]))

        # Group into rows (cells with similar y values)
        rows = []
        current_row = [cells_sorted[0]]
        row_y = cells_sorted[0][1]

        for cell in cells_sorted[1:]:
            if abs(cell[1] - row_y) < 15:
                current_row.append(cell)
            else:
                rows.append(sorted(current_row, key=lambda c: c[0]))
                current_row = [cell]
                row_y = cell[1]
        if current_row:
            rows.append(sorted(current_row, key=lambda c: c[0]))

        # OCR each cell
        md_rows = []
        for r_idx, row in enumerate(rows):
            cells_text = []
            for x1, y1, x2, y2 in row:
                cell_img = table_img[y1:y2, x1:x2]
                if cell_img.size > 0:
                    text = ocr_fn(cell_img).replace("\n", " ").strip()
                else:
                    text = ""
                cells_text.append(text)

            md_rows.append("| " + " | ".join(cells_text) + " |")
            if r_idx == 0:
                md_rows.append("|" + "|".join([" --- " for _ in cells_text]) + "|")

        return "\n".join(md_rows) if md_rows else "[TABLE - Empty]"
