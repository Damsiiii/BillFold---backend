"""OCR-based extraction for photographed or scanned bank statements."""
import io
import logging
import math
from pathlib import Path
from typing import Any, BinaryIO, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from models.transaction import ExtractionResult
from services.file_validator import MAX_IMAGE_PIXELS
from services.pdf_parser import PDFParser, _get_ocr_engine
from services.transaction_normalizer import (
    correct_dates_from_leading_descriptions,
    extract_statement_currency,
)

logger = logging.getLogger(__name__)
MIN_OCR_CONFIDENCE = 0.70


class ImageParser:
    """Extract transactions from supported statement images using PaddleOCR."""

    def parse(
        self,
        file_source: Union[str, Path, bytes, BinaryIO],
        filename: str,
        file_type: str,
        dayfirst: Optional[bool] = None,
    ) -> ExtractionResult:
        raw_bytes, read_error = self._read_bytes(file_source)
        if read_error or not raw_bytes:
            return self._failure(
                filename,
                file_type,
                read_error or "Image file is empty.",
            )

        try:
            with Image.open(io.BytesIO(raw_bytes)) as source_image:
                expected_format = "JPEG" if file_type in {"jpg", "jpeg"} else "PNG"
                if source_image.format != expected_format:
                    return self._failure(
                        filename,
                        file_type,
                        f"Image content does not match {file_type.upper()} format.",
                    )
                if source_image.width * source_image.height > MAX_IMAGE_PIXELS:
                    return self._failure(
                        filename,
                        file_type,
                        "Image dimensions exceed the supported OCR limit.",
                    )
                source_image.load()
                oriented_image = ImageOps.exif_transpose(source_image)
                oriented_image.thumbnail((4000, 4000), Image.Resampling.LANCZOS)
                image = np.asarray(oriented_image.convert("RGB"))
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            return self._failure(
                filename,
                file_type,
                f"Image is unreadable or corrupted ({file_type.upper()}).",
            )

        try:
            ocr_engine = _get_ocr_engine()
            predictions = ocr_engine.predict(image)
        except ImportError:
            return self._failure(
                filename,
                file_type,
                "Image statement OCR is not configured in this environment.",
            )
        except Exception:
            logger.exception(
                "OCR failed while processing statement image",
                extra={"source_filename": filename, "stage": "image_ocr_failed"},
            )
            return self._failure(
                filename,
                file_type,
                "OCR could not process this image. Please try a clearer image.",
            )

        try:
            words, ignored_regions = self._recognized_words(predictions)
        except (IndexError, KeyError, TypeError, ValueError):
            logger.warning(
                "OCR returned malformed statement image data",
                exc_info=True,
                extra={
                    "source_filename": filename,
                    "stage": "image_ocr_output_invalid",
                },
            )
            return self._failure(
                filename,
                file_type,
                "OCR returned malformed data; no transactions were extracted.",
                warnings=["OCR output was incomplete or malformed."],
            )
        if not words:
            warnings = []
            if ignored_regions:
                warnings.append(
                    f"Ignored {ignored_regions} low-confidence or malformed OCR text region(s)."
                )
            else:
                warnings.append(
                    "OCR returned no usable text regions; verify that the image is clear."
                )
            return self._failure(
                filename,
                file_type,
                "OCR could not confidently recognize readable statement text.",
                warnings=warnings,
            )

        parser = PDFParser(dayfirst=dayfirst)
        page_text = self._words_as_text(words)
        currency = extract_statement_currency(page_text)
        positioned_transactions, positioned_warnings = (
            parser._extract_from_positioned_words(
                [words], filename, currency, dayfirst
            )
        )
        line_transactions, line_warnings = parser._extract_from_text(
            [page_text], filename, currency, dayfirst
        )

        # Prefer coordinate-based column mapping whenever it yields records;
        # otherwise use the line parser. Combining both can reinterpret the same
        # OCR row as a second transaction.
        preferred_transactions = positioned_transactions or line_transactions
        transactions = [
            transaction
            for transaction in preferred_transactions
            if transaction.description.strip()
            and transaction.description != "Unknown Transaction"
            and transaction.transaction_type != "unknown"
        ]
        correct_dates_from_leading_descriptions(transactions, dayfirst=dayfirst)
        warnings = positioned_warnings + line_warnings
        rejected_rows = len(preferred_transactions) - len(transactions)
        if rejected_rows:
            warnings.append(
                f"Skipped {rejected_rows} OCR transaction row(s) with missing "
                "descriptions or unclear debit/credit direction."
            )
        if ignored_regions:
            warnings.append(
                f"Ignored {ignored_regions} low-confidence OCR text region(s); "
                "extraction may be incomplete."
            )
        if not transactions:
            warnings.append(
                "No valid transactions could be confidently extracted from this image."
            )
            return self._failure(
                filename,
                file_type,
                "OCR did not find any sufficiently reliable transaction rows.",
                warnings=warnings,
            )

        return ExtractionResult(
            success=True,
            filename=filename,
            file_type=file_type,
            total_transactions=len(transactions),
            transactions=transactions,
            warnings=warnings,
            error=None,
        )

    @staticmethod
    def _read_bytes(
        file_source: Union[str, Path, bytes, BinaryIO],
    ) -> Tuple[Optional[bytes], Optional[str]]:
        if isinstance(file_source, bytes):
            return file_source, None
        if isinstance(file_source, (str, Path)):
            path = Path(file_source)
            try:
                return path.read_bytes(), None
            except OSError as exc:
                return None, f"Could not read image file: {exc}"
        if hasattr(file_source, "read"):
            try:
                if hasattr(file_source, "seek"):
                    file_source.seek(0)
                data = file_source.read()
                if hasattr(file_source, "seek"):
                    file_source.seek(0)
                return (data.encode("latin-1") if isinstance(data, str) else data), None
            except (OSError, ValueError) as exc:
                return None, f"Could not read image stream: {exc}"
        return None, "Unsupported image source type."

    @staticmethod
    def _recognized_words(
        predictions: Any,
    ) -> Tuple[List[dict[str, Any]], int]:
        words: List[dict[str, Any]] = []
        ignored_regions = 0

        def as_list(value: Any) -> Any:
            return value.tolist() if hasattr(value, "tolist") else value

        if predictions is None:
            return words, ignored_regions
        for prediction in predictions:
            result = getattr(prediction, "json", {})
            if not isinstance(result, dict):
                ignored_regions += 1
                continue
            result_data = result.get("res", {})
            if not isinstance(result_data, dict):
                ignored_regions += 1
                continue

            texts = as_list(result_data.get("rec_texts", []))
            boxes = as_list(result_data.get("rec_boxes", []))
            scores = as_list(result_data.get("rec_scores"))
            polygons = as_list(result_data.get("rec_polys", []))
            if not isinstance(texts, (list, tuple)):
                ignored_regions += 1
                continue
            boxes = boxes if isinstance(boxes, (list, tuple)) else []
            polygons = polygons if isinstance(polygons, (list, tuple)) else []

            for index, recognized_text in enumerate(texts):
                text = str(recognized_text).strip()
                if not text:
                    continue
                if scores is not None:
                    try:
                        confidence = float(scores[index])
                    except (IndexError, TypeError, ValueError):
                        ignored_regions += 1
                        continue
                    if not math.isfinite(confidence) or confidence < MIN_OCR_CONFIDENCE:
                        ignored_regions += 1
                        continue

                box = boxes[index] if index < len(boxes) else None
                if (box is None or len(box) < 4) and index < len(polygons):
                    polygon = polygons[index]
                    if len(polygon) >= 4 and all(len(point) >= 2 for point in polygon):
                        box = [
                            min(point[0] for point in polygon),
                            min(point[1] for point in polygon),
                            max(point[0] for point in polygon),
                            max(point[1] for point in polygon),
                        ]
                if box is None or len(box) < 4:
                    ignored_regions += 1
                    continue
                x_min, y_min, x_max, y_max = map(float, box[:4])
                if (
                    not all(
                        math.isfinite(value)
                        for value in (x_min, y_min, x_max, y_max)
                    )
                    or x_max <= x_min
                    or y_max <= y_min
                ):
                    ignored_regions += 1
                    continue
                words.append({
                    "x0": x_min,
                    "x1": x_max,
                    "top": y_min,
                    "bottom": y_max,
                    "text": text,
                })
        return words, ignored_regions

    @staticmethod
    def _words_as_text(words: List[dict[str, Any]]) -> str:
        lines = PDFParser._group_positioned_words(words)
        return "\n".join(
            " ".join(str(word["text"]) for word in line) for line in lines
        )

    @staticmethod
    def _failure(
        filename: str,
        file_type: str,
        error: str,
        warnings: Optional[List[str]] = None,
    ) -> ExtractionResult:
        return ExtractionResult(
            success=False,
            filename=filename,
            file_type=file_type,
            total_transactions=0,
            transactions=[],
            warnings=warnings or [],
            error=error,
        )


def parse_image(
    file_source: Union[str, Path, bytes, BinaryIO],
    filename: str,
    file_type: str,
    dayfirst: Optional[bool] = None,
) -> ExtractionResult:
    """Convenience function to OCR and parse a JPG, JPEG, or PNG statement."""
    return ImageParser().parse(file_source, filename, file_type, dayfirst)
