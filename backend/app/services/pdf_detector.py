from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

try:
	from pypdf import PdfReader
except ImportError as exc:
	PdfReader = None
	_PDF_IMPORT_ERROR = exc
else:
	_PDF_IMPORT_ERROR = None

from ..mod7 import explain_message


MAX_PDF_SIZE_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 50
MAX_EXTRACTED_TEXT_CHARS = 100_000
PDF_CONTENT_TYPE = "application/pdf"


class PDFDetectionError(ValueError):
	"""Raised when a PDF cannot be safely extracted for analysis."""


def _require_pdf_parser() -> Any:
	if PdfReader is None:
		raise PDFDetectionError(
			"PDF detection requires the 'pypdf' package. Install backend requirements first."
		) from _PDF_IMPORT_ERROR
	return PdfReader


def _validate_pdf_input(pdf_bytes: bytes, filename: str | None = None) -> None:
	if not isinstance(pdf_bytes, bytes) or not pdf_bytes:
		raise PDFDetectionError("The uploaded PDF is empty.")

	if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
		raise PDFDetectionError(
			f"The PDF is too large. Maximum supported size is {MAX_PDF_SIZE_BYTES // (1024 * 1024)} MB."
		)

	if filename and Path(filename).suffix.lower() != ".pdf":
		raise PDFDetectionError("Only files with a .pdf extension are supported.")

	if not pdf_bytes.startswith(b"%PDF-"):
		raise PDFDetectionError("The uploaded file is not a valid PDF document.")


def extract_pdf_text(pdf_bytes: bytes, filename: str | None = None) -> dict[str, Any]:
	"""Extract bounded text and page metadata from a PDF without opening it."""
	_validate_pdf_input(pdf_bytes, filename)
	reader_class = _require_pdf_parser()

	try:
		reader = reader_class(BytesIO(pdf_bytes), strict=False)
		page_count = len(reader.pages)
	except Exception as exc:
		raise PDFDetectionError("The PDF could not be parsed safely.") from exc

	if page_count == 0:
		raise PDFDetectionError("The PDF does not contain any pages.")
	if page_count > MAX_PDF_PAGES:
		raise PDFDetectionError(
			f"The PDF has too many pages. Maximum supported page count is {MAX_PDF_PAGES}."
		)

	page_text: list[str] = []
	text_length = 0
	for page_number, page in enumerate(reader.pages, start=1):
		try:
			extracted = (page.extract_text() or "").strip()
		except Exception as exc:
			raise PDFDetectionError(
				f"Text could not be extracted from page {page_number}."
			) from exc

		if not extracted:
			continue

		remaining = MAX_EXTRACTED_TEXT_CHARS - text_length
		if remaining <= 0:
			break

		bounded_text = extracted[:remaining]
		page_text.append(f"[Page {page_number}]\n{bounded_text}")
		text_length += len(bounded_text)

	extracted_text = "\n\n".join(page_text).strip()
	if not extracted_text:
		raise PDFDetectionError(
			"No selectable text was found. This PDF may be image-only and requires OCR before analysis."
		)

	return {
		"page_count": page_count,
		"text": extracted_text,
		"text_characters": len(extracted_text),
		"truncated": text_length >= MAX_EXTRACTED_TEXT_CHARS,
	}


def analyze_pdf(
	pdf_bytes: bytes,
	filename: str | None = None,
	content_type: str | None = None,
) -> dict[str, Any]:
	"""Extract a PDF and return the standard ScamShield verdict bundle."""
	if content_type and content_type.lower() != PDF_CONTENT_TYPE:
		raise PDFDetectionError("The uploaded content must have type application/pdf.")

	extracted = extract_pdf_text(pdf_bytes, filename)
	analysis = explain_message(extracted["text"], metadata={"source": "pdf"})
	analysis.update(
		{
			"source": "pdf",
			"filename": filename or "uploaded.pdf",
			"content_type": PDF_CONTENT_TYPE,
			"page_count": extracted["page_count"],
			"text_characters": extracted["text_characters"],
			"text_truncated": extracted["truncated"],
		}
	)
	return analysis
