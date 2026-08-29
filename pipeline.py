"""One-call integration point for the image/OCR backend."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from extractor import extract_information
from ocr_adapter import from_extraction
from rule_engine import RuleEngine


def analyze_ocr(normalized_ocr: Mapping[str, Any], **classification: Any) -> dict[str, Any]:
    """Extract OCR declarations, normalize them, then evaluate legal rules.

    `classification` should be supplied by the API/classifier, for example:
    is_imported, is_food, manufacturer_is_packer,
    may_become_unfit_for_human_consumption, dimensions_relevant, generic_name,
    and declarations_legible_and_prominent.
    """
    extracted = extract_information(normalized_ocr)
    package = from_extraction(extracted, **classification)
    return RuleEngine(Path(__file__).with_name("rules.json")).analyze(package)
