"""Boundary between OCR/CV output and RuleEngine's normalized package input.

The OCR team should return a field dictionary using the keys below plus optional
evidence (bounding boxes) and confidence. This module refuses silent guessing:
unknown/missing information remains None so the engine returns FAIL or NEEDS_REVIEW.
"""
from __future__ import annotations

from typing import Any, Mapping


FIELD_GUIDE = {
    "generic_name": "Common/generic product name, e.g. 'Biscuits'; do not use the brand alone.",
    "net_quantity": "Exact declared quantity, e.g. '500 g', '1 L', or '12 count'.",
    "manufacture_pack_import_month_year": "Mfg/Pkd/Imported month and year, e.g. '08/2026'.",
    "best_before_month_year": "Best before/use-by month and year, when printed.",
    "mrp_declaration": "Entire declaration, e.g. 'MRP Rs. 120 (inclusive of all taxes)'.",
    "country_of_origin": "Country of origin/manufacture/assembly, when printed.",
    "dimensions": "Dimensions when relevant, e.g. '30 cm x 20 cm'.",
}


def to_package(ocr: Mapping[str, Any], *, package_scope: str = "retail_prepackaged",
               is_food: bool = False, is_imported: bool = False,
               may_become_unfit_for_human_consumption: bool = False,
               dimensions_relevant: bool = False,
               manufacturer_is_packer: bool | None = None) -> dict[str, Any]:
    """Create RuleEngine input from an OCR team's structured result.

    Expected OCR subobjects:
      responsible_party: {name, address}
      consumer_care: {name, address, phone?, email?}
      evidence: {field_name: [{text, confidence, bbox: [x, y, width, height], page?}]}
      field_confidence: {field_name: 0.0..1.0}
    """
    result = {
        "package_scope": package_scope,
        "is_food": is_food,
        "is_imported": is_imported,
        "may_become_unfit_for_human_consumption": may_become_unfit_for_human_consumption,
        "dimensions_relevant": dimensions_relevant,
        "manufacturer_is_packer": manufacturer_is_packer,
        "manufacturer": ocr.get("manufacturer"),
        "packer": ocr.get("packer"),
        "importer": ocr.get("importer"),
        "consumer_care": ocr.get("consumer_care"),
        "evidence": ocr.get("evidence", {}),
        "field_confidence": ocr.get("field_confidence", {}),
        # CV/QA must explicitly set this only after using image geometry and clarity.
        "declarations_legible_and_prominent": ocr.get("declarations_legible_and_prominent", False),
    }
    for field in FIELD_GUIDE:
        result[field] = ocr.get(field)
    return result


def _unwrap(item: Any) -> Any:
    """Turn extractor records of the form {value, evidence, confidence} into values."""
    return item.get("value") if isinstance(item, Mapping) and "value" in item else item


def from_extraction(extracted: Mapping[str, Any], **classification: Any) -> dict[str, Any]:
    """Adapt the output of extractor.extract_information() to RuleEngine input."""
    source_to_target = {
        "generic_name": "generic_name",
        "net_quantity": "net_quantity",
        "manufacture_pack_import_month_year": "manufacture_pack_import_month_year",
        "best_before_month_year": "best_before_month_year",
        "mrp_declaration": "mrp_declaration",
        "country_of_origin": "country_of_origin",
        "dimensions": "dimensions",
        "manufacturer": "manufacturer",
        "packer": "packer",
        "importer": "importer",
        "consumer_care": "consumer_care",
    }
    raw: dict[str, Any] = {}
    evidence: dict[str, list[dict[str, Any]]] = {}
    confidence: dict[str, float] = {}
    for target, source in source_to_target.items():
        record = extracted.get(source)
        raw[target] = _unwrap(record)
        if isinstance(record, Mapping):
            evidence[target] = record.get("evidence", [])
            if record.get("confidence") is not None:
                confidence[target] = float(record["confidence"])
    raw["evidence"] = evidence
    raw["field_confidence"] = confidence
    raw.update(classification)
    return to_package(raw, package_scope=raw.get("package_scope", "retail_prepackaged"),
                      is_food=raw.get("is_food", False), is_imported=raw.get("is_imported", False),
                      may_become_unfit_for_human_consumption=raw.get("may_become_unfit_for_human_consumption", False),
                      dimensions_relevant=raw.get("dimensions_relevant", False),
                      manufacturer_is_packer=raw.get("manufacturer_is_packer"))
