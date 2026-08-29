"""Extract legally relevant declarations from OCR line records.

Input lines must use: {text, bbox, confidence?, page?, avg_height?}.
`bbox` is [x1, y1, x2, y2] in original-image pixels.  Keep front and back OCR
lines in one array; `page` identifies the image.  No database guessing occurs.
"""
from __future__ import annotations

import re
from statistics import mean
from typing import Any, Callable, Mapping, Optional


Line = Mapping[str, Any]
Record = dict[str, Any]
ROLE_RE = {
    "manufacturer": re.compile(r"\b(?:manufactured|made)\s+by\b", re.I),
    "packer": re.compile(r"\b(?:packed|marketed)\s+by\b", re.I),
    "importer": re.compile(r"\bimported\s+by\b", re.I),
}
LABEL_RE = re.compile(r"\b(?:manufactured|made|packed|marketed|imported)\s+by|\b(?:mrp|net\s*(?:qty|quantity|wt|weight|content|vol|volume)|best\s*before|use\s*by|consumer\s*care|customer\s*care|country\s*of\s*origin)\b", re.I)
MRP_RE = re.compile(r"\b(?:m\.?\s*r\.?\s*p\.?|maximum\s+retail\s+price)\b\s*[:\-]?\s*(?:rs\.?|inr|₹)?\s*(\d+(?:[,.]\d{1,2})?)", re.I)
QUANTITY_RE = re.compile(r"\bnet\s*(?:qty|quantity|wt\.?|weight|content|vol\.?|volume)?\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(mg|g|gm|gms|kg|kgs|ml|l|ltr|litre|litres|pcs|pieces?|nos?\.?|n)\b", re.I)
DATE_LABEL_RE = re.compile(r"\b(?:mfg\.?|mfd\.?|manufactur(?:ed|ing)?|pack(?:ed|ing)?|import(?:ed)?)(?:\s*date)?\b\s*[:\-]?\s*([a-z]{3,9}\s*[-/]?\s*\d{2,4}|\d{1,2}\s*[-/]\s*\d{2,4})", re.I)
EXPIRY_LABEL_RE = re.compile(r"\b(?:best\s*before|use\s*by|expiry\s*date|exp\.?\s*date|expires?\s*(?:on|by)?)\b\s*[:\-]?\s*([a-z0-9 ./-]{4,25})", re.I)
ORIGIN_RE = re.compile(r"\b(?:country\s+of\s+origin|origin|made\s+in)\b\s*[:\-]?\s*([a-z][a-z .'-]{1,60})", re.I)
UNIT_MAP = {"g": "g", "gm": "g", "gms": "g", "kg": "kg", "kgs": "kg", "ml": "ml", "l": "l", "ltr": "l", "litre": "l", "litres": "l", "pc": "count", "pcs": "count", "piece": "count", "pieces": "count", "no": "count", "nos": "count", "n": "count"}
PHONE_RE = re.compile(r"(?:\+91[- ]?)?[6-9]\d{9}|1800[- ]?\d{3,4}[- ]?\d{3,4}")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def _evidence(line: Line) -> dict[str, Any]:
    return {key: line[key] for key in ("text", "bbox", "confidence", "page", "avg_height") if key in line}


def _record(value: Any, lines: list[Line]) -> Record:
    confidences = [float(line["confidence"]) for line in lines if line.get("confidence") is not None]
    return {"value": value, "evidence": [_evidence(line) for line in lines],
            "confidence": round(mean(confidences), 4) if confidences else None}


def _find(lines: list[Line], pattern: re.Pattern[str], parse: Callable[[re.Match[str]], Any]) -> Optional[Record]:
    for line in lines:
        match = pattern.search(str(line.get("text", "")))
        if match:
            return _record(parse(match), [line])
    return None


def _after_label(text: str, match: re.Match[str]) -> str:
    return text[match.end():].lstrip(" :,-")


def _entity(lines: list[Line], role: str) -> Optional[Record]:
    label = ROLE_RE[role]
    for index, line in enumerate(lines):
        text = str(line.get("text", ""))
        match = label.search(text)
        if not match:
            continue
        chunks = [_after_label(text, match)]
        used = [line]
        # Addresses commonly wrap over the next two OCR lines. Stop at another label.
        for next_line in lines[index + 1:index + 3]:
            candidate = str(next_line.get("text", "")).strip()
            if not candidate or LABEL_RE.search(candidate):
                break
            chunks.append(candidate)
            used.append(next_line)
        content = " ".join(part for part in chunks if part).strip()
        if not content:
            continue
        # A reliable brand/address split needs NER; retain the complete declaration
        # and let a reviewer correct the name if only a long single line is available.
        name, _, address = content.partition(",")
        return _record({"name": name.strip(), "address": address.strip() or content}, used)
    return None


def _consumer_care(lines: list[Line]) -> Optional[Record]:
    label = re.compile(r"\b(?:consumer|customer)\s*care\b", re.I)
    for index, line in enumerate(lines):
        text = str(line.get("text", ""))
        match = label.search(text)
        if not match:
            continue
        values = [_after_label(text, match)]
        used = [line]
        for next_line in lines[index + 1:index + 4]:
            candidate = str(next_line.get("text", "")).strip()
            if not candidate or (LABEL_RE.search(candidate) and not label.search(candidate)):
                break
            values.append(candidate)
            used.append(next_line)
        content = " ".join(values)
        phone = PHONE_RE.search(content)
        email = EMAIL_RE.search(content)
        name = values[0].strip() if values and values[0].strip() else None
        return _record({"name": name, "address": content, "phone": phone.group(0) if phone else None,
                        "email": email.group(0) if email else None}, used)
    return None


def _mrp(lines: list[Line]) -> Optional[Record]:
    for index, line in enumerate(lines):
        match = MRP_RE.search(str(line.get("text", "")))
        if not match:
            continue
        used = [line]
        text = str(line["text"]).strip()
        # OCR engines often put '(inclusive of all taxes)' on the following line.
        if "tax" not in text.lower() and index + 1 < len(lines):
            next_text = str(lines[index + 1].get("text", "")).strip()
            if "tax" in next_text.lower():
                text = f"{text} {next_text}"
                used.append(lines[index + 1])
        return _record(text, used)
    return None


def extract_information(normalized_ocr: Mapping[str, Any]) -> dict[str, Optional[Record]]:
    """Return extractor records for use with ocr_adapter.from_extraction()."""
    lines = [line for line in normalized_ocr.get("lines", []) if str(line.get("text", "")).strip()]
    mrp = _mrp(lines)
    quantity = _find(lines, QUANTITY_RE, lambda m: f"{m.group(1)} {UNIT_MAP.get(m.group(2).lower().rstrip('.'), m.group(2))}")
    return {
        "mrp_declaration": mrp,
        "net_quantity": quantity,
        "manufacture_pack_import_month_year": _find(lines, DATE_LABEL_RE, lambda m: m.group(1).strip()),
        "best_before_month_year": _find(lines, EXPIRY_LABEL_RE, lambda m: m.group(1).strip()),
        "country_of_origin": _find(lines, ORIGIN_RE, lambda m: m.group(1).strip()),
        "manufacturer": _entity(lines, "manufacturer"),
        "packer": _entity(lines, "packer"),
        "importer": _entity(lines, "importer"),
        "consumer_care": _consumer_care(lines),
        # These should normally be supplied by the product-classification/layout
        # stage. Do not guess a generic name from a brand string.
        "generic_name": _record(normalized_ocr["generic_name"], []) if normalized_ocr.get("generic_name") else None,
        "dimensions": _record(normalized_ocr["dimensions"], []) if normalized_ocr.get("dimensions") else None,
    }
