"""Auditable, configuration-driven package-declaration checks.

This module is a decision-support tool, not legal advice or a substitute for
inspection.  It never turns absent/low-confidence OCR or unsupported visual
requirements into a PASS.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Optional


OUTCOME_FAIL = "FAIL"
OUTCOME_PASS = "PASS"
OUTCOME_REVIEW = "NEEDS_REVIEW"
OUTCOME_NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class Rule:
    id: str
    name: str
    provision: str
    severity: str
    applies_when: dict[str, Any]
    assertion: dict[str, Any]
    message: str
    fix: str


@dataclass
class Finding:
    rule_id: str
    rule_name: str
    outcome: str
    severity: str
    field: Optional[str]
    observed: Any
    requirement: str
    message: str
    fix: str
    provision: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConfigurationError(ValueError):
    pass


QUANTITY_RE = re.compile(
    r"^\s*(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>mg|g|kg|ml|l|m|cm|mm|sq\.?\s*m|m2|cm2|count|nos?\.?|pieces?)\s*$",
    re.IGNORECASE,
)
MONTH_YEAR_RE = re.compile(
    r"^\s*(?:(?P<month_num>0?[1-9]|1[0-2])\s*[-/]\s*(?P<year_num>\d{4})|"
    r"(?P<month_name>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+?(?P<year_name>\d{4}))\s*$",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
PHONE_RE = re.compile(r"^(?:\+91[-\s]?)?[6-9]\d{9}$|^1800[-\s]?\d{3,4}[-\s]?\d{3,4}$")
PIN_RE = re.compile(r"\b[1-9]\d{5}\b")


def _present(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _as_bool(value: Any) -> bool:
    return value is True


def parse_quantity(value: Any) -> Optional[dict[str, str]]:
    if not _present(value):
        return None
    match = QUANTITY_RE.fullmatch(str(value))
    if not match:
        return None
    amount = Decimal(match.group("amount"))
    unit = match.group("unit").lower().replace(".", "").replace(" ", "")
    aliases = {"nos": "count", "no": "count", "piece": "count", "pieces": "count"}
    return {"amount": format(amount, "f"), "unit": aliases.get(unit, unit)}


def valid_month_year(value: Any) -> bool:
    return bool(_present(value) and MONTH_YEAR_RE.fullmatch(str(value)))


def parse_mrp(value: Any) -> Optional[dict[str, str]]:
    """Accept only a complete MRP declaration, including all-taxes wording."""
    if not _present(value):
        return None
    text = " ".join(str(value).replace("₹", " Rs. ").split())
    match = re.fullmatch(
        r"(?:MRP|Maximum\s+Retail\s+Price)\s*[:\-]?\s*(?:Rs\.?|INR)\s*"
        r"(?P<amount>\d+(?:\.\d{1,2})?)\s*(?:/\-)?\s*"
        r"(?:\(?\s*(?:incl(?:usive)?\.?\s*(?:of)?\s*all\s*taxes|inclusive\s+of\s+taxes)\s*\)?)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    try:
        amount = Decimal(match.group("amount"))
    except InvalidOperation:
        return None
    if amount < 0:
        return None
    return {"amount": format(amount, "f")}


def _condition_matches(condition: Mapping[str, Any], data: Mapping[str, Any]) -> bool:
    for key, expected in condition.items():
        actual = data.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


class RuleEngine:
    def __init__(self, rules_path: str | Path | None = None):
        path = Path(rules_path) if rules_path else Path(__file__).with_name("rules.json")
        self.rules, self.metadata = self._load_rules(path)

    @staticmethod
    def _load_rules(path: Path) -> tuple[list[Rule], dict[str, Any]]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"Cannot load rules from {path}: {exc}") from exc
        if not isinstance(raw, dict) or not isinstance(raw.get("rules"), list):
            raise ConfigurationError("rules.json must be an object containing a rules array")
        required = {"id", "name", "provision", "severity", "applies_when", "assertion", "message", "fix"}
        rules: list[Rule] = []
        seen: set[str] = set()
        for item in raw["rules"]:
            missing = required - set(item)
            if missing:
                raise ConfigurationError(f"Rule missing keys {sorted(missing)}: {item!r}")
            if item["id"] in seen:
                raise ConfigurationError(f"Duplicate rule id: {item['id']}")
            if item["severity"] not in {"HIGH", "MEDIUM", "LOW"}:
                raise ConfigurationError(f"Invalid severity in {item['id']}")
            if not isinstance(item["applies_when"], dict) or not isinstance(item["assertion"], dict):
                raise ConfigurationError(f"Invalid condition/assertion in {item['id']}")
            seen.add(item["id"])
            rules.append(Rule(**item))
        return rules, raw.get("metadata", {})

    def _evaluate(self, rule: Rule, data: Mapping[str, Any]) -> Finding:
        assertion = rule.assertion
        kind = assertion.get("kind")
        field_name = assertion.get("field")
        observed = data.get(field_name) if field_name else None
        evidence = list(data.get("evidence", {}).get(field_name, [])) if field_name else []
        confidence = data.get("field_confidence", {}).get(field_name) if field_name else None
        minimum_confidence = assertion.get("min_confidence", 0.0)

        def finding(outcome: str, requirement: str) -> Finding:
            return Finding(rule.id, rule.name, outcome, rule.severity, field_name, observed,
                           requirement, rule.message, rule.fix, rule.provision, evidence)

        if kind == "presence":
            return finding(OUTCOME_PASS if _present(observed) else OUTCOME_FAIL, "Declaration must be present.")
        if kind == "text_components":
            components = assertion.get("components", [])
            if not isinstance(observed, Mapping):
                return finding(OUTCOME_REVIEW, "Structured extraction is required for this declaration.")
            absent = [name for name in components if not _present(observed.get(name))]
            return finding(OUTCOME_FAIL if absent else OUTCOME_PASS,
                           "Required components: " + ", ".join(components))
        if kind == "party":
            if not isinstance(observed, Mapping):
                return finding(OUTCOME_FAIL, "A structured name-and-address declaration is required.")
            if not _present(observed.get("name")) or not _present(observed.get("address")):
                return finding(OUTCOME_FAIL, "The party name and address must be declared.")
            # Address completeness needs jurisdiction/context; a missing Indian PIN is
            # reviewable rather than automatically treated as a compliant address.
            if not PIN_RE.search(str(observed["address"])):
                return finding(OUTCOME_REVIEW, "Address is present but requires review for completeness (no Indian PIN detected).")
            return finding(OUTCOME_PASS, "The party name and address are present; a six-digit Indian PIN was detected.")
        if kind == "boolean_confirmation":
            if observed is None:
                return finding(OUTCOME_REVIEW, "This applicability fact was not classified from the package.")
            return finding(OUTCOME_PASS if isinstance(observed, bool) else OUTCOME_REVIEW,
                           "A boolean classification is required.")
        if kind == "quantity":
            result = parse_quantity(observed)
            return finding(OUTCOME_PASS if result else OUTCOME_FAIL,
                           "A positive quantity in a supported standard unit or count is required.")
        if kind == "month_year":
            return finding(OUTCOME_PASS if valid_month_year(observed) else OUTCOME_FAIL,
                           "Month and year must be declared (for example, 08/2026 or August 2026).")
        if kind == "mrp":
            return finding(OUTCOME_PASS if parse_mrp(observed) else OUTCOME_FAIL,
                           "Use MRP/Rs amount and state that it is inclusive of all taxes.")
        if kind == "contact":
            if not isinstance(observed, Mapping):
                return finding(OUTCOME_REVIEW, "Structured consumer-care extraction is required.")
            address_ok = _present(observed.get("address"))
            phone_ok = _present(observed.get("phone")) and bool(PHONE_RE.fullmatch(str(observed["phone"]).replace(" ", "")))
            email_ok = _present(observed.get("email")) and bool(EMAIL_RE.fullmatch(str(observed["email"])))
            name_ok = _present(observed.get("name"))
            return finding(OUTCOME_PASS if name_ok and address_ok and (phone_ok or email_ok) else OUTCOME_FAIL,
                           "Consumer-care name, address, and a valid telephone number or email are required.")
        if kind == "visual_review":
            if confidence is not None and confidence < minimum_confidence:
                return finding(OUTCOME_REVIEW, "OCR confidence is below the configured threshold.")
            if _as_bool(data.get(assertion.get("confirmation_field"))):
                return finding(OUTCOME_PASS, "A vision component confirmed this visual requirement.")
            return finding(OUTCOME_REVIEW, "Requires image geometry/legibility verification; OCR text alone is insufficient.")
        raise ConfigurationError(f"Unsupported assertion kind {kind!r} in {rule.id}")

    def analyze(self, package: Mapping[str, Any]) -> dict[str, Any]:
        """Analyze a normalized package record. See README for the input contract."""
        scope = package.get("package_scope")
        if scope != "retail_prepackaged":
            return {"outcome": OUTCOME_NOT_APPLICABLE, "findings": [],
                    "summary": "This configuration evaluates retail pre-packaged commodities only."}
        findings: list[Finding] = []
        skipped: list[str] = []
        for rule in self.rules:
            if _condition_matches(rule.applies_when, package):
                findings.append(self._evaluate(rule, package))
            else:
                skipped.append(rule.id)
        failures = [item for item in findings if item.outcome == OUTCOME_FAIL]
        reviews = [item for item in findings if item.outcome == OUTCOME_REVIEW]
        if failures:
            outcome = OUTCOME_FAIL
        elif reviews:
            outcome = OUTCOME_REVIEW
        else:
            outcome = OUTCOME_PASS
        return {
            "outcome": outcome,
            "legal_basis_version": self.metadata.get("legal_basis_version"),
            "disclaimer": self.metadata.get("disclaimer"),
            "findings": [item.to_dict() for item in findings],
            "skipped_rule_ids": skipped,
            "counts": {"fail": len(failures), "needs_review": len(reviews), "pass": sum(i.outcome == OUTCOME_PASS for i in findings)},
        }
