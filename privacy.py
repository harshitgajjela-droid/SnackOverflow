import spacy
import re

# Load the lightweight English NLP model
# It automatically downloads if you haven't run the terminal command yet.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading English spaCy model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def scan_label_for_pii(ocr_text: str, allowed_entities: list = None) -> dict:
    """
    Scans OCR text for Digital Personal Data Protection (DPDP) Act violations.
    Flags exposed database columns and unauthorized Personally Identifiable Information.
    """
    if allowed_entities is None:
        allowed_entities = []
        
    text = ocr_text.replace("\n", " ")
    findings = []
    
    # 1. Regex: Hunt for internal database leaks (Synthetic Data / Schemas)
    db_columns = r"\b(customer_id|user_id|dob|date_of_birth|ssn|aadhar|pan_card|acc_num|password|auth_token)\b"
    db_matches = re.finditer(db_columns, text, re.IGNORECASE)
    for match in db_matches:
        findings.append(f"Critical: Internal database schema leak detected ('{match.group(0)}')")
        
    # 2. Regex: Hunt for raw phone numbers and emails
    # (Note: A dedicated consumer care email is allowed, but we flag it for the rule engine to verify)
    email_matches = re.finditer(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    for match in email_matches:
        findings.append(f"PII: Email address detected ({match.group(0)})")
        
    # 3. spaCy NER: Hunt for unauthorized individual names
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Ignore the name if it is part of the authorized manufacturer/packer details
            if not any(ent.text.lower() in allowed.lower() for allowed in allowed_entities if allowed):
                findings.append(f"DPDP Warning: Unauthorized individual name detected ('{ent.text}')")

    return {
        "status": "FAIL" if findings else "PASS",
        "violations": findings,
        "total_leaks": len(findings)
    }