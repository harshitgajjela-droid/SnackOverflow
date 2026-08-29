import streamlit as st
from PIL import Image
import cv2
import numpy as np
from modules.privacy import scan_label_for_pii

# Import your custom AI and Compliance modules
from modules.image_processing import load_image, resize_image, deskew_image, generate_ocr_versions
from modules.ocr import run_ocr, draw_ocr_boxes
from modules.classifier import get_product_classification
from pipeline import analyze_ocr

# ==========================================
# UI: PAGE CONFIG & SIDEBAR BRANDING
# ==========================================
st.set_page_config(page_title="SIH 2026 | Compliance Engine", layout="wide", initial_sidebar_state="expanded")

with st.sidebar:
    st.title("🛡️ Snack Overflow")
    st.markdown("### SIH 2026 Submission")
    st.divider()
    st.info("**Engine:** Legal Metrology Rule Auditor")
    st.markdown("Automatically scans packaged commodities for Digital Personal Data Protection Act and Metrology rule compliance.")
    st.divider()
    st.success("Hardware Acceleration: Active")

# ==========================================
# UI: MAIN DASHBOARD HEADER
# ==========================================
st.title("🏷️ Automated Label Audit")
st.markdown("Upload a product package image to instantly detect regulatory violations.")

uploaded_file = st.file_uploader("Insert Product Label Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    if st.button("Execute Compliance Audit", type="primary"):
        with st.spinner("Extracting text and auditing legal rules..."):
            
            # --- BACKEND PROCESSING ---
            temp_path = "temp_label.jpg"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            image = load_image(temp_path)
            resized_image = resize_image(image)
            deskewed_image, _ = deskew_image(resized_image)
            ocr_versions = generate_ocr_versions(deskewed_image)
            
            # Run OCR 
            result = run_ocr(ocr_versions["enhanced"])
            
            # Generate the bounding box visualization
            ocr_vis_bgr = draw_ocr_boxes(ocr_versions["enhanced"], result["detections"])
            ocr_vis_rgb = cv2.cvtColor(ocr_vis_bgr, cv2.COLOR_BGR2RGB) # Fix colors for web
            
            # Run Compliance Engine
            normalized_ocr = {"lines": result["detections"], "generic_name": None, "dimensions": None}
            classification_flags = get_product_classification(result["text"])
            report = analyze_ocr(normalized_ocr, **classification_flags)
            
            # --- UI: VISUAL PROOF (COLUMNS) ---
            st.divider()
            col_img1, col_img2 = st.columns(2)
            
            with col_img1:
                st.markdown("#### Original Image")
                st.image(Image.open(uploaded_file), use_container_width=True)
                
            with col_img2:
                st.markdown("#### AI Text Detection")
                st.image(ocr_vis_rgb, use_container_width=True)

            # --- UI: COMPLIANCE METRICS ---
            st.divider()
            st.subheader(f"Audit Outcome: {report['outcome']}")
            
            # Metric Cards
            met1, met2, met3 = st.columns(3)
            # --- UI: DPDP ACT PRIVACY SCANNER ---
            st.divider()
            st.subheader("🔐 DPDP Act Privacy Audit")
            
            # Pass the extracted text and the authorized company name to the scanner
            authorized_company = report.get('manufacturer', {}).get('name', '')
            privacy_report = scan_label_for_pii(result["text"], allowed_entities=[authorized_company])
            
            if privacy_report["status"] == "PASS":
                st.success("✅ No unauthorized PII or database leaks detected.")
            else:
                st.error(f"🚨 {privacy_report['total_leaks']} Data Privacy Violations Detected!")
                for violation in privacy_report["violations"]:
                    st.warning(f"- {violation}")
            met1.metric("Rules Passed ✅", report['counts']['pass'])
            met2.metric("Needs Review ⚠️", report['counts']['needs_review'])
            met3.metric("Failed ❌", report['counts']['fail'])
            
            # --- UI: ACTIONABLE FINDINGS ---
            if report["findings"]:
                st.markdown("### 🚨 Regulatory Violations Detected")
                for finding in report["findings"]:
                    if finding["outcome"] != "PASS":
                        # Use Streamlit's colored message boxes
                        if finding["outcome"] == "FAIL":
                            st.error(f"**{finding['rule_id']}: {finding['rule_name']}**\n\n*Issue:* {finding['message']}\n\n*Required Fix:* {finding['fix']}")
                        else:
                            st.warning(f"**{finding['rule_id']}: {finding['rule_name']}**\n\n*Issue:* {finding['message']}\n\n*Required Fix:* {finding['fix']}")
            
            # --- UI: RAW DATA EXPANDER ---
            with st.expander("🔍 View Raw OCR Extraction Data"):
                st.text(result["text"])