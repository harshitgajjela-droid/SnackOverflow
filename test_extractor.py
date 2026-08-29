import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extractor import extract_information
from pipeline import analyze_ocr


class ExtractorTests(unittest.TestCase):
    def setUp(self):
        self.ocr = {"generic_name": "Biscuits", "lines": [
            {"text": "Manufactured by: ABC Foods Pvt Ltd", "bbox": [0, 0, 250, 20], "confidence": .99, "page": "back"},
            {"text": "1 MG Road, Bengaluru 560001", "bbox": [0, 22, 250, 42], "confidence": .99, "page": "back"},
            {"text": "Net Qty: 500g", "bbox": [0, 44, 150, 64], "confidence": .99, "page": "back"},
            {"text": "Mfd: 08/2026", "bbox": [0, 66, 150, 86], "confidence": .99, "page": "back"},
            {"text": "MRP Rs. 120.00", "bbox": [0, 88, 180, 108], "confidence": .99, "page": "back"},
            {"text": "(inclusive of all taxes)", "bbox": [0, 110, 180, 130], "confidence": .99, "page": "back"},
            {"text": "Consumer Care: ABC Care, 1 MG Road Bengaluru 560001", "bbox": [0, 132, 350, 152], "confidence": .99, "page": "back"},
            {"text": "1800-123-4567", "bbox": [0, 154, 150, 174], "confidence": .99, "page": "back"},
        ]}

    def test_extracts_multiline_mrp_and_quantity(self):
        result = extract_information(self.ocr)
        self.assertEqual(result["net_quantity"]["value"], "500 g")
        self.assertEqual(result["mrp_declaration"]["value"], "MRP Rs. 120.00 (inclusive of all taxes)")
        self.assertEqual(len(result["mrp_declaration"]["evidence"]), 2)

    def test_pipeline_runs_from_ocr_lines(self):
        report = analyze_ocr(
            self.ocr, package_scope="retail_prepackaged", is_food=False,
            is_imported=False, manufacturer_is_packer=True,
            may_become_unfit_for_human_consumption=False, dimensions_relevant=False,
            declarations_legible_and_prominent=True,
        )
        self.assertEqual(report["outcome"], "PASS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
