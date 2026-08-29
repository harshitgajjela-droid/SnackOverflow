import unittest
from rule_engine import OUTCOME_FAIL, OUTCOME_PASS, OUTCOME_REVIEW, RuleEngine, parse_mrp, parse_quantity, valid_month_year


def complete_package(**overrides):
    package = {
        "package_scope": "retail_prepackaged",
        "is_food": False,
        "is_imported": False,
        "may_become_unfit_for_human_consumption": False,
        "dimensions_relevant": False,
        "manufacturer": {"name": "Example Foods Pvt Ltd", "address": "1 MG Road, Bengaluru, Karnataka 560001"},
        "manufacturer_is_packer": True,
        "generic_name": "Biscuits",
        "net_quantity": "500 g",
        "manufacture_pack_import_month_year": "08/2026",
        "mrp_declaration": "MRP Rs. 120.00 (inclusive of all taxes)",
        "consumer_care": {"name": "Example Care", "address": "1 MG Road, Bengaluru 560001", "phone": "1800-123-4567"},
        "declarations_legible_and_prominent": True,
    }
    package.update(overrides)
    return package


class RuleEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = RuleEngine()

    def test_parsers(self):
        self.assertEqual(parse_quantity("500g"), {"amount": "500", "unit": "g"})
        self.assertEqual(parse_quantity("1L"), {"amount": "1", "unit": "l"})
        self.assertIsNone(parse_quantity("500 grams plus"))
        self.assertTrue(valid_month_year("August 2026"))
        self.assertFalse(valid_month_year("18/2026"))
        self.assertIsNotNone(parse_mrp("MRP ₹120.50 (inclusive of all taxes)"))
        self.assertIsNone(parse_mrp("Rs 120"))

    def test_complete_package_passes(self):
        report = self.engine.analyze(complete_package())
        self.assertEqual(report["outcome"], OUTCOME_PASS)

    def test_imported_product_requires_origin(self):
        report = self.engine.analyze(complete_package(is_imported=True))
        self.assertEqual(report["outcome"], OUTCOME_FAIL)
        failed = [f["rule_id"] for f in report["findings"] if f["outcome"] == OUTCOME_FAIL]
        self.assertIn("LM-6-ORIGIN", failed)
        self.assertIn("LM-6-IMPORTER", failed)

    def test_visual_check_needs_review_not_pass(self):
        report = self.engine.analyze(complete_package(declarations_legible_and_prominent=False))
        self.assertEqual(report["outcome"], OUTCOME_REVIEW)

    def test_non_retail_scope_not_applicable(self):
        self.assertEqual(self.engine.analyze({"package_scope": "wholesale"})["outcome"], "NOT_APPLICABLE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
