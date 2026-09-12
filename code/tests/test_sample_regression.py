from __future__ import annotations

import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.sample_regression import (  # noqa: E402
    _expected_values,
    _parse_plan,
)
from evaluation.sample_regression import FieldDiff, _field_category  # noqa: E402


class SampleRegressionTests(unittest.TestCase):
    def test_plan_comparison_parser_is_semantic(self):
        self.assertEqual(_parse_plan("2025-01-01:1.00|2025-02-01:2"), ((date(2025, 1, 1), Decimal("1.00")), (date(2025, 2, 1), Decimal("2"))))

    def test_expected_row_is_loaded_without_request_specific_logic(self):
        row = {"amount_safe_to_pay": "10.0", "affordability_status": "affordable_now", "recommended_payment_method": "full_payment", "payment_plan": "2025-01-01:10", "earliest_date_for_full_payment": "2025-01-01", "spending_changes_needed": "none"}
        values = _expected_values(row)
        self.assertEqual(values["amount_safe_to_pay"], Decimal("10.0"))
        self.assertEqual(values["payment_plan"][0][1], Decimal("10"))

    def test_categories(self):
        self.assertEqual(_field_category("amount_safe_to_pay", "1", "2"), "SAFE_AMOUNT")
        self.assertEqual(_field_category("earliest_date_for_full_payment", "x", "y"), "EARLIEST_DATE")
        self.assertEqual(_field_category("payment_plan", "x", "y"), "PLAN_RANKING")
        self.assertEqual(_field_category("recommended_payment_method", "installments", "wait"), "INSTALLMENT")
        self.assertEqual(_field_category("spending_changes_needed", "none", "stop:e1"), "SPENDING_CHANGE")


if __name__ == "__main__":
    unittest.main()
