import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from decide import expected_value_of_offer
class Economics(unittest.TestCase):
    def test_certain_stayer_only_incurs_discount_and_contact(self):
        self.assertAlmostEqual(expected_value_of_offer(0,2000,150),-24)
    def test_no_treatment_response_never_creates_retained_margin(self):
        self.assertAlmostEqual(expected_value_of_offer(1,2000,150,s=0),-1.5)
    def test_certain_rescue_uses_margin_less_discount_and_contact(self):
        self.assertAlmostEqual(expected_value_of_offer(1,2000,150,s=1),776)
    def test_zero_value_and_no_discount_leaves_contact_cost(self):
        self.assertAlmostEqual(expected_value_of_offer(.6,0,0),-1.5)
    def test_offer_design_can_change_expected_sign(self):
        self.assertGreater(expected_value_of_offer(.6,2000,150),0)
        self.assertLess(expected_value_of_offer(.6,2000,150,scope='annual'),0)
if __name__=='__main__':unittest.main()
