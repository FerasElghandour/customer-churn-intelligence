"""Offline example using the public implementation; see README for evidence scope."""
import sys,json,pathlib,os
repo=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'src'));os.chdir(repo)
from decide import expected_value_of_offer, breakeven_probability
print(json.dumps({"fixture":"synthetic customer; assumed treatment response","next_order_ev":float(expected_value_of_offer(.6,2000,150)),"annual_discount_ev":float(expected_value_of_offer(.6,2000,150,scope="annual")),"breakeven_next":float(breakeven_probability(2000,150))}))
