#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:27:06 2022

@author: rishi
"""

from BayesNet import BayesNet
from BNReasoner import BNReasoner


bn = BayesNet()
bn.load_from_bifxml("testing/dog_problem.BIFXML")
br = BNReasoner(bn)
cpt = bn.get_cpt("dog-out")
x = 'family-out'
print(bn.get_all_cpts())
print("\n \n")
print(cpt)
print(br.variable_elimination("dog-out",{"family-out":True,"bowel-problem":False}, cpt))
