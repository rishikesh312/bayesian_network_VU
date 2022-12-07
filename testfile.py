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
#print(br.node_pruning("dog-out,family-out"))
#print(br.edge_prune("dog-out"))
#print(br.node_prune("family-out","dog-out"))
#print(bn.draw_structure())
#print(br.sum_out(x, cpt))
#print(br.joint_distribution(cpt))
#print(br.max_out(x, cpt))
print(br.variable_elimination("dog-out",{"family-out":True,"bowel-problem":False}, cpt))
#con=[bn.get_cpt("family-out")]
#print(con)
