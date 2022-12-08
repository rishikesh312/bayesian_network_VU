from BayesNet import BayesNet
from BNReasoner import BNReasoner
   
#Network Purning
br = BNReasoner("testing/dog_problem.BIFXML")
br.bn.draw_structure()#shows the graph before pruning
br.network_pruning("family-out",{"hear-bark":True})
br.bn.draw_structure()#shows the graph after pruning

#marginal distribution
br = BNReasoner("testing/dog_problem.BIFXML")
br.bn.draw_structure()
print(br.marginal_distribution(["dog-out"],{"family-out":True,"bowel-problem":False},["hear-bark"]))
