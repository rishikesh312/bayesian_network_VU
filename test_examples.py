from BNReasoner import BNReasoner
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# These are examples and they are only readable, if you wanna test, test with your own test.py

def draw_graph(graph):
    nx.draw(graph, with_labels=True, node_size=3000)
    plt.show()


"""d-separation test"""
br = BNReasoner("testing/dog_problem.BIFXML")
br.bn.draw_structure()
x, y, z = 'bowel-problem', 'light-on', 'dog-out'
print(br.dsep(x, y, z))


"""ordering test"""
br = BNReasoner("testing/dog_problem.BIFXML")
vars = ['family-out', 'dog-out']
heuristic = 'degree'
order, interaction_graph = br.ordering(vars, heuristic='fill')
print(order, interaction_graph)
draw_graph(interaction_graph)


"""sum_out and max_out test"""
br = BNReasoner("testing/dog_problem.BIFXML")
cpt = br.bn.get_cpt("dog-out")
x = 'family-out'
print(cpt)
print(br.sum_out(x, cpt))
print(br.max_out(x, cpt))


"""test for multiplication"""
br = BNReasoner("testing/dog_problem.BIFXML")
cpt1 = br.bn.get_cpt("dog-out")
cpt2 = br.bn.get_cpt("light-on")
print(cpt1)
print(cpt2)
print(br.factor_multiply(cpt1, cpt2))


"""test for variable elimination"""
br = BNReasoner("testing/abc.BIFXML")
br.bn.draw_structure()
factors = list(br.bn.get_all_cpts().values())
for f in factors:
    print(f)
ins = pd.Series({"A": True})
results = (br.variable_eliminate(["A", "B"], ins, factors))
for f in results:
    print(f)


"""test for map"""
br = BNReasoner("testing/map_mpe.BIFXML")
br.bn.draw_structure()
factors = list(br.bn.get_all_cpts().values())
for f in factors:
    print(f)
ins = pd.Series({"O": True})
results = (br.map(["I", "J"], ins, factors))
for f in results:
    print(f)
   
""" Network Purning"""
br = BNReasoner("testing/dog_problem.BIFXML")
br.bn.draw_structure()
br.network_pruning("family-out",{"hear-bark":True})
br.bn.draw_structure()

""" marginal distribution"""
br = BNReasoner("testing/dog_problem.BIFXML")
br.bn.draw_structure()
print(br.marginal_distribution(["dog-out"],{"family-out":True,"bowel-problem":False},["hear-bark"]))
