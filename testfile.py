from BayesNet import BayesNet
from BNReasoner import BNReasoner
   
# #Network Purning
# br = BNReasoner("testing/dog_problem.BIFXML")
# br.bn.draw_structure()#shows the graph before pruning
# br.network_pruning("family-out",{"hear-bark":True})
# br.bn.draw_structure()#shows the graph after pruning



# #prior marginal distribution
# br = BNReasoner("testing/Usecase(xml).BIFXML")
# # br.bn.draw_structure()
# print(br.marginal_distribution2(["Dying?(DIE)","Steroid?(ST)"],{}))



# #posterior marginal distribution
# br = BNReasoner("testing/Usecase(xml).BIFXML")
# # br.bn.draw_structure()
# print(br.marginal_distribution2(["Dying?(DIE)","Steroid?(ST)"],{"Old_Age?(OA)":True}))


# #MAP
# br = BNReasoner("testing/Usecase(xml).BIFXML")
# # br.bn.draw_structure()
# # for f in factors:
# #     print(f)
# ins = {"Diabetes?(D)": True}
# result = (br.map(["Dying?(DIE)","Severe_Covid?(SC)"], ins, prune=True))
# print(result)

# #MPE
# br = BNReasoner("testing/Usecase(xml).BIFXML")
# # br.bn.draw_structure()
# factors = list(br.bn.get_all_cpts().values())
# # for f in factors:
# #     print(f)
# envidence = {"Insomnia?(I)": True, "Dying?(DIE)": True}
# result = (br.mpe(envidence, prune=False))
# print(result)  

