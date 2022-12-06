from typing import Union
from BayesNet import BayesNet
from copy import deepcopy
import itertools
import pandas as pd
import numpy as np
from pgmpy.readwrite import XMLBIFReader
class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    # TODO: This is where your methods should go
    # METHODS FOR INFERENCE ---------------------------------------------------------------------------------------

    def dsep(self, x: str, y: str, z: str) -> bool:
        """
        Given three sets of variables X, Y, and Z, determine whether X is d-separated of Y given Z.
        """
        bn = self.bn.structure 
        
        def _prune(): 
            _iterable = False
            nodes = deepcopy(bn.nodes)
            # Delete every leaf node(&edges) W not in (X, Y, Z)
            for node in nodes:
                if bn.out_degree(node) == 0 and node not in [x, y, z]:
                    bn.remove_node(node)
                    _iterable = True
            # Delete all edges outgoing from Z
            z_successors = [n for n in bn.successors(z)]
            if len(z_successors) > 0:
                _iterable = True
                for suc in z_successors:
                    bn.remove_edge(z, suc)
            return _iterable

        iterable = True
        while iterable:
            iterable = _prune()
        
        def _is_connected(bn, a, b):
            bn = bn.to_undirected()
            read = set()
            def __exist_path(a, b):
                neighbors = [n for n in bn.neighbors(a)]
                if bn.has_edge(a, b):
                    return True
                for n in neighbors:
                    if n not in read:
                        read.add(n)
                    else:
                        continue
                    connec = __exist_path(n, b)
                    if connec:
                        return True
                return False
            return __exist_path(a, b)

        return not _is_connected(bn, x, y)

    def independence(self, x:str, y:str, z:str) -> bool:
        """
        Given three sets of variables X, Y, and Z, determine whether X is independent of Y given Z.
        """
        if self.dsep(self,x,y,z) == True:
            return True
        else:
            return False

    def sum_out(self,x: str, cpt: pd.DataFrame):
        """
        Given a factor and a variable X, compute the CPT in which X is summed-out.
        """
        tags = cpt.columns.tolist()
        # remove x from tag and used as columns of new cpt
        tags.remove(x)
        entries ={}
        pr_tag = tags[-1]
        # get all variables
        vars = tags[:-1]
        worlds = [list(i) for i in itertools.product([False, True], repeat=len(vars))]
        # remove column x from cpt
        cpt = cpt[tags]
        new_cpt = pd.DataFrame(columns=vars)
        for world in worlds:
            ins = pd.Series(world, index=vars)
            compats = self.bn.get_compatible_instantiations_table(ins, cpt)
            # get sum-out pr and update to row 0
            compats = compats.append(compats.sum(), ignore_index=True)
            compats.loc[compats.index[0], pr_tag] = compats.loc[compats.index[-1], pr_tag]
            # append row 0 to new cpt
            new=compats.loc[compats.index[0]]
            new_cpt = new_cpt.append(new)
            all_probs=new['p'].tolist()
            key=tuple(world)
            entries[key]=[all_probs]
        return (vars,entries)
    

    def max_out(self, x: str, cpt: pd.DataFrame):
        """
        Given a factor and a variable X, compute the CPT in which X is max-out,
        also keep track of which instantiation of X led to the maximized value.
        """
        tags = cpt.columns.tolist()
        # remove x from tag, use for max-out operation
        tags.remove(x)
        pr_tag = tags[-1]
        # get all variables
        vars = tags[:-1]
        worlds = [list(i) for i in itertools.product([False, True], repeat=len(vars))]
        for world in worlds:
            ins = pd.Series(world, index=vars)
            compats = self.bn.get_compatible_instantiations_table(ins, cpt)
            # get min index and delete from original cpt, which is equivalent of max-out
            min_index = compats[compats[pr_tag] == compats[pr_tag].min()].index.tolist()[0]
            cpt.drop(min_index, inplace=True)
        # reset columns, move maximize variable to the end
        new_columns = vars + [pr_tag] + [x]
        cpt = cpt.reindex(columns=new_columns)
        return cpt.reset_index(drop=True)

    def ordering(self, vars: set, heuristic: str):
        """
        Given a set of variables X in the Bayesian network, compute a good ordering for the elimination of X
        based on the min-degree heuristics and the min-fill heuristics.
        """
        interaction_graph = self.bn.get_interaction_graph()
        order = list()

        def _eliminate(x, interaction_graph):
            # sum-out x in interaction_graph
            if x not in interaction_graph:
                raise Exception('Variable not in the interaction graph')
            neighbors = list(interaction_graph.neighbors(x))
            for i in range(len(neighbors)-1):
                for j in range(i+1, len(neighbors)):
                    if not interaction_graph.has_edge(neighbors[i], neighbors[j]):
                        interaction_graph.add_edge(neighbors[i], neighbors[j])
            interaction_graph.remove_node(x)

        def _get_least_degree_var(vars, interaction_graph):
            # get the var that its elimination cause least new connection adding
            degrees = list(interaction_graph.degree)
            for one in deepcopy(degrees):
                if one[0] not in vars:
                    degrees.remove(one)
            return max(degrees)[0]

        def _get_least_adding_var(vars, interaction_graph):
            # get the var that has minimum degree
            adding = dict()
            for var in vars:
                count = 0
                neighbors = list(interaction_graph.neighbors(var))
                for i in range(len(neighbors)-1):
                    for j in range(i+1, len(neighbors)):
                        if not interaction_graph.has_edge(neighbors[i], neighbors[j]):
                            count += 1
                adding[var] = count
            return min(adding, key=adding.get)

        def _iter_eliminate(vars, select_func, interaction_graph):
            # iteratively run variable elimination
            nodes = deepcopy(vars)
            while len(nodes) > 0:
                selected = select_func(nodes, interaction_graph)
                order.append(selected)
                _eliminate(selected, interaction_graph)
                nodes.remove(selected)

        if heuristic == 'degree':
            _iter_eliminate(vars, _get_least_degree_var, interaction_graph)
            
        if heuristic == 'fill':
            _iter_eliminate(vars, _get_least_adding_var, interaction_graph)
        
        return order, interaction_graph
    def edge_prune(self,e):
        bn = self.bn.structure
        e_connection =[conn for conn in bn.successors(e)]
        for con in e_connection:
            bn.remove_edge(e,con)
    def node_prune(self,Q,e):
        bn = self.bn.structure
        #nodes = deepcopy(bn.nodes)
        nodes = deepcopy(bn.nodes)
        for node in nodes:
            if bn.out_degree(node)==0 and node not in [Q,e]:
                bn.remove_node(node)
    def joint_distribution(self,cpt):
       names=cpt.columns.tolist()
       print(names)
       variable=names[:-1]
       newt = pd.DataFrame(columns=variable)
       worlds = [list(i) for i in itertools.product([False, True], repeat=len(variable))]
       for world in worlds:
           inst=pd.Series(world,index=variable)
           compart=self.bn.get_compatible_instantiations_table(inst, cpt)
           newt.append(compart)
           print(newt)
#------------------------------Variable Elimination---------------------------------------------------------------
    def multip(self,var,factor1,factor2):
        newvar=[]
        #print(var)
        newvar.extend(factor1[0])
        newvar.extend(factor2[0])
        newvar=list(set(newvar))
        newvar.sort()
        perms=self.genpermutations(len(newvar))
        newtbl={}
        asg={}
        for perm in perms:
            for pair in zip(newvar,perm):
                asg[pair[0]]=pair[1]
            key=tuple(asg[v] for v in newvar)
            key1=tuple(asg[v] for v in factor1[0])
            key2=tuple(asg[v] for v in factor2[0])
            prob = factor1[1][key1][0]*factor2[1][key2][0]
            newtbl[key] =prob
        return (newvar,newtbl)
    
    def querygiven(self,Q,e):
        #Given the probablity of p(Q|e)
        #it finds the probablity
        bn=self.bn.structure
        node=bn.nodes
        cpt = self.bn.get_cpt(Q)
        all_probs=cpt['p'].tolist()
        evidence=pd.Series(e)
        parent=[p for p in bn.predecessors(Q)]
        #if there is not parent
        if len(parent)==0:
            prob=[all_probs[1] if e[Q] else all_probs[0]]
        #if there is atleast one parent get the value of the parents, then query for p(Y)=y
        else:
            w=self.bn.get_compatible_instantiations_table(evidence, cpt)
            prob=w['p'].tolist()
        return prob
            
    def normalize(self,probs):
        return tuple(x*1/sum(probs) for x in probs)
    
    def genpermutations(self,length):
        #creates various combinations of True and False w.r.t to length
        perms=[list(i) for i in itertools.product([False, True], repeat=length)]    
        return perms

    def makefactor(self, var, factorvars, e):
        bn = self.bn.structure
        nodes = deepcopy(bn.nodes)
        parent_node=[]
        for parent in nodes:
            if bn.out_degree(parent)>0:
                parent_node.append(parent)
        variables = factorvars[var]
        variables.sort() 
        allvars = deepcopy(parent_node)
        allvars.append(var)
        perms = self.genpermutations(len(allvars))
        entries = {}
        asg = {}
        for perm in perms:
            violate = False
            for pair in zip(allvars, perm): # tuples of ('var', value)
                if pair[0] in e and e[pair[0]] != pair[1]:
                    violate = True
                    break
                asg[pair[0]] = pair[1]

            if violate:
                continue
            key = tuple(asg[v] for v in variables)
            prob = self.querygiven(var, asg)
            entries[key] = prob
        return (variables, entries)
    
    def variable_elimination(self,Q,e,cpt):
        bn = self.bn.structure
        nodes = deepcopy(bn.nodes)
        names=cpt.columns.tolist()
        title=names[:-1]
        eliminate = set()
        factor=[]
        leaf_node=[]
        parent_node=[]
        while len(eliminate)<len(title):
            #filters the eliminated variables
            variables= filter(lambda a: a not in eliminate,list(title))
            #print(variables)
            #calculates the leaf node
            for leaf in nodes:
                leaf_node.append(self.bn.get_children(leaf))
            #calculates the parent node
            for parent in nodes:
                if bn.out_degree(parent)>0:
                    parent_node.append(parent)
            #filters the variable that has some children that is not eliminated
            variables=filter(lambda v: (c in eliminate for c in leaf_node),variables)
            factor_variables={}
            #Enmerates the variables in factor associated with the variale
            for v in variables:
                factor_variables[v]=[p for p in parent_node if p not in e]
                if v not in e:
                    factor_variables[v].append(v)
            #sorts w.r.t number of variables and alphabetical order
            var=sorted(factor_variables.keys(),key=(lambda x: (len(factor_variables[x]),x)))[0]
            #Making factors
            if  len(factor_variables[var])>0:
                factor.append(self.makefactor(var, factor_variables, e))
            #if the selected var is not in the query or evidence then the factor is summed out  
            if var !=Q and var not in e:
                factor=[self.sum_out(var, cpt)]    
            #updating the eliminated value
            eliminate.add(var)    
        #calculating the product
        if len(factor[0]) >=2:
                for fact in factor[0:]:
                    result=self.multip(var,factor[0],fact)
        else:
            result=factor[0]
        i=True
        #normalizing it
        for t in result:
            solution=self.normalize((result[1][(False,i)],result[1][True,i]))
            i=False
        return solution
 #---------------------------------------------------------------------------------------------------           
        
        
bn = BayesNet()
bn.load_from_bifxml("testing/dog_problem.BIFXML")
br = BNReasoner(bn)
"""sum_out and max_out test"""
cpt = bn.get_cpt("dog-out")
x = 'family-out'
print(bn.get_all_cpts())
print("\n \n")
print(cpt)

#print(br.edge_prune("dog-out"))
#print(br.node_prune("family-out","dog-out"))
#print(bn.draw_structure())
#print(br.sum_out(x, cpt))
#print(br.joint_distribution(cpt))
#print(br.max_out(x, cpt))
print(br.variable_elimination("family-out",{"dog-out":False,"family-out":False}, cpt))
#con=[bn.get_cpt("family-out")]
#print(con)