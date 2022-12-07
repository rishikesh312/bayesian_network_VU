from typing import Union
from BayesNet import BayesNet
from copy import deepcopy
import itertools
import pandas as pd
import numpy as np
import networkx as nx
from pgmpy.readwrite import XMLBIFReader
from typing import List
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
            new_cpt = new_cpt.append(compats.loc[compats.index[0]])
        return new_cpt.reset_index(drop=True)

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

    def factor_multiply(self, f1: pd.DataFrame, f2: pd.DataFrame) -> pd.DataFrame:
        """
        Given two factors f1 and f2, compute the multiplied factor f=f1*f2. 
        """
        vars1 = f1.columns.tolist()
        vars1.remove("p")
        vars2 = f2.columns.tolist()
        vars2.remove("p")
        union = set(vars1).union(set(vars2))

        worlds_all = [list(i) for i in itertools.product([False, True], repeat=len(union))]
        new_factor = pd.DataFrame(columns=list(union)+['p'])

        for world in worlds_all:
            values = world
            ins = pd.Series(values, index=union)
            compats1 = self.bn.get_compatible_instantiations_table(ins, f1)
            compats2 = self.bn.get_compatible_instantiations_table(ins, f2)
            p = compats1.iloc[0].at['p'] * compats2.iloc[0].at['p']
            ins = pd.Series(values+[p], index=list(union)+['p'])
            new_factor = new_factor.append(ins, ignore_index=True)

        return new_factor

    def _in_factor(self, var: str, factor: pd.DataFrame) -> bool:
            columns = factor.columns.tolist()
            if var in set(columns[:columns.index('p')]):
                return True
            else:
                return False

    def _find_factor_index(self, f, factors: pd.DataFrame) -> int:
        for i, factor in enumerate(factors):
            if f.equals(factor):
                return i
        return False

    def _reduce_factors(self, ins: pd.Series, factors: List[pd.DataFrame]) -> List[pd.DataFrame]:
        new_factors = []
        for factor in factors:
            new_factors.append(self.bn.reduce_factor(ins, factor))
        return new_factors

    def _eliminate(self, var: str, factors: List[pd.DataFrame], eli_type: str) -> List[pd.DataFrame]:
        to_eli = deepcopy(factors)
        eli = False
        for f in to_eli:
            if self._in_factor(var, f):
                if not eli:
                    eli_factor = f
                else:
                    eli_factor = self.factor_multiply(eli_factor, f)
                factors.pop(self._find_factor_index(f, factors))
                eli = True
            else:
                continue
        if eli_type == "sum-out":
            result_factor = self.sum_out(var, eli_factor)
        elif eli_type == "max-out":
            result_factor = self.max_out(var, eli_factor)
        else:
            raise ImportError
        factors.append(result_factor)

        return factors

    def variable_eliminate(self, querys: list, evidence: pd.Series, factors: list):
        # reduce factors w.r.t evidence
        factors = self._reduce_factors(evidence, factors)
        # get sum-out (eliminate) order
        order_for_sum_out, _ = self.ordering(querys, heuristic="degree")
        # sum-out (eliminate) variables
        for var in order_for_sum_out:
            factors = self._eliminate(var, factors, eli_type="sum-out")

        return factors

    def map(self, query, evidence, factors):
        """
        Maximum A-posteriori Query
        Compute the maximum a-posteriory instantiation + value of query variables Q, given a possibly empty evidence e. 
        """
        variables = self.bn.get_all_variables()
        cpts = list(self.bn.get_all_cpts().values())
        eli_vars = list(set(variables)-set(query))
        
        order_for_sum_out, _ = self.ordering(eli_vars, heuristic="degree")

        factors = self.variable_eliminate(order_for_sum_out, evidence, cpts) 
        # get max-out (eliminate) order
        order_for_max_out, _ = self.ordering(query, heuristic="degree")
        # max-out (eliminate) variables
        for var in order_for_max_out:
            factors = self._eliminate(var, factors, eli_type="max-out")

        return factors













# -------------------------------------------------

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
    #print(probs)
    d=probs
    key=list(d.keys())
    indi_probs=[]
    answer=[]
    entry={}
    for k in key:
        indi_probs.append(probs[k])
    #print(indi_probs)
    total=sum(indi_probs)
    for k in key:
        answer=probs[k]/total
        entry[k]=answer
    return entry

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