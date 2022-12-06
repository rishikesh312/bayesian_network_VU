from typing import Union
from BayesNet import BayesNet
from copy import deepcopy
import itertools
import pandas as pd


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

    def map(self, query, evidence, factors):
        """
        Maximum A-posteriori Query
        Compute the maximum a-posteriory instantiation + value of query variables Q, given a possibly empty evidence e. 
        """
        variables = self.bn.get_all_variables()
        cpts = self.bn.get_all_cpts()
        eli_vars = list(set(variables)-set(query))
        
        order_for_sum_out, _ = self.ordering(eli_vars, heuristic="degree")
        # TODO variable_eliminate(vars: list[str], evidences: list[str], cpts: list[df.Dataframe] )
        factors = self.variable_eliminate(order_for_sum_out, evidence, cpts) 

        def in_factor(var, factor):
            columns = factor.columns.tolist()
            if var in set(columns[:columns.index('p')]):
                return True
            else:
                return False

        def max_out_eliminate(var, factors):
            to_multiply = factors
            mul = False
            for f in to_multiply:
                if in_factor(var, f):
                    if not mul:
                        mul_factor = f
                    else:
                         # TODO factor_multiply(f1: pd.Datafame, f2: pd.Datafame) -> pd.Datafame:
                        mul_factor = self.factor_multiply(mul_factor, f)
                    factors.remove(f)
                    mul = True
                else:
                    continue
            
            max_out_factor = self.max_out(mul_factor)
            factors.append(max_out_factor)
            return factors

        order_for_max_out, _ = self.ordering(query, heuristic="degree")
        for var in order_for_max_out:
            factors = max_out_eliminate(var, factors)

        return factors

