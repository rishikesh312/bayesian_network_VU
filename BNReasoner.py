from typing import Union
from BayesNet import BayesNet


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
        bn = deepcopy(self.structure) 
        
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
            compats = self.get_compatible_instantiations_table(ins, cpt)
            # get sum-out pr and update to row 0
            compats = compats.append(compats.sum(), ignore_index=True)
            compats.loc[compats.index[0], pr_tag] = compats.loc[compats.index[-1], pr_tag]
            # append row 0 to new cpt
            new_cpt = new_cpt.append(compats.loc[compats.index[0]])
        return new_cpt.reset_index(drop=True)

    def ordering(self, vars: set, heuristic: str):
        """
        Given a set of variables X in the Bayesian network, compute a good ordering for the elimination of X
        based on the min-degree heuristics and the min-fill heuristics.
        """
        interaction_graph = self.get_interaction_graph()
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