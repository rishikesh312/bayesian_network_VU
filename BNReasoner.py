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
        self.backup_bn = deepcopy(self.bn)
    
    def resume_network(self) -> None:
        """
        resume the newwork to its initial structure
        """
        self.bn = deepcopy(self.backup_bn)

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
            return nx.has_path(self.bn.structure, a, b)

        return not _is_connected(bn, x, y)

    def independence(self, x:str, y:str, z:str) -> bool:
        """
        Given three sets of variables X, Y, and Z, determine whether X is independent of Y given Z.
        """
        return self.dsep(x,y,z)

    def sum_out(self, x: str, cpt: pd.DataFrame) -> pd.DataFrame:
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
            compats = compats.append(compats.sum(numeric_only=True), ignore_index=True)
            compats.loc[compats.index[0], pr_tag] = compats.loc[compats.index[-1], pr_tag]
            # append row 0 to new cpt
            new_cpt = new_cpt.append(compats.loc[compats.index[0]])
        return new_cpt.reset_index(drop=True)

    def max_out(self, x: str, cpt: pd.DataFrame) -> pd.DataFrame:
        """
        Given a factor and a variable X, compute the CPT in which X is max-out,
        also keep track of which instantiation of X led to the maximized value.
        """
        tags = cpt.columns.tolist()
        # remove x from tag, use for max-out operation 
        tags.remove(x)
        # get tail tags
        tail = tags[tags.index('p'):]
        vars = tags[:tags.index('p')]
        # if vars = [], directly take max dataframe
        if vars == []:
            cpt = cpt[cpt['p'] == cpt['p'].max()]
        else:
            worlds = [list(i) for i in itertools.product([False, True], repeat=len(vars))]
            for world in worlds:
                ins = pd.Series(world, index=vars)
                compats = self.bn.get_compatible_instantiations_table(ins, cpt)
                # if lines of compats < 2, min() is meaningless
                if compats.shape[0]<2:
                    continue
                # get min index and delete from original cpt, which is equivalent of max-out
                min_index = compats[compats['p'] == compats['p'].min()].index.tolist()[0]
                cpt.drop(min_index, inplace=True)
        # reset columns, move maximize variable to the end
        new_columns = vars + tail + [x]
        cpt = cpt.reindex(columns=new_columns)
        return cpt.reset_index(drop=True)

    def ordering(self, vars: set, heuristic: str) -> pd.DataFrame:
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
            if degrees == []:
                return None
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

        def _iter_eliminate(vars, order, select_func, interaction_graph):
            # iteratively run variable elimination
            nodes = deepcopy(vars)
            while len(nodes) > 0:
                selected = select_func(nodes, interaction_graph)
                # if no more var can be selected, break the loop
                if not selected:
                    order += nodes
                    break
                order.append(selected)
                _eliminate(selected, interaction_graph)
                nodes.remove(selected)
            return order

        if heuristic == 'degree':
            order = _iter_eliminate(vars, order, _get_least_degree_var, interaction_graph)
            
        if heuristic == 'fill':
            order = _iter_eliminate(vars, order, _get_least_adding_var, interaction_graph)
        
        return order, interaction_graph

    def factor_multiplication(f: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
        """
        Smart factor mulplication.
        """ 
        vars_f = f.columns.tolist()
        vars_g = g.columns.tolist()

        join_var = [var for var in vars_f if var in vars_g and var != 'p']

        merged_cpt = pd.merge(f, g, left_on=join_var, right_on=join_var)
        merged_cpt['p'] = merged_cpt['p_x'] * merged_cpt['p_y']

        h = merged_cpt.drop(['p_x','p_y'], axis=1)

        return h

    def factor_multiply(self, f1: pd.DataFrame, f2: pd.DataFrame) -> pd.DataFrame:
        """
        Given two factors f1 and f2, compute the multiplied factor f=f1*f2. 
        """
        # for max-out factor, information after 'p' should also be tracked.
        # so we need to record the tail and its value
        columns1 = f1.columns.tolist()
        tail1 = columns1[columns1.index('p'):]
        columns2 = f2.columns.tolist()
        tail2 = columns2[columns2.index('p'):]
        # record tail (tacked variables)
        tail = set(tail1).union(set(tail2))

        join_var = list(set(columns1[:columns1.index('p'):]).intersection(set(columns2[:columns2.index('p'):])))

        merged_factor = pd.merge(f1, f2, left_on=join_var, right_on=join_var)
        merged_factor['p'] = merged_factor['p_x'] * merged_factor['p_y']
        merged_factor = merged_factor.drop(['p_x','p_y'], axis=1)
        # move fail variable to be the end of columns

        columns = list(set(merged_factor.columns.to_list()).difference(tail)) + ['p']
        tail.remove('p')
        columns += list(tail)
 
        merged_factor = merged_factor[columns]

        return merged_factor

    def _in_factor(self, var: str, factor: pd.DataFrame) -> bool:
            columns = factor.columns.tolist()
            if var in set(columns[:columns.index('p')]):
                return True
            else:
                return False

    def _find_factor_index(self, f: pd.DataFrame, factors: pd.DataFrame) -> int:
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

    def _merge_factors(self, factors: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge independent factors
        """
        p = 1
        new_data = {}
        for factor in factors:
            p *= factor.iloc[0].at['p']
            line = factor.iloc[-1].to_dict()
            del line['p']
            new_data ={**new_data, **line}
        result = {**{'p': p}, **new_data}
        return [pd.DataFrame([result.values()], columns=result.keys())]

    def variable_eliminate(self, queries: list, evidences: dict, factors: list) -> List[pd.DataFrame]:
        """
        Given queries, evidences, and factors, eliminate V/queries and return reduced factors
        """
        evidences = pd.Series(evidences)
        # reduce factors w.r.t evidence
        factors = self._reduce_factors(evidences, factors)
        # get sum-out (eliminate) order
        order_for_sum_out, _ = self.ordering(queries, heuristic="degree")
        # sum-out (eliminate) variables
        for var in order_for_sum_out:
            factors = self._eliminate(var, factors, eli_type="sum-out")

        return factors

    def map(self, queries: List[str], evidences: dict, prune=False, heuristic="degree") -> pd.DataFrame:
        """
        Maximum A-posteriori Query
        Compute the maximum a-posteriory instantiation + value of query variables Q, given a possibly empty evidence e. 
        """
        if prune:
            self.network_pruning(queries, evidences)
            # self.bn.draw_structure()
            
        variables = self.bn.get_all_variables()
        cpts = list(self.bn.get_all_cpts().values())
        eli_vars = list(set(variables)-set(queries))
 
        order_for_sum_out, _ = self.ordering(eli_vars, heuristic)

        factors = self.variable_eliminate(order_for_sum_out, evidences, cpts) 
        # get max-out (eliminate) order
        order_for_max_out, _ = self.ordering(queries, heuristic)
        # max-out (eliminate) variables
        for var in order_for_max_out:
            factors = self._eliminate(var, factors, eli_type="max-out")

        if len(factors) > 1:
            return self._merge_factors(factors)

        return factors

    def mpe(self, evidences: dict, prune=False, heuristic="degree") -> pd.DataFrame:
        if prune:
            self.network_pruning([],evidences)

        variables = self.bn.get_all_variables()
        cpts = list(self.bn.get_all_cpts().values())

        factors = self._reduce_factors(pd.Series(evidences), cpts)

        # get max-out (eliminate) order
        order_for_max_out, _ = self.ordering(variables, heuristic)
        # max-out (eliminate) variables
        for var in order_for_max_out:
            factors = self._eliminate(var, factors, eli_type="max-out")

        if len(factors) > 1:
            return self._merge_factors(factors)

        return factors

    def marginal_distribution2(self, queries: list, evidence: dict = None) -> pd.DataFrame:
        """
        Given query variables Q and possibly empty evidence e, compute the marginal distribution P(Q|e).
        """
        evidence = {} if not evidence else evidence

        variables = self.bn.get_all_variables()
        cpts = list(self.bn.get_all_cpts().values())
        # sum-out all variables expect queries, to joint distribution
        eli_vars = list(set(variables)-set(queries))

        self.network_pruning(queries, evidence)
        
        order_for_sum_out, _ = self.ordering(eli_vars, heuristic="degree")
        joint_distribution = self.variable_eliminate(order_for_sum_out, evidence, cpts)[0]

        if not evidence:
            return joint_distribution

        # sum-out all variables expect evidence, to get marginalization
        eli_vars = list(set(variables)-set(evidence))
        e_key = list(evidence.keys())[0]
        # if evidence is in root node, directly return its reduce factor, otherwise calulate p(e) using variable elimination
        if self.bn.structure.in_degree(e_key)==0:
            marginalization = self.bn.reduce_factor(pd.Series(evidence), self.bn.get_cpt(e_key))
        else:
            order_for_sum_out, _ = self.ordering(eli_vars, heuristic="degree")
            marginalization = self.variable_eliminate(order_for_sum_out, evidence, cpts)[0]
        # divide joint_distribution by marginalization to get posterior
        marginalization = marginalization[marginalization['p']!=0]
        pr_e = marginalization.iloc[0].at['p']
        joint_distribution['p'] = joint_distribution['p'] / pr_e
     
        return joint_distribution

    def network_pruning(self, Q: list, e: dict) -> None:
        """
        Network Purning
        Input:
            Q ->Query (Input type: list)
            e -> evidence (Input type: dict)
        Output:
            Doesnt return a value, but it does edge and leaf node pruning 
        """
        #edge pruning
        for var,truth_val in zip(e.keys(),e.values()):
            cpt = self.bn.get_cpt(var)
            update_cpt =self.bn.get_compatible_instantiations_table(pd.Series({var:truth_val}),cpt)
            self.bn.update_cpt(var, update_cpt)
            #checking if node in evidence has children (children of the pruned edge) 
            if self.bn.get_children(var)==[]:
                pass
            else:
                for child in self.bn.get_children(var):
                    #remove the edge bwt deleted node and child
                    self.bn.del_edge((var,child))
                    #update the cpt
                    cpt = self.bn.get_cpt(child)
                    cpt_update = self.bn.get_compatible_instantiations_table(pd.Series({var:truth_val}), cpt)
                    #self.bn.update_cpt(child, cpt_update)
            #Leaf node pruning
            exit_loop = False
            while not exit_loop:
                exit_loop = True
                for variable in self.bn.get_all_variables():
                    #check if leaf node affects the Q or e (meaning: having no child)
                    if self.bn.get_children(variable)==[]:
                        #leaf node not in Q or e
                        if variable not in set(Q) and variable not in set(e.keys()):
                            #removing leaf node and running again to check if there is any leaf node left
                            self.bn.del_var(variable)
                            exit_loop=False

    def multiply_fact(self,X):
        """
        multiple_factor(only for mariginal distribution purposes)
        Input:
            X ->list of CPT to multiply (Input type: list)
        Output:
            product of factors
        """
        factor = X[0]
        for index in range(1, len(X)):
            x = X[index]
            column_x = [col for col in x.columns if col != 'p']
            column_factor = [col for col in factor.columns if col != 'p']
            match = list(set(column_x) & set(column_factor))
            
            if len(match) != 0:
                df_mul = pd.merge(x, factor, how='left', on=match)
                df_mul['p'] = (df_mul['p_x'] * df_mul['p_y'])
                df_mul.drop(['p_x', 'p_y'],inplace=True, axis = 1)
                factor = df_mul
        return factor

    def marginal_distribution(self,Q,e):
        """
        Marginal Distribution
        Input:
            Q -> Query (Input type: List)
            e -> evidence (Input type: Dict)
        Return:
            Marginal Distribution w.r.t (Q,e)
        """

        #prunes the network based on Q and e
        self.network_pruning(Q, e)
        evidence_fact=1
        #get the probability of evidence
        for variable in e:
            evidence_fact *=self.bn.get_cpt(variable)['p'].sum()
        #get all cpts where the variables occur
        src=self.bn.get_all_cpts()
        factor = 0
        var=[]
        for q in Q:
            var = [p for p in self.bn.structure.predecessors(q)]
        for variable in var:
            factor_var ={}
            
            for cpt_var in src:
                if variable in src[cpt_var]:
                    factor_var[cpt_var]=src[cpt_var]
            if len(factor_var) >= 2:
                 multiplied_cpt = self.multiply_fact(list(factor_var.values()))
                 
                 new_cpt = self.sum_out(variable,multiplied_cpt)
                 for factor_variable in factor_var:
                     del src[factor_variable]
                 
                 factor +=1
                 src["factor "+str(factor)] = new_cpt
             # when there is only one cpt, don't multiply
            elif len(factor_var)==1:
               df = pd.DataFrame(list(factor_var.values())[0])
               new_cpt =self.sum_out(variable,df)    
               for factor_variable in factor_var:
                     del src[factor_variable]
               factor +=1
               src["factor "+str(factor)] = new_cpt


        if len(src)>1:
           marginal_distrib=self.multiply_fact(list(src.values()))
        else:
            marginal_distrib=list(src.values())[0]
        marginal_distrib=marginal_distrib.dropna()
        marginal_distrib["p"] = marginal_distrib["p"].div(evidence_fact)
        
        return marginal_distrib
