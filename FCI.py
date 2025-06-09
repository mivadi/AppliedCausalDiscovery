from causalGraph import CausalGraph
from FCIFunctions import deleteSeparatedEdges, addColliders, deleteSeparatedEdgesStage2, zhangsInferenceRules, completeUndirectedEdges, completeDirectedEdges, jci, background, arbiterFCI, adjmatrix2CausalGraph2
from independenceTest import IndependenceTest
from collections import defaultdict
import copy
import pandas as pd
from copy import deepcopy
import numpy as np
from searchFunctions import findAncestors


def FCI(variables, data=None, corr=None, imp=None, data_size=-1, alpha=0.05, indTest='Fisher', var_types=None, binary=None, conservative_FCI=False, 
    selection_bias=False, oracle_graph=None, oracle_SS=None, JCI='0', context=[],
    bkg_info=None, tiers=None, aug_PC=False, skeleton=False, complete_value_indicator=None, verbose=False, smart_way_to_delete_edges=True):
    """
    Returns a causalGraph object that represents a (over/undercomplete) PAG.
        :param variables: list of the non-context variables of the system (strings)
        :param data: dictionary (keys:variables/context) of np-arrays
        :param corr: correlation matrix
        :param imp: object gained from mice in R containing multiple imputations
        :param data_size: integer representing the number of data points, default is -1
        :param alpha: threshold for independence for m-separation, default is 0.05
        :param indTest: string describing independence test, default is 'Fisher'; 'oracle'
                         and 'kernel' is other option
        :param var_types: R list indicating the type of variables ('cont', 'ord', 'cat')
        :param binary: R list indicating with 0 and 1 if the type of variables are binary
        :param conservative_FCI: boolean indicating if we run conservative FCI
        :param selection_bias: boolean indicating if we run selection bias, only possible 
                        if JCI='0' 
        :param oracle_graph: CausalGgraog object
        :param oracle_SS: dictionary (keys:variables) of dictionary (keys:variables) for
                        separating sets -> {v1:{v2:[ss1, ss2]}} means v1 and v2 are
                        separated by set ss1 and separated by set ss2 (default is None)
        :param JCI: string ('0', '1', '12', '123', default is '0') indicating which  
                        assumtions from 'Joint Causal Inference from Multiple Contexts'
                        by Joris Mooij et al. (2020) are assumed to be true
        :param context: list of the context variables of the system (strings)
        :param bkg_info: background information in the form of an ancestral graph 
                        represented by a matrix where [i,j]=1 if i is an ancestor of j, 
                        [i,j]=1 if i is a non-descendants of j, and 0 means no info
        :param tiers: tiered background information (lists that contains the 
                        tiers in the order [tier 1, tier 2, etc.])
        :param aug_PC: boolean indicating if we run augmented PC instead of FCI 
                        (skip D in FCI from Causation Prediction and Search (p.145))
        :param skeleton: boolean indicating if only the skeleton should be returned
        :param missing_values_indicator: indicates with 1 if value is not missing 
                        (and is 0 when value is missing)
        :param verbose: boolean whether to print information during the run
    """

    ancestors = None

    if oracle_SS is None and oracle_graph is None:
        if data is None and imp is None and not corr is None and indTest!='Fisher':
            indTest = 'Fisher'
            print('Independence test is set to Fisher-Z test since data set was not avaiable.')
        elif data is None and not imp is None and corr is None and (indTest!='mixMI' and indTest!='mixmisCI' and indTest!='misReg'):
            indTest = 'mixMI'
            print('Independence test is set to mixMI test since data set was not avaiable.')
        elif data is None and not imp is None and not corr is None and indTest=='kernel':
            raise ValueError('Provide either a data set, correlation matrix or oracle_SS information.')
        elif data is None and corr is None and imp is None:
            raise ValueError('Provide either a data set, correlation matrix or oracle_SS information.')
        elif not data is None and corr is None and indTest=='Fisher':
            data_ = {}
            for var in variables:
                data_[var] = list(data[var][:,0])
            for var in context:
                data_[var] = list(data[var][:,0])
            data_ = pd.DataFrame.from_dict(data_)
            corr = data_.corr(method='spearman') # monotonic relationship
            data_size = data[[*data][0]].shape[0]
            # corr.to_csv('../../corr.csv', header=False, index=False)
        elif imp is None and (indTest=='mixMI' or indTest=='mixmisCI'):
            raise ValueError('Provide the multiple imputations.')
    elif not oracle_graph is None:
        ancestors = findAncestors(oracle_graph)

    # define independence test object
    IT = IndependenceTest(alpha=alpha, indTest=indTest, oracle_graph=oracle_graph, ancestors=ancestors, oracle=oracle_SS, var_types=var_types, binary=binary, complete_value_indicator=complete_value_indicator)

    # check if selection bias assumption is in line with background info assumptions
    if selection_bias:
        if len(context)>0 and JCI!='0':
            selection_bias = False
            print('Selection bias is set to false due to conflicting parameters.')

    if not tiers is None:
        for X in variables:
            for Y in context:
                if tiers[Y] >= tiers[X]:
                    raise ValueError('The tier of a context variable is not before the tier of a system variable.')
        if not bkg_info is None:
            for var1 in variables:
                for var2 in variables:
                    if tiers[var1] < tiers[var2] and bkg_info[var1][var2]!=-1:
                        raise ValueError('The background information is not in line with tiers.')
            for var1 in context:
                for var2 in variables:
                    if tiers[var1] < tiers[var2] and bkg_info[var2][var1]!=1:
                        raise ValueError('The background information is not in line with tiers.')
        else:
            bkg_info=pd.DataFrame(np.zeros(shape=(len(tiers),len(tiers))),columns=tiers.keys(), index=tiers.keys())
            for var1 in variables:
                for var2 in variables:
                    if tiers[var1] < tiers[var2]:
                        bkg_info[var1][var2]=-1
            for var1 in context:
                for var2 in variables:
                    if tiers[var1] < tiers[var2]:
                        bkg_info[var2][var1]=1

    # check if background information is consistent with context
    if not bkg_info is None and len(context)>0:
        for X in context:
            if -1 in bkg_info.loc[X, :].unique() or 1 in bkg_info.loc[:, X].unique():
                raise ValueError('The background information is not in line with the context variables.')

    # if JCI is not used: add context variables to other variables
    if JCI == '0':
        if len(context) > 0:
            variables = variables + context
            context = []

    if verbose: print('initialize causal graph')

    # define a complete undirected graph
    pag = CausalGraph(variables, data=data, corr=corr, imp=imp, data_size=data_size, unknown='u', default_edge='u', context=context)
    # separating_sets = defaultdict(list)
    separating_sets = defaultdict(lambda:None)
    uncertain_triples = defaultdict(list)

    if verbose: print('delete separated edges')

    # delete edges between variables which are d-separated based on sets in oracle
    deleteSeparatedEdges(pag, separating_sets, JCI=JCI, IT=IT, oracle=oracle_SS, tiers=tiers, smart_way_to_delete_edges=smart_way_to_delete_edges, verbose=verbose)

    if verbose: print('orient context variables')

    # orient edges according to JCI algorithm
    jci(pag, JCI=JCI)


    if verbose: print('orient background information')

    # add background information based on ancestral graph
    background(pag, bkg_info=bkg_info)

    if not aug_PC and oracle_SS is None and smart_way_to_delete_edges:

        if verbose: print('delete separated edges: stage 2')

        # add colliders: R0
        pag_copy = copy.deepcopy(pag)
        addColliders(pag_copy, separating_sets, IT=IT, uncertain_triples=uncertain_triples, bkg_info=bkg_info, tiers=tiers, verbose=verbose)

        # search for separated vertices and delete edges
        deleteSeparatedEdgesStage2(pag, pag_copy, separating_sets, IT, tiers=tiers, verbose=verbose)

        # reset uncertain triples
        uncertain_triples = defaultdict(list)

    if skeleton:

        if verbose: print("Return skeleton")

        return pag

    if verbose: print('orient colliders')

    # add colliders: R0
    addColliders(pag, separating_sets, conservative=conservative_FCI, IT=IT, uncertain_triples=uncertain_triples, bkg_info=bkg_info, tiers=tiers, verbose=verbose)

    if verbose: print('run rule R1-R4')

    # Zhangs version of Meeks orientation rules: R1-R4
    zhangsInferenceRules(pag, separating_sets, uncertain_triples=uncertain_triples, IT=IT, conservative=conservative_FCI, tiers=tiers, verbose=verbose)

    # additional orientation rules if there is selection bias
    if selection_bias:

        if verbose: print('run rule R5-R7')

        completeUndirectedEdges(pag, verbose=verbose)

    if verbose: print('run rule R8-R10')

    # complete directed edges: R8-R10
    completeDirectedEdges(pag, verbose=verbose)


    return pag




