from itertools import combinations, permutations, product, chain
from collections import defaultdict
from causalGraph import CausalGraph
from independenceTest import IndependenceTest

from searchFunctions import *
from KCIT import *
from boolFunctions import isSeparated
from copy import deepcopy
from selectPossibleDeconfoundingSets import PAG2MAG


def deleteSeparatedEdges(graph, separating_sets, testable_separating_sets=None, JCI='0', IT=None, oracle=None, tiers=None, smart_way_to_delete_edges=True, track_deletions=False, verbose=False):
    """
    Learning the skeleton: first phase.
        :param graph: CausalGraph
        :param separating_sets: dictionary of sets for each pair of separated vertices
        :param testable_separating_sets: dictionary of testable separating sets for each pair of separated vertices
        :param JCI: string ('0', '1', '12', '123', default is '0') indicating which  
                        assumtions from 'Joint Causal Inference from Multiple Contexts'
                        by Joris Mooij et al. (2020) are assumed to be true
        :param IT: IndependenceTest object
        :param oracle: true separating_sets
    """

    if not isinstance(IT, IndependenceTest) and oracle is None:
        raise ValueError('IndependenceTest or oracle required.')

    if track_deletions:
        rule = 'phase1'
    else:
        rule = None

    if oracle is None:

        if smart_way_to_delete_edges:

            depth = 0
            done = False

            while not done:

                if verbose: print('Depth is:', depth)

                # initilize done again
                done = True
                edges_to_delete = []

                for X in graph.all_variables:

                    # find the adjacency variables of X
                    # with stable version: https://www.jmlr.org/papers/volume15/colombo14a/colombo14a.pdf
                    adjacency_variables = graph.adjacencies(X)

                    # find the amount of adjacency variables of X
                    num_adj = len(adjacency_variables)

                    # if there are less than [depth] adjacency variables excluding Y
                    # then continue to the next variable X
                    if num_adj < depth: continue

                    # otherwise we are not done
                    # and we will check whether we may delete edges
                    done=False

                    for Y in adjacency_variables:

                        # do test if X and Y are no missingness pair
                        if graph.V2type[X]!='O' and graph.V2type[Y]!='O' and graph.V2num[X] == graph.V2num[Y]: continue

                        if (X,Y) in edges_to_delete or (Y,X) in edges_to_delete: continue

                        # keep the edge if assumption 3 of JCI is assumed
                        if '3' in JCI and X in graph.context and Y in graph.context: continue

                        # remove Y from adjacency set
                        adjXnotY = findAdjXnotY(graph.adjacencies(X), Y)

                        # remove missingness variable if X or Y is missingness indicator
                        if graph.V2type[Y]=='R' and graph.R2M[Y] in adjXnotY:
                            adjXnotY.remove(graph.R2M[Y])
                        if graph.V2type[X]=='R' and graph.R2M[X] in adjXnotY:
                            adjXnotY.remove(graph.R2M[X])

                        # add missingness indicators of missing variables in adjXnotY
                        for V in adjXnotY:
                            if graph.V2type[V]=='M' and not graph.M2R[V] in adjXnotY:
                                adjXnotY.append(graph.M2R[V])

                        # count missingness indicators if X or Y is missingness variable
                        num_missingness_indicators = 0
                        ss_extended = []
                        if graph.V2type[Y]=='M':
                            num_missingness_indicators += 1
                            if graph.M2R[Y] in adjXnotY:
                                adjXnotY.remove(graph.M2R[Y])
                            ss_extended.append(graph.M2R[Y])
                        if graph.V2type[X]=='M':
                            num_missingness_indicators += 1
                            if graph.M2R[X] in adjXnotY:
                                adjXnotY.remove(graph.M2R[X])
                            ss_extended.append(graph.M2R[X])

                        # only select variables from tiers which are not 
                        if not tiers is None:
                            adjXnotY = [Z for Z in adjXnotY if tiers[X]>=tiers[Z] or tiers[Y]>=tiers[Z]]
                        elif X in graph.context and Y in graph.context:
                            adjXnotY = [Z for Z in adjXnotY if Z in graph.context]

                        # find all possible subsets of size [depth] in adjXnotY
                        depth_tmp = depth - num_missingness_indicators
                        if depth_tmp < 0: continue

                        subsets = combinations(adjXnotY, depth_tmp)
                        for subset in subsets:

                            # check if subset is testable
                            testable_subset = list(subset)
                            testable = True                            
                            for V in subset:
                                if graph.V2type[V]=='M':
                                    if graph.M2R[V] in subset:
                                        testable_subset.remove(graph.M2R[V])
                                    else:
                                        testable = False
                                        break

                            testable_subset = tuple(testable_subset)

                            # check if X and Y are independent given [testable_subset]
                            if verbose and testable: 
                                print('Testing for', X, 'independent', Y, 'given', tuple(list(subset)+ss_extended), 'with testable set',testable_subset)
                            if testable:
                                if IT.oracle_testing:
                                    ind = IT.execute( graph, X, Y, condition=tuple(list(subset)+ss_extended))
                                else:
                                    ind = IT.execute( graph, X, Y, condition=testable_subset)

                                if ind:
                                    # save pair: we will delete it on a later stage
                                    edges_to_delete.append((X,Y))

                                    # save the separating sets
                                    separating_sets[(X,Y)] = tuple(list(subset)+ss_extended)
                                    separating_sets[(Y,X)] = tuple(list(subset)+ss_extended)

                                    # save the testable separating sets
                                    if not testable_separating_sets is None:
                                        testable_separating_sets[(X,Y)] = testable_subset
                                        testable_separating_sets[(Y,X)] = testable_subset

                                    break

                for X, Y in edges_to_delete:
                    if X in graph.adjacencies(Y):
                        # print(X,Y)
                        # remove the edge between X and Y
                        # if (X == 'chronic_lung_disease' and Y == 'pre_covid_pan') or (Y == 'chronic_lung_disease' and X == 'pre_covid_pan'):
                        #     print('this is the sepset', separating_sets[(X,Y)])
                        graph.deleteEdge(X, Y, rule=rule)
                        if verbose: print('Delete edge:', X, Y)

                depth += 1

        else:

            visited = defaultdict(lambda: defaultdict(lambda: False))

            print('add check for when we have missingness indicators ')

            for X in graph.all_variables:
                adjacency_variables = graph.adjacencies(X)

                for Y in adjacency_variables:

                    # keep the edge if assumption 3 of JCI is assumed
                    if '3' in JCI and X in graph.context and Y in graph.context: continue

                    done = False

                    if visited[X][Y]: continue

                    visited[X][Y] = True
                    visited[Y][X] = True

                    # find the list with adjacency variable of X excluding Y
                    possible_d_ss = findAdjXnotY(findAdjXnotY(graph.all_variables, Y), X)

                    for depth in range(len(graph.all_variables)-2):
                        
                        # find all possible subsets of size [depth] in possible_d_ss
                        subsets = combinations(possible_d_ss, depth)
                        for subset in subsets:

                            # check if X and Y are independent given [subset]
                            if IT.execute(graph, X, Y, condition=subset):
                                print(X,Y, subset)
                                # remove the edge between X and Y
                                graph.deleteEdge(X, Y)
                                # save the separating sets
                                separating_sets[(X,Y)] = subset
                                separating_sets[(Y,X)] = subset
                                done = True
                                break

                        if done: break

    else:
        for X in graph.all_variables:
            adjacency_variables = graph.adjacencies(X)
            for Y in adjacency_variables:
                if X in oracle.keys() and Y in oracle[X].keys():
                    subset = tuple(oracle[X][Y][0])
                    graph.deleteEdge(X, Y)
                    separating_sets[(X,Y)] = subset
                    separating_sets[(Y,X)] = subset
                elif Y in oracle.keys() and X in oracle[Y].keys():
                    subset = tuple(oracle[Y][X][0])
                    graph.deleteEdge(X, Y)
                    separating_sets[(X,Y)] = subset
                    separating_sets[(Y,X)] = subset


def jci(graph, JCI='0'):
    """
    Orientations according to 'Joint causal inference from multiple contexts' by Joris Mooij et al. (2020).
        :param graph: CausalGraph
        :param JCI: string ('0', '1', '12', '123', default is '0') indicating which  
                    assumtions from 'Joint Causal Inference from Multiple Contexts'
                    by Joris Mooij et al. (2020) are assumed to be true
    """
    if JCI != '0':
        for X in graph.context:
            for Y in graph.adjacencies(X):
                if Y in graph.context:
                    if '3' in JCI:
                        # update edge to X<->Y
                        graph.updateEdge(graph.head, X, Y)
                        graph.updateEdge(graph.head, Y, X)
                elif '1' in JCI:
                    # update edge Xo->Y
                    graph.updateEdge(graph.head, Y, X)
                    if '2' in JCI:
                        # update edge X->Y
                        graph.updateEdge(graph.tail, X, Y)


def background(graph, bkg_info=None):
    """
    Orientations according to background information.
        :param graph: CausalGraph
        :param bkg_info: background information in the form of an ancestral graph 
                    represented by a matrix where [i,j]=1 if i is an ancestor of j, 
                    [i,j]=1 if i is a non-descendants of j, and 0 means no info
    """
    if not bkg_info is None:
        for X in graph.all_variables:
            for Y in graph.adjacencies(X):
                if bkg_info.loc[X,Y] == 1 and bkg_info.loc[Y,X] == 1:
                    raise ValueError('Background information suggests cycles: redefine background or choose algorithm that works for cycles.')
                elif bkg_info.loc[X,Y] == 1:
                    # bkg_info.loc[Y,X] is 0 or -1
                    # X -> Y
                    graph.updateCause(X,Y)
                elif bkg_info.loc[Y,X] == 1:
                    # bkg_info.loc[X,Y] is 0 or -1
                    # X <- Y
                    graph.updateCause(Y,X)
                elif bkg_info.loc[X,Y] == -1 and bkg_info.loc[Y,X] == -1:
                    # X <-> Y
                    graph.updateEdge(graph.head, X, Y)
                    graph.updateEdge(graph.head, Y, X)
                elif bkg_info.loc[X,Y] == -1:
                    # bkg_info.loc[Y,X] is 0
                    # X <-* Y
                    graph.updateEdge(graph.head, X, Y)
                elif bkg_info.loc[Y,X] == -1:
                    # bkg_info.loc[X,Y] is 0
                    # X *-> Y
                    graph.updateEdge(graph.head, Y, X)


def addColliders(graph, separating_sets, testable_separating_sets=None, ind_test_result=None, PC=False, conservative=False, IT=None, uncertain_triples=None, bkg_info=None, tiers=None, oracle_conservative=False, mFCI_R0=False,verbose=False):
    """
    Rule 0 in Zhang (2008) where the colliders are added.
        :param graph: CausalGraph
        :param separating_sets: dictionary of sets for each pair of separated vertices
        :param testable_separating_sets: dictionary of testable separating sets for each pair of separated vertices
        :param ind_test_result: dictionary with independence test results 
                (e.g. ind_test_result[A][B][missing_num] = (Bool_without, Bool_with_Ri, Bool_with_XiRi))
        :param PC: boolean (if PC, we cannot orient bidirected edges)
        :param conservative: boolean indicating whether we do an additional conservative check
        :param IT: IndependenceTest object
        :param uncertain_triples: defaultdict(list) to track the uncertain triples when conservative=True
        :param bkg_info: background information in the form of an ancestral graph 
                        represented by a matrix where [i,j]=1 if i is an ancestor of j, 
                        [i,j]=1 if i is a non-descendants of j, and 0 means no info
        :param oracle_conservative: Boolean indicating whether there is oracle information given over the uncertain triples
    """

    # check if IT is correctly defined when necessary
    if conservative or mFCI_R0:
        if not isinstance(IT, IndependenceTest):
            raise TypeError('IndependenceTest IT not correctly defined.')

    # add colliders
    for Y in graph.variables:

        adjacencies = list(graph.adjacencies(Y))
        if graph.V2type[Y] == 'M':
            adjacencies = list(set(adjacencies)-{graph.M2R[Y]})
        elif graph.V2type[Y] == 'R':
            adjacencies = list(set(adjacencies)-{graph.R2M[Y]})

        # find all pairs of adjacency variables of Y
        adjacency_pairs = combinations(adjacencies, 2)

        for X, Z in adjacency_pairs:

            # skip pair if X and Y are missingness pair
            if graph.V2type[X] != 'O' and graph.V2type[Z] != 'O' and graph.V2num[X]==graph.V2num[Z]: 
                continue

            # check if X and Y are not adjacent
            if X not in graph.adjacencies(Z):

                if verbose: print('Consider unshielded triplet:', X, Y, Z)

                # background info overrules independence info
                if not bkg_info is None and (graph.incomingArrowType(Y, X) == graph.tail or graph.incomingArrowType(X,Y) == graph.tail):
                    continue

                # check if Y is not in the separating set of X and Z if exists
                if not separating_sets[(X,Z)] is None and not Y in separating_sets[(X,Z)]:

                    # conservative testing (only run in case there is no background knowledge about the tiers)
                    if conservative and (tiers is None or (tiers[X]>=tiers[Y] or tiers[Z]>=tiers[Y])):

                        if verbose: print('Test conservative')
                        if oracle_conservative:
                            uncertain_collider = Y in uncertain_triples[(X,Z)] or Y in uncertain_triples[(Z,X)]
                        else:
                            # we do not have to check here if Y is in a later tier
                            # because, in that case, Y is a collider by convention
                            # idem dito for rule R4
                            uncertain_collider = IT.execute(graph, X, Z, condition=list(separating_sets[(X,Z)])+[Y])
                            if uncertain_collider:
                                uncertain_triples[(X,Z)].append(Y)
                                if verbose: print('Conservative check: we found uncertain collider:', X, Y, Z)
                            else:
                                if verbose: print('Conservative check: we found collider:', X, Y, Z)

                    # check if pair is oriented according to missingness pair orientation
                    elif mFCI_R0:

                        # print('Do mFCI_R0')

                        # ind_given_Ry = True
                        if graph.V2type[Y] == 'M':

                            if verbose: print('Get independence test results for', Y, 'in M with separating set', separating_sets[(X,Z)], 'for', X, Z, ind_test_result[X][Z][graph.V2num[Y]])

                            if not graph.M2R[Y] in separating_sets[(X,Z)]:

                                # track independence test results
                                # if ind_test_result[X][Z][graph.V2num[Y]][0] is None:
                                #     ind_test_result[X][Z][graph.V2num[Y]][0] = True
                                #     ind_test_result[Z][X][graph.V2num[Y]][0] = ind_test_result[X][Z][graph.V2num[Y]][0]
                                if ind_test_result[X][Z][graph.V2num[Y]][1] is None:
                                    if verbose: print('R0(1): test', X, Z, 'independent given testable set', list(testable_separating_sets[(X,Z)])+[graph.M2R[Y]])
                                    if IT.oracle_testing:
                                        ind_test_result[X][Z][graph.V2num[Y]][1] = IT.execute(graph, X, Z, condition=list(separating_sets[(X,Z)])+[graph.M2R[Y]])
                                    else:
                                        ind_test_result[X][Z][graph.V2num[Y]][1] = IT.execute(graph, X, Z, condition=list(testable_separating_sets[(X,Z)])+[graph.M2R[Y]])
                                    ind_test_result[Z][X][graph.V2num[Y]][1] = ind_test_result[X][Z][graph.V2num[Y]][1]
                                if ind_test_result[X][Z][graph.V2num[Y]][2] is None:
                                    if verbose: print('R0(2): test', X, Z, 'independent given testable set', list(testable_separating_sets[(X,Z)])+[Y])
                                    if IT.oracle_testing:
                                        ind_test_result[X][Z][graph.V2num[Y]][2] = IT.execute(graph, X, Z, condition=list(separating_sets[(X,Z)])+[graph.M2R[Y],Y])
                                    else:
                                        ind_test_result[X][Z][graph.V2num[Y]][2] = IT.execute(graph, X, Z, condition=list(testable_separating_sets[(X,Z)])+[Y])
                                    ind_test_result[Z][X][graph.V2num[Y]][2] = ind_test_result[X][Z][graph.V2num[Y]][2]

                            else:
                                # track independence test results
                                # if ind_test_result[X][Z][graph.V2num[Y]][0] is None:
                                #     ind_test_result[X][Z][graph.V2num[Y]][0] = False
                                #     ind_test_result[Z][X][graph.V2num[Y]][0] = ind_test_result[X][Z][graph.V2num[Y]][0]
                                if ind_test_result[X][Z][graph.V2num[Y]][1] is None:
                                    ind_test_result[X][Z][graph.V2num[Y]][1] = True
                                    ind_test_result[Z][X][graph.V2num[Y]][1] = ind_test_result[X][Z][graph.V2num[Y]][1]
                                if ind_test_result[X][Z][graph.V2num[Y]][2] is None:
                                    if verbose: print('R0(3): test', X, Z, 'independent given testable set', list(set(testable_separating_sets[(X,Z)])-{graph.M2R[Y]})+[Y])
                                    if IT.oracle_testing:
                                        ind_test_result[X][Z][graph.V2num[Y]][2] = IT.execute(graph, X, Z, condition=list(separating_sets[(X,Z)])+[Y])
                                    else:
                                        ind_test_result[X][Z][graph.V2num[Y]][2] = IT.execute(graph, X, Z, condition=list(set(testable_separating_sets[(X,Z)])-{graph.M2R[Y]})+[Y])
                                    ind_test_result[Z][X][graph.V2num[Y]][2] = ind_test_result[X][Z][graph.V2num[Y]][2]

                            # to orient we must have independence given Ri and not independence given Ri and Xi
                            uncertain_collider = not (ind_test_result[X][Z][graph.V2num[Y]][1] and not ind_test_result[X][Z][graph.V2num[Y]][2])

                        elif graph.V2type[Y] == 'R':

                            if verbose: print('Get independence test results for', Y, 'in R with separating set', separating_sets[(X,Z)], 'for', X, Z, ind_test_result[X][Z][graph.V2num[Y]])

                            # in this case X(Y) not in sep set by definition
                            if graph.R2M[Y] in separating_sets[(X,Z)]:
                                raise ValueError(X,Y,Z, 'Sep set is not well defined in code.')

                            # if ind_test_result[X][Z][graph.V2num[Y]][0] is None:
                            #     ind_test_result[X][Z][graph.V2num[Y]][0] = True
                            #     ind_test_result[Z][X][graph.V2num[Y]][0] = ind_test_result[X][Z][graph.V2num[Y]][0]
                            if ind_test_result[X][Z][graph.V2num[Y]][1] is None:
                                if verbose: print('R0(4): test', X, Z, 'independent given testable set', list(testable_separating_sets[(X,Z)])+[Y])
                                if IT.oracle_testing:
                                    ind_test_result[X][Z][graph.V2num[Y]][1] = IT.execute(graph, X, Z, condition=list(separating_sets[(X,Z)])+[Y])
                                else:
                                    ind_test_result[X][Z][graph.V2num[Y]][1] = IT.execute(graph, X, Z, condition=list(testable_separating_sets[(X,Z)])+[Y])
                                ind_test_result[Z][X][graph.V2num[Y]][1] = ind_test_result[X][Z][graph.V2num[Y]][1]

                            uncertain_collider = ind_test_result[X][Z][graph.V2num[Y]][1]
                            
                        else:
                            if verbose: print('Get independence test results for', Y, 'in O with separating set', separating_sets[(X,Z)], 'for', X, Z)

                            if IT.oracle_testing:
                                uncertain_collider = IT.execute(graph, X, Z, condition=list(separating_sets[(X,Z)])+[Y])
                            else:
                                uncertain_collider = IT.execute(graph, X, Z, condition=list(testable_separating_sets[(X,Z)])+[Y])

                        if uncertain_collider:
                            uncertain_triples[(X,Z)].append(Y)
                            if verbose: print('Extra check: we found uncertain collider:', X, Y, Z)
                        else:
                            if verbose: print('Extra check: we found collider:', X, Y, Z)


                    else:
                        uncertain_collider = False

                    if PC: # never conservative
                        if graph.incomingArrowType(X, Y) == graph.tail and graph.incomingArrowType(Z, Y) == graph.tail:
                            # update edge X-Y to X->Y  and edge Z-Y to Z->Y
                            graph.updateEdge(graph.head, Y, X)
                            graph.updateEdge(graph.head, Y, Z)
                            if verbose:
                                print('R0')
                                print(X, '->', Y, '<-', Z)

                    elif not uncertain_collider: # FCI
                        # update edge Xo-oY to Xo->Y  and edge Zo-oY to Zo->Y
                        graph.updateEdge(graph.head, Y, X)
                        graph.updateEdge(graph.head, Y, Z)
                        # if Y=='delirium' and (X=='bmi'or Z=='bmi'):
                        #     print('R0')
                        if verbose:
                            print('R0: orient', X, '*->', Y, '<-*', Z)


def addNonCollidersPC(graph):
    """
    Remaining orientation rules of PC.
        :param graph: CausalGraph
    """

    done = False
    while not done:

        # initilize done again
        done = True

        directed_edges = graph.directedEdges()
        for A, B in directed_edges:
            # find the list with adjacency variable of B excluding A
            adjBnotA = findAdjXnotY(graph.adjacencies(B), A)
            for C in adjBnotA:
                # check if C is not adjacent to A
                # and check if B does not have an incoming arrow head from C
                if C not in graph.adjacencies(A) and (C,B) not in graph.directedEdges():
                    if (B,C) not in graph.directedEdges():
                        # update the edge (B-C) to (B->C)
                        graph.updateEdge(graph.head, C, B)
                        # if an update is possible, we have to do the whole check again
                        done = False

        # NOTE: this part assumes acyclicity
        directed_paths = graph.directedPaths()
        for A, B in directed_paths:
            if B in graph.adjacencies(A):
                if (A, B) not in graph.directedEdges():
                    # update the edge (A-B) to (A->B)
                    # graph.updateCause(A, B)
                    graph.updateEdge(graph.head, B, A)
                    # if an update is possible, we have to do the whole check again
                    done = False


def qreach(x, A, Xm=[]):
    # https://rdrr.io/cran/pcalg/man/qreach.html

    A_ = A!=0
    PSEP = list(np.where(A_[x,:])[0])
    Q = PSEP[:]
    nb = PSEP[:]
    P = len(Q)*[x]
    A_[x,nb] = False
    while len(Q)>0:
        # set current vertex
        a = Q[0]
        # reset list
        Q = Q[1:]
        # set previous vertex
        pred = P[0]
        # reset list
        P = P[1:]
        # select new neighboors
        nb = list(np.where(A_[a,:])[0])
        # loop over the neighboors
        for b in nb:
            # check if a in Xm or there is a collider at a or a triangle
            if a in Xm or (A[pred,a]==2  and A[b,a]==2) or A[pred,b]!=0:
                # reset to A_ to not revisite this edge
                A_[a,b] = False
                # append b, a to Q, P (resp) to revisit this path later
                Q.append(b)
                P.append(a)
                # append to b to PSEP
                PSEP.append(b)
    # sort PSEP and remove doubles to return
    return list(set(PSEP))



def deleteSeparatedEdgesStage2(graph, graph_copy, separating_sets, IT, run_mFCI=False, testable_separating_sets=None, tiers=None, track_deletions=False, verbose=True):
    """
        :param testable_separating_sets: dictionary of testable separating sets for each pair of separated vertices

    """

    if track_deletions:
        rule = 'phase2'
    else:
        rule = None

    # A = np.multiply(adjacencyMatrix(graph_copy)!=0,1)
    A = graph_copy.adjacencyMatrix()

    # indices of missingness variables
    if run_mFCI:
        Xm = [graph.all_variables.index(V) for V in graph.missingness_variables]
    else:
        Xm = []

    # get for each variable x the pdsep
    allPdsep = []
    for x, X in enumerate(graph.all_variables):
        Pdsep = qreach(x, A, Xm)
        allPdsep.append([graph.all_variables[v] for v in Pdsep])
    allPdsep_tmp = []
    
    # test independence between each pair X and Y
    for x, X in enumerate(graph.all_variables):

        # define adjacencies
        adjacencies = list(graph.adjacencies(X))

        # remove the corresponding missingness indicators/variable from the adjacencies
        if graph.V2type[X]=='M' and graph.M2R[X] in adjacencies:
            adjacencies.remove(graph.M2R[X])
        elif graph.V2type[X]=='R' and graph.R2M[X] in adjacencies:
            adjacencies.remove(graph.R2M[X])

        # extended adjacencies are the missingness indicators that
        # are required in the testable independence statements
        extended_adjacencies = []
        for V in adjacencies:
            if graph.V2type[V] == 'M' and not graph.M2R[V] in adjacencies:
                extended_adjacencies.append(graph.M2R[V])

        # remove X from pdsep
        allPdsep_tmp.append(list(set(allPdsep[x])-{X}))

        for Y in adjacencies:

            # select variables of possible separating set according to tier order
            if not tiers is None:
                tf = [Z for Z in allPdsep_tmp[x] if tiers[X]>=tiers[Z] or tiers[Y]>=tiers[Z]]
            else:
                tf = list(allPdsep_tmp[x])
            
            # select all variables except for Y
            tf = list(set(tf)-{Y})

            # intiate extended S (incl the missingness indicators considered for this pair)
            ss_extended = []
            # initiate depth
            i = 0
            extended_adjacencies_tmp = list(extended_adjacencies)
            # if X or Y are missingness variables: add missigness indicator to extended S + reset depth
            # if X or Y are missingnes indicators: remove the correspnoding missingness variable from pos ss
            if graph.V2type[X] == 'M':
                ss_extended.append(graph.M2R[X])
                tf = list(set(tf) - {graph.M2R[X]})
                extended_adjacencies_tmp = list(set(extended_adjacencies_tmp) - {graph.M2R[X]})
                i += 1
            elif graph.V2type[X] == 'R':
                tf = list(set(tf) - {graph.R2M[X]})
            if graph.V2type[Y] == 'M':
                ss_extended.append(graph.M2R[Y])
                tf = list(set(tf) - {graph.M2R[Y]})
                extended_adjacencies_tmp = list(set(extended_adjacencies_tmp) - {graph.M2R[Y]})
                i += 1
            elif graph.V2type[Y] == 'R':
                tf = list(set(tf) - {graph.R2M[Y]})

            len_ss_extended = len(ss_extended)
            visited = adjacencies + extended_adjacencies_tmp

            # include missingness indicators
            for V in tf:
                if graph.V2type[V] == 'M' and not graph.M2R[V] in tf:
                    tf.append(graph.M2R[V])

            # find max depth
            max_depth = len(tf)+len_ss_extended
            
            done = False
            while not done and i < max_depth:

                i += 1

                if i == 1:

                    # for depth=1 we do not need to test for adjacencies
                    diff_set = list(set(tf)-set(visited))

                    # for depth=1 we can not test for missingness variables
                    diff_set = [V for V in diff_set if graph.V2type[V]!='M']

                    # note that: if len(ss_extended)>0 then i > 1
                    for S in diff_set:

                        if verbose:
                            print('Testing for', X, 'independent', Y, 'given', [S])

                        if IT.execute(graph, X, Y, condition=[S]):

                            if verbose: print('Delete edge:', X, Y)

                            # delete edge
                            graph.deleteEdge(X, Y, rule=rule)
                            done = True

                            # save the separating sets
                            separating_sets[(X,Y)] = [S]
                            separating_sets[(Y,X)] = [S]
                            if not testable_separating_sets is None:
                                testable_separating_sets[(X,Y)] = [S]
                                testable_separating_sets[(Y,X)] = [S]

                            break

                else:
                    # i > 1

                    # get combinations of tf
                    subsets = combinations(tf, i)

                    for subset in subsets:

                        # check if subset is testable
                        testable_subset = list(subset)
                        testable = True                            
                        for V in subset:
                            if graph.V2type[V]=='M':
                                if graph.M2R[V] in subset:
                                    testable_subset.remove(graph.M2R[V])
                                else:
                                    testable = False
                                    break

                        testable_subset = tuple(testable_subset)

                        # if the subset is testable
                        if testable:
                            # test if some elements in S are different to what is already tested
                            if i > len(visited) or not set(subset).issubset(set(visited)):

                                if verbose:
                                    print('Testing for', X, 'independent', Y, 'given', list(subset) + ss_extended, 'and testable set', testable_subset)

                                if IT.oracle_testing:
                                    ind = IT.execute(graph, X, Y, condition=tuple(list(subset) + ss_extended))
                                else:
                                    ind = IT.execute(graph, X, Y, condition=testable_subset)

                                if ind:

                                    if verbose: print('Delete edge:', X, Y)

                                    # delete edge
                                    graph.deleteEdge(X, Y, rule=rule)
                                    done = True

                                    # save the separating sets
                                    separating_sets[(X,Y)] = tuple(list(subset) + ss_extended)
                                    separating_sets[(Y,X)] = tuple(list(subset) + ss_extended)
                                    if not testable_separating_sets is None:
                                        testable_separating_sets[(X,Y)] = testable_subset 
                                        testable_separating_sets[(Y,X)] = testable_subset

                                    break



def searchPathsRecursive(graph, current, previous, final, visited):
    """
    Recursive function returns all possible paths where each path is an array of colliders and non-colliders. 
    The order of the vertices in the path is not important for this purpose, so we don't save the order.
        :param graph: causalGraph object
        :param current, previous: current and previous vertex on a path.
        :param final: final vertex on the path
        :param visited: all visited vertices on the path
    """

    # select new next vertices on the path
    nexts = list(set(graph.adjacencies(current)) - set(visited))

    # initialize possible paths
    colliders_in_paths = []
    pos_non_colliders_in_paths = []

    # loop over the next variables
    for next_var in nexts:

        # first we search for rest of paths
        if next_var != final:
            rest_colliders_in_paths, rest_pos_non_colliders_in_paths = searchPathsRecursive(graph, next_var, current, final, visited+[current])
        else:
            M = len(graph.all_variables)
            rest_colliders_in_paths, rest_pos_non_colliders_in_paths = [np.zeros(M)], [np.zeros(M)]
            
        # check if the current triple form a collider
        if graph.incomingArrowType(current, previous)==graph.head and graph.incomingArrowType(current, next_var)==graph.head:
            for colliders_in_path in rest_colliders_in_paths:
                colliders_in_path[graph.all_variables.index(current)] = 1
        else:
            for pos_non_colliders_in_path in rest_pos_non_colliders_in_paths:
                pos_non_colliders_in_path[graph.all_variables.index(current)] = 1
        
        colliders_in_paths = colliders_in_paths + rest_colliders_in_paths
        pos_non_colliders_in_paths = pos_non_colliders_in_paths + rest_pos_non_colliders_in_paths

    return colliders_in_paths, pos_non_colliders_in_paths


def zhangsInferenceRules(graph, separating_sets, testable_separating_sets=None, ind_test_result=None, CDC=False, uncertain_triples=None, IT=None, conservative=False, tiers=None, oracle_conservative=False, verbose=False):
    """
    Rules 1-4 in Zhang (2008).
        :param graph: CausalGraph
        :param separating_sets: dictionary of sets for each pair of separated vertices
        :param testable_separating_sets: dictionary of testable separating sets for each pair of separated vertices
        :param ind_test_result: dictionary with independence test results 
                (e.g. ind_test_result[A][B][missing_num] = (Bool_without, Bool_with_Ri, Bool_with_XiRi))
        :param CDC: boolean (if CDC, R4 is slightly different and track CDC orientations)
        :param uncertain_triples: defaultdict(list) to track the uncertain triples when conservative=True
        :param IT: IndependenceTest object
        :param conservative: boolean indicating whether we do an additional conservative check
        :param oracle_conservative: Boolean indicating whether there is oracle information given over the uncertain triples
    """

    found_orientation = True

    if CDC:
        rule='FCI'
    else:
        rule=None

    while found_orientation:

        found_orientation = False

        for B in graph.variables:

            # find two adjacent variables
            adjacency_pairs = permutations(graph.adjacencies(B), 2)

            for A, C in adjacency_pairs:

                # if verbose: print('Consider triplet', A, B, C)

                # shielded triple
                if A in graph.adjacencies(C):

                    # if verbose: print('Triplet is shielded')

                    # R2 pull through directed path
                    if graph.incomingArrowType(C, A) == graph.unknown:
                        if graph.isDirectedEdge(A, B) and graph.incomingArrowType(C, B) == graph.head:
                            graph.updateEdge(graph.head, C, A, rule)
                            if C=='delirium' and A=='bmi':
                                print('R2')
                            if verbose:
                                print('R2')
                                print('Shielded triplet', A, B, C)
                                print(A, '*->', C)
                            # R2 update (1)
                            found_orientation = False
                        elif graph.isDirectedEdge(B, C) and graph.incomingArrowType(B, A) == graph.head:
                            graph.updateEdge(graph.head, C, A, rule)
                            if C=='delirium' and A=='bmi':
                                print('R2')
                            if verbose:
                                print('R2')
                                print('Shielded triplet', A, B, C)
                                print(A, '*->', C)
                            # R2 update (2)
                            found_orientation = True

                    # R4 discriminating path orientation
                    if graph.incomingArrowType(B, C) == graph.unknown and graph.score(B, C)!=-1:

                        # A is possible collider on discriminating path
                        # A is parent of C
                        # A-B and A - C are real edges
                        if graph.incomingArrowType(A, B) == graph.head and graph.isDirectedEdge(A, C) and graph.score(A, B)!=-1 and graph.score(A, C)!=-1:
                            D = findStartDiscriminatingPath(graph, A, B, C)
                            if D is not None:
                                uncertain_collider = False
                                if (conservative or graph.num_missingness_pairs!=0) and not CDC:
                                    if B in uncertain_triples[(C,D)] or B in uncertain_triples[(D,C)]:
                                        uncertain_collider = True
                                    elif separating_sets[(C,D)] is None:
                                        uncertain_collider = True
                                    elif not B in separating_sets[(C,D)] and not oracle_conservative:
                                        if conservative and (tiers is None or (tiers[C]>=tiers[B] or tiers[D]>=tiers[B])):
                                            if IT.execute(graph, C, D, condition=list(separating_sets[(C,D)])+[B]):
                                                uncertain_collider = True
                                                uncertain_triples[(C,D)].append(B)
                                        else:
                                            # run extra check for m-FCI
                                            if graph.V2type[B] == 'M':

                                                # track independence test results
                                                if not graph.M2R[B] in separating_sets[(C,D)]:
                                                    if ind_test_result[C][D][graph.V2num[B]][1] is None:
                                                        if IT.oracle_testing:
                                                            ind_test_result[C][D][graph.V2num[B]][1] = IT.execute(graph, C, D, condition=list(separating_sets[(C,D)])+[graph.M2R[B]])
                                                        else:
                                                            ind_test_result[C][D][graph.V2num[B]][1] = IT.execute(graph, C, D, condition=list(testable_separating_sets[(C,D)])+[graph.M2R[B]])
                                                        ind_test_result[D][C][graph.V2num[B]][1] = ind_test_result[C][D][graph.V2num[B]][1]
                                                    if ind_test_result[C][D][graph.V2num[B]][2] is None:
                                                        if IT.oracle_testing:
                                                            ind_test_result[C][D][graph.V2num[B]][2] = IT.execute(graph, C, D, condition=list(separating_sets[(C,D)])+[graph.M2R[B],B])
                                                        else:
                                                            ind_test_result[C][D][graph.V2num[B]][2] = IT.execute(graph, C, D, condition=list(testable_separating_sets[(C,D)])+[B])
                                                        ind_test_result[D][C][graph.V2num[B]][2] = ind_test_result[C][D][graph.V2num[B]][2]
                                                else:
                                                    if ind_test_result[C][D][graph.V2num[B]][1] is None: 
                                                        ind_test_result[C][D][graph.V2num[B]][1] = True
                                                        ind_test_result[D][C][graph.V2num[B]][1] = ind_test_result[C][D][graph.V2num[B]][1]
                                                    if ind_test_result[C][D][graph.V2num[B]][2] is None: 
                                                        if IT.oracle_testing:
                                                            ind_test_result[C][D][graph.V2num[B]][2] = IT.execute(graph, C, D, condition=list(separating_sets[(C,D)])+[B])
                                                        else:
                                                            ind_test_result[C][D][graph.V2num[B]][2] = IT.execute(graph, C, D, condition=list(set(testable_separating_sets[(C,D)]) - {graph.M2R[B]})+[B])
                                                        ind_test_result[D][C][graph.V2num[B]][2] = ind_test_result[C][D][graph.V2num[B]][2]

                                                # uncertain direction if we do not find that
                                                # C ind D given Ri, and not C ind D given Ri and Xi
                                                uncertain_collider = not (ind_test_result[C][D][graph.V2num[B]][1] and not ind_test_result[C][D][graph.V2num[B]][2])

                                            elif graph.V2type[B] == 'R':

                                                # track independence test results
                                                if ind_test_result[C][D][graph.V2num[B]][1] is None:
                                                    if IT.oracle_testing:
                                                        ind_test_result[C][D][graph.V2num[B]][1] = IT.execute(graph, C, D, condition=list(separating_sets[(C,D)])+[B])
                                                    else:
                                                        ind_test_result[C][D][graph.V2num[B]][1] = IT.execute(graph, C, D, condition=list(testable_separating_sets[(C,D)])+[B])
                                                    ind_test_result[D][C][graph.V2num[B]][1] = ind_test_result[C][D][graph.V2num[B]][1]

                                                # uncertain direction if we do not find that C ind D given Ri
                                                uncertain_collider = ind_test_result[C][D][graph.V2num[B]][1] 

                                            # elif conservative_mFCI:
                                            #     uncertain_collider = IT.execute(graph, C, D, condition=sepset_CD_tmp+[B])

                                            if uncertain_collider:
                                                uncertain_triples[(C,D)].append(B)

                                if not uncertain_collider:
                                    if B in separating_sets[(C,D)] or CDC:
                                        graph.updateEdge(graph.head, C, B, rule)
                                        graph.updateEdge(graph.tail, B, C, rule)
                                        if verbose:
                                            print('R4')
                                            print('Shielded triplet', A, B, C)
                                            print(B, '->', C)
                                        # if C=='delirium' and B=='bmi':
                                        #     print('R4')
                                    else:
                                        graph.updateEdge(graph.head, C, B)
                                        graph.updateEdge(graph.head, B, C)
                                        graph.updateEdge(graph.head, A, B)
                                        graph.updateEdge(graph.head, B, A)
                                        if verbose:
                                            print('R4')
                                            print('Shielded triplet', A, B, C)
                                            print(B, '<->', C)
                                            print(B, '<->', A)

                # unshielded triple
                else:

                    # if verbose: print('Triplet is unshielded')

                    # if A*->B and A-B, B-C are real
                    if graph.incomingArrowType(B, A) == graph.head and graph.score(A, B)!=-1 and graph.score(B, C)!=-1:

                        # R1 add non-collider
                        if graph.incomingArrowType(B, C) == graph.unknown:
                            uncertain_collider = False
                            if conservative or graph.num_missingness_pairs!=0:
                                uncertain_collider = B in uncertain_triples[(A,C)] or B in uncertain_triples[(C,A)]
                            if not uncertain_collider:
                                graph.updateEdge(graph.head, C, B, rule)
                                graph.updateEdge(graph.tail, B, C, rule)
                                if verbose:
                                    print('R1')
                                    print('Unshielded triplet', A, B, C)
                                    print(B, '->', C)
                                found_orientation = True

                        # R3 complete triple collider
                        elif graph.incomingArrowType(B, C) == graph.head:
                            for D in graph.adjacencies(B):
                                if graph.incomingArrowType(B, D) == graph.unknown and graph.score(B, D)!=-1:
                                    if A in graph.adjacencies(D) and graph.incomingArrowType(D, A) == graph.unknown and graph.score(A, D)!=-1:
                                        if C in graph.adjacencies(D) and graph.incomingArrowType(D, C) == graph.unknown and graph.score(C, D)!=-1:
                                            graph.updateEdge(graph.head, B, D, rule)
                                            found_orientation = True
                                            if verbose:
                                                print('R3')
                                                print('Unshielded triplet', A, B, C)
                                                print(D, '*->', B)


def completeUndirectedEdges(graph, verbose=False):
    """
        Complete undirected edges when we assume that there might be selection bias.
            :param graph: CausalGraph
    """

    found_orientation = True
    while found_orientation:
        found_orientation = False

        # R5
        for A in graph.variables:
            circle_adjacencies = []
            for variable in graph.adjacencies(A):
                if graph.score(A, variable)!=-1 and graph.incomingArrowType(A, variable) == graph.unknown and graph.incomingArrowType(variable, A) == graph.unknown:
                    circle_adjacencies.append(variable)
            circle_adjacency_pairs = permutations(circle_adjacencies, 2)
            for B, C in circle_adjacency_pairs:
                if B not in graph.adjacencies(C):
                    for D in graph.adjacencies(B):
                        if A != D and D not in graph.adjacencies(A):
                            if graph.incomingArrowType(D, B) == graph.unknown and graph.incomingArrowType(B, D) == graph.unknown and graph.score(B, D)!=-1:
                                uncovered_circle_paths = findUncoveredCirclePath(graph, A, B, C, D)
                                for path in uncovered_circle_paths:
                                    for i in range(len(path)-1):
                                        graph.updateEdge(graph.tail, path[i], path[i+1])
                                        graph.updateEdge(graph.tail, path[i+1], path[i])
                                        if verbose:
                                            print('R5')
                                            print(path[i], '-', path[i+1])

                                if uncovered_circle_paths != []:
                                    graph.updateEdge(graph.tail, B, A)
                                    graph.updateEdge(graph.tail, A, B)
                                    found_orientation = True
                                    if verbose:
                                        print('R5')
                                        print(A, '-', B)

        # R6 and R7
        for B in graph.variables:
            adjacency_pairs = permutations(graph.adjacencies(B), 2)
            for A, C in adjacency_pairs:
                if graph.incomingArrowType(B, C) == graph.unknown and graph.incomingArrowType(A, B) == graph.tail:
                    if graph.incomingArrowType(B, A) == graph.tail:
                        graph.updateEdge(graph.tail, B, C)
                        found_orientation = True
                        if verbose:
                            print('R6')
                            print(B, '-*', C)
                    elif graph.incomingArrowType(B, A) == graph.unknown and A not in graph.adjacencies(C):
                        graph.updateEdge(graph.tail, B, C)
                        found_orientation = True
                        if verbose:
                            print('R7')
                            print(B, '-*', C)


def completeDirectedEdges(graph, CDC=False, verbose=False):
    """
    Rules 8-10 in Zhang (2008).
        :param graph: CausalGraph
        :param CDC: boolean (if CDC, track CDC orientations)
    """

    found_orientation = True

    if CDC:
        rule='FCI'
    else:
        rule=None

    while found_orientation:
        found_orientation = False

        for A in graph.variables:
            for C in graph.adjacencies(A):
                if graph.incomingArrowType(C, A) == graph.head and graph.incomingArrowType(A, C) == graph.unknown:

                    directed_edge = False

                    # R8
                    for B in graph.adjacencies(A):
                        if B in graph.adjacencies(C):
                            if graph.incomingArrowType(B, A) != graph.tail and graph.incomingArrowType(A, B) == graph.tail:
                                if graph.isDirectedEdge(B,C):
                                    directed_edge = True
                                    if verbose:
                                        print('R8')
                                        print(A, '->', C)
                                    break

                    # R9
                    if not directed_edge and graph.score(A, C)!=-1:
                        uncovered_pd_paths = findUncoveredPotentiallyDirectedPath(graph, A, C)
                        for path in uncovered_pd_paths:
                            if path[1]!=C and path[1] not in graph.adjacencies(C):
                                directed_edge = True
                                if verbose:
                                    print('R9')
                                    print(A, '->', C)
                                break

                    # R10
                    if not directed_edge and graph.score(A, C)!=-1:
                        adjacencies = graph.adjacencies(C)
                        adjacencies.remove(A)
                        parents = [variable for variable in adjacencies if graph.isDirectedEdge(variable, C) and graph.score(variable, C)!=-1]
                        parents_pairs = combinations(parents, 2)
                        for B, D in parents_pairs:
                            if directed_edge: break
                            AB_uncovered_pd_paths = findUncoveredPotentiallyDirectedPath(graph, A, B)
                            AD_uncovered_pd_paths = findUncoveredPotentiallyDirectedPath(graph, A, D)
                            for AB_path in AB_uncovered_pd_paths:
                                if directed_edge: break
                                for AD_path in AD_uncovered_pd_paths:
                                    if AB_path[1] != AD_path[1] and AB_path[1] not in graph.adjacencies(AD_path[1]):
                                        directed_edge = True
                                        if verbose:
                                            print('R10')
                                            print(A, '->', C)
                                        break

                    if directed_edge:
                        graph.updateEdge(graph.tail, A, C, rule)
                        found_orientation = True


def adjmatrix2CausalGraph2(variables, M):
    """
    Returns CausalGraph object corresponding to M.
        :param variables: the list of variables
        :param M: graph in matrix form
    """
    G = CausalGraph(variables, unknown='u', default_edge='u')
    for i in range(M.shape[0]):

        for j in range(M.shape[1]):

            if M[i,j] != 0  and not variables[i] in G.adjacencies(variables[j]):
                raise ValueError('Inconsistant adjacency matrix.')

            if variables[i] in G.adjacencies(variables[j]):

                if M[i,j] == 0:
                    # remove edge
                    G.deleteEdge(variables[i], variables[j])

                elif M[i,j] == 2:
                    # add head: i <-* j
                    G.updateEdge(G.head, variables[i], variables[j])

                elif M[i,j] == 3:
                    # add head: i <-* j
                    G.updateEdge(G.tail, variables[i], variables[j])

    return G


