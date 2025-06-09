from causalGraph import CausalGraph
from searchFunctions import *
from boolFunctions import isSeparated
import copy


def selectPossibleDeconfoundingSets(PAG, X, Y):
    """
    Returns the definite and possible deconfounding set (which are disjoint sets).
        :param PAG: causalGraph object
        :param X/Y: vertex
    """

    # D1: select possible parents
    possible_parents = possibleParents(PAG, Y)
    if X in possible_parents:
        possible_parents.remove(X)

    # compute MAG as in Theorem 2 [Zhang (2008)]
    MAG = PAG2MAG(PAG)

    # remove edges adjacent to Y
    adjacencies = MAG.adjacencies(Y)
    for neighbour in adjacencies:
        MAG.deleteEdge(Y, neighbour)

    # find ancestors
    ancestors = findAncestors(MAG)

    # D2: select possible parents which are relevant for m-separation
    possible_parents_copy = possible_parents.copy()
    for parent in possible_parents:
        possible_parents_copy.remove(parent)
        if isSeparated(MAG, parent, X, possible_parents_copy, ancestors)=='open':
            possible_parents_copy.append(parent)

    # D3: select the set satisfying Lemma 3 (i)
    def_deconf = [] # P
    pos_deconf = [] # Q \ P
    for parent in possible_parents_copy:
        if lemma_3_i(PAG, X, Y, parent):
            def_deconf.append(parent)
        else:
            pos_deconf.append(parent)

    return def_deconf, pos_deconf


def possibleParents(graph, Y):
    """
    Returns all possible parents (including definite parents).
        :param graph: causalGraph object
        :param Y: vertex
    """
    possible_parents = []
    for Z in graph.adjacencies(Y):
        if graph.incomingArrowType(Y,Z)!=graph.tail and graph.incomingArrowType(Z,Y)!=graph.head:
            # Z o-o Y, Z o-> Y, Z -> Y
            possible_parents.append(Z)
    return possible_parents


def PAG2MAG(PAG):
    """
    Theorem 2 Zhang&co (2008)
        :param PAG: causalGraph object
    """
    MAG = copy.deepcopy(PAG)
    unknown_edges = {}
    unknown_adj = {}

    for A in MAG.all_variables:
        for B in MAG.adjacencies(A):

            if MAG.incomingArrowType(A,B)==MAG.unknown and MAG.incomingArrowType(B,A)==MAG.head:
                # if A o-> B, then A -> B
                MAG.updateCause(A, B)

            elif MAG.incomingArrowType(A,B)==MAG.unknown and MAG.incomingArrowType(B,A)==MAG.tail:
                # if A o- B, then A <- B
                MAG.updateCause(B, A)

            elif MAG.incomingArrowType(A,B)==MAG.unknown and MAG.incomingArrowType(B,A)==MAG.unknown:

                # track number of unknown edges
                if A not in unknown_edges.keys():
                    unknown_edges[A] = 0
                unknown_edges[A] +=1

                # track vertex adjacent to unknown edge
                if A not in unknown_adj.keys():
                    unknown_adj[A] = []
                unknown_adj[A].append(B)

    new_parent_edges = {}
    while len(unknown_edges)>0:

        if len(new_parent_edges)==0:

            # select (as initial) vertex adjacent to the most o-o edges
            A = max(unknown_edges, key=unknown_edges.get)

        else:

            # select vertex with most new oriented parents
            A = max(new_parent_edges, key=new_parent_edges.get)

            # delete key
            del new_parent_edges[A]

        for B in unknown_adj[A]:

            # update cause
            MAG.updateCause(A, B)

            # update number of new parents into B
            if B not in new_parent_edges.keys():
                new_parent_edges[B] = 0
            new_parent_edges[B]+=1

            # delete vertex from unknown edge adjacent to B
            unknown_adj[B].remove(A)

        # delete vertex from unknown edge
        del unknown_edges[A]

    return MAG


def lemma_3_i(graph, X, Y, parent):
    """
    Check if Lemma 4.4.i holds in Diepen&co (2023).
        :param graph: causalGraph object
        :param X/Y/parent: vertex
    """
    lemma_i = False
    if parent in graph.adjacencies(X) and parent in graph.adjacencies(Y):
        if graph.isDirectedEdge(parent, X) and graph.isDirectedEdge(parent, Y):
            lemma_i = True
        elif graph.isDirectedEdge(parent, Y) and graph.incomingArrowType(parent,X)==graph.unknown and graph.incomingArrowType(X,parent)==graph.head:
            lemma_i = True
        elif graph.isDirectedEdge(parent, X) and graph.incomingArrowType(parent,Y)==graph.unknown and graph.incomingArrowType(Y,parent)==graph.head:
            lemma_i = True
    return lemma_i
