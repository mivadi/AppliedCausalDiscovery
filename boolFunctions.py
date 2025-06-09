from causalGraph import CausalGraph


def isOpenRecursive(graph, current, previous, final_var, visited, ancestors_Z, Z, acyclic, SCCs, pos_ancestors_Z=None):
    """
    Recursive function based on basebal search algorithm that returns a boolean
    indicating if there is an open path found given seperating set Z.
        :param graph: causalGraph object
        :param current, previous: current and previous vertex on a path.
        :param final: final vertex on the path
        :param visited: all visited vertices on the path
        :param ancestors_Z: list of ancestors for vertices Z
        :param Z: possible separating set (set of vertices)
        :param acylic: true if acyclic graph
        :param SCCs: dictionary that contains all strongly connected component info
    """
    # select new next vertices on the path
    nexts = [V for V in graph.adjacencies(current) if not V in visited] # this can faster
    m_sep = 'blocked'
    i = 0
    # once we find an open path, we know that we cannot block the vertices
    while m_sep == 'blocked' and i < len(nexts):
        # check if the current triple form a collider
        collider = graph.incomingArrowType(current, previous)==graph.head and graph.incomingArrowType(current, nexts[i])==graph.head
        non_collider = not collider and not (graph.incomingArrowType(current, previous)==graph.unknown or graph.incomingArrowType(current, nexts[i])==graph.unknown)
        # path is not (yet) blocked when current is a collider and current is an ancestor of Z
        if collider and current in ancestors_Z:
            m_sep = 'open'
        elif collider and not pos_ancestors_Z is None and current in pos_ancestors_Z:
            m_sep = 'uncertain'
        # path is not (yet) blocked when current is a non-collider and current is not in Z
        elif not collider and not current in Z:
            m_sep = 'open'
        elif not collider and not non_collider:
            m_sep = 'uncertain'
        # for a cyclic graph we also check if the path is blocked according to sigma-separation
        elif non_collider and not acyclic:
            if not ( graph.isDirectedEdge(current, nexts[i]) and nexts[i] not in SCCs[current] ):
                if not ( graph.isDirectedEdge(current, previous) and previous not in SCCs[current] ):
                    print("We have to test if it also works for cyclic graphs.")
                    m_sep = 'open'
        # if the path is open, we continue to the next triple on the path and test
        # again if the path is open for the next triple
        if m_sep and nexts[i] != final_var:
            m_sep = isOpenRecursive(graph, nexts[i], current, final_var, visited+[current], ancestors_Z, Z, acyclic, SCCs)

        i = i+1

    return m_sep



def isSeparated(graph, X, Y, Z, ancestors, pos_ancestors=None, acyclic=True, SCCs=None):
    """
    Checks if X and Y are m-separated or sigma-separated by Z.
        :param graph: causalGraph object
        :param X, Y: starting and end vertex
        :param Z: possible separating set (set of vertices)
        :param ancestors: dictionary that contains all list of ancestors for each vertex
        :param acylic: true if acyclic graph
        :param SCCs: dictionary that contains all strongly connected component info
    """

    # select the ancestors for Z
    ancestors_Z = []
    for variable in Z:
        ancestors_Z = ancestors_Z + ancestors[variable]
    ancestors_Z = list(set(ancestors_Z))

    if not pos_ancestors is None:
        pos_ancestors_Z = []
        for variable in Z:
            pos_ancestors_Z = pos_ancestors_Z + pos_ancestors[variable]
        pos_ancestors_Z = list(set(pos_ancestors_Z))
    else:
        pos_ancestors_Z = None

    # select the adjacencies of X as 'current' vertices
    currents = graph.adjacencies(X)
    m_sep = 'blocked'
    i = 0

    # while we do not find an open path, keep trying to find one
    while m_sep == 'blocked' and i < len(currents):
        m_sep = isOpenRecursive(graph, currents[i], X, Y, [X], ancestors_Z, Z, acyclic, SCCs, pos_ancestors_Z=pos_ancestors_Z)
        i += 1

    # X and Y are separated by Z if there is no open path
    # if pos_ancestors is None:
    #     return m_sep == 'blocked'
    # else:
    return m_sep



def isOpenRecursive1(graph, current, previous, final_var, visited, ancestors_Z, Z, acyclic, SCCs):
    """
    Recursive function based on basebal search algorithm that returns a boolean
    indicating if there is an open path found given seperating set Z.
        :param graph: causalGraph object
        :param current, previous: current and previous vertex on a path.
        :param final: final vertex on the path
        :param visited: all visited vertices on the path
        :param ancestors_Z: list of ancestors for vertices Z
        :param Z: possible separating set (set of vertices)
        :param acylic: true if acyclic graph
        :param SCCs: dictionary that contains all strongly connected component info
    """
    # select new next vertices on the path
    nexts = [V for V in graph.adjacencies(current) if not V in visited] # this can faster

    open_path = False
    i = 0
    # once we find an open path, we know that we cannot block the vertices
    while not open_path and i < len(nexts):
        # check if the current triple form a collider
        collider = graph.incomingArrowType(current, previous)==graph.head and graph.incomingArrowType(current, nexts[i])==graph.head

        # print(collider, previous, current, nexts[i])


        # path is not (yet) blocked when current is a collider and current is an ancestor of Z
        if collider and current in ancestors_Z:
            open_path = True
        # path is not (yet) blocked when current is a non-collider and current is not in Z
        elif not collider and not current in Z:
            open_path = True
        # for a cyclic graph we also check if the path is blocked according to sigma-separation
        elif not collider and not acyclic:
            if not ( graph.isDirectedEdge(current, nexts[i]) and nexts[i] not in SCCs[current] ):
                if not ( graph.isDirectedEdge(current, previous) and previous not in SCCs[current] ):
                    print("We have to test if it also works for cyclic graphs.")
                    open_path = True
        # if the path is open, we continue to the next triple on the path and test
        # again if the path is open for the next triple
        if open_path and nexts[i] != final_var:
            open_path = isOpenRecursive1(graph, nexts[i], current, final_var, visited+[current], ancestors_Z, Z, acyclic, SCCs)

        i = i+1

    return open_path


def isSeparated1(graph, X, Y, Z, ancestors, pos_ancestors=None, acyclic=True, SCCs=None):
    """
    Checks if X and Y are m-separated or sigma-separated by Z.
        :param graph: causalGraph object
        :param X, Y: starting and end vertex
        :param Z: possible separating set (set of vertices)
        :param ancestors: dictionary that contains all list of ancestors for each vertex
        :param acylic: true if acyclic graph
        :param SCCs: dictionary that contains all strongly connected component info
    """

    # print(X,Y,Z)

    # select the ancestors for Z
    ancestors_Z = []
    for variable in Z:
        ancestors_Z = ancestors_Z + ancestors[variable]
    ancestors_Z = list(set(ancestors_Z))

    # select the adjacencies of X as 'current' vertices
    currents = graph.adjacencies(X)
    open_path = False
    i = 0

    # while we do not find an open path, keep trying to find one
    while not open_path and i < len(currents):
        if currents[i]==Y:
            open_path = True
        else:
            open_path = isOpenRecursive1(graph, currents[i], X, Y, [X], ancestors_Z, Z, acyclic, SCCs)
        i += 1

    # print(not open_path)
    # raise ValueError('STOP')

    # X and Y are separated by Z if there is no open path
    return not open_path



