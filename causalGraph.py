
import numpy as np
import copy
from collections import defaultdict


class CausalGraph(object):

    def __init__(self, variables, data=None, corr=None, imp=None, data_size=-1, head='h', tail='t', non_collider=None, unknown=None, default_edge='t', default_score=0, missingness_pairs=[], complete_variables=[], \
        missingness_variables = [], missingness_indicators = [], V2num={}, V2type = defaultdict(lambda: 'O'), M2R={}, R2M={}, context=[], track_directed_paths=False, track_conv=False, oracle=None):
        """
        CausalGraph is a graph representing the causalities between the [variables].
            :param variables: list of possible variables (or vertices)
            :param data: dictionary of numpy arrays of shape (1,N) where N is number of data points
            :param corr: correlation matrix
            :param imp: object gained from mice in R containing multiple imputations
            :param head: string representing arrow head (default='h')
            :param tail: string representing arrow tail (default='t')
            :param non_collider: string representing definite non-collider (default=None)
            :param unknown: string representing unkown edge-mark (default=None)
            :param default_edge:
            :param default_score: score that is used to initialize edges with (default is 0)
            :param missingness_pairs: list of tuples containing missingness pairs (default is empty list)
            :param complete_variables: list of complete variables (default is empty list - when there are no missing values, empty list will do)
            :param missingness_variables: list of variables with missing values (default is empty list)
            :param missingness_indicators: list of missingness indicators (default is empty list)
            :param V2num: dictionary mapping pair of missingness variable and indicator to same number (default is empty dict)
            :param V2type: dictionary mapping complete variable to 'O', missingness variable to 'M' and missingness indicator to 'R' (default maps all to 'O')
            :param M2R: dictionary mapping a missingness variable to its missingness indicator (default is empty dict)
            :param R2M: dictionary mapping a missingness variable to its missingness indicator (default is empty dict)
            :param context: list of vertices
            :param track_directed_paths: boolean, helpful for PC-algorithm
            :param track_conv: boolean wheather to track deletions and orientations in convergence 
            :param oracle: Oracle Causal graph default is None
        """

        self.imp = imp
        self.corr = corr
        self.data_size = data_size
        if not corr is None and data_size == -1:
            raise ValueError('Provide the number of data points.')

        self.data = data
        if data is not None:
            self.data_size = data[[*data][0]].shape[0]

        self.variables = variables
        self.context = context
        self.all_variables = variables + context

        self.default_edge = default_edge

        self.head = head
        self.tail = tail
        self.non_collider = non_collider
        self.unknown = unknown
        self.default_score = default_score

        self.possible_types = [head, tail]
        if not self.unknown is None:
            self.possible_types.append(self.unknown)
        elif default_edge=='u':
            self.unknown = default_edge
            self.possible_types.append(self.unknown)

        # if non_collider is not None:
        #     self.possible_types.append(non_collider)
        #     self.def_non_colliders = True
        # else:
        #     self.def_non_colliders = False

        if not unknown is None:
            self.assign_unknown = True
            self.possible_types.append(unknown)
        else:
            self.assign_unknown = False

        self.missingness_pairs = missingness_pairs
        self.complete_variables = complete_variables
        self.missingness_variables = missingness_variables
        self.missingness_indicators = missingness_indicators
        self.num_missingness_pairs = len(missingness_pairs)
        self.V2num = V2num
        self.V2type = V2type
        self.M2R = M2R
        self.R2M = R2M

        self.track_directed_paths = track_directed_paths

        if self.track_directed_paths:
            self.directed_edges = set()
            self.directed_paths = set()

        # initialization of the complete undirected graph
        self.graph = {}
        for current in self.all_variables:
            self.graph[current] = {}
            for neighbour in self.all_variables:
                # no self-loops
                if current != neighbour:
                    # add edges with default incoming arrow types
                    # set default edge with non-collider on [] (list of all other vertices were it has to be a non-collider with) and score on 0
                    self.graph[current][neighbour] = [default_edge, [], default_score]

        self.track_conv=track_conv
        self.oracle=oracle
        if track_conv:
            if oracle is None:
                raise ValueError('Provide Oracle information.')
            self.conv_results = [[],[],[],[]]
            self.iteration = 0

    def do(self, variable):
        """
        Do-operator: removes all incoming edges of variable.
            :param variable: current variable
        """
        parents = self.parents(variable)
        for parent in parents:
            self.deleteEdge(variable, parent)


    def getData(self, variables):
        """
        Returns the data (length N) of the K variables: NxK array
            :param variables: subset (list) of variables from self.variables
        """
        if self.data is None:
            raise ValueError('Data is None.')

        if len(variables) > 0:

            vdata = self.data[variables[0]]

            if len(variables) > 1:
                for variable in variables[1:]:
                    vdata = np.append(vdata, self.data[variable], 1)
        else:
            vdata = None

        return vdata

    def resetAssignUnknown(self, bool):
        self.assign_unknown = bool

    def incomingArrowType(self, current, neighbour):
        """
        Returns the type of the arrow at [current] coming from [neighbour].
            :param current: variable from variables
            :param neighbour: variable from variables
        """
        return self.graph[current][neighbour][0]

    def isDefiniteNonCollider(self, neighbour1, current, neighbour2):
        """
        Returns a boolean if there is a definite non-collider at current between neighbour1 and neighbour2
            :param neighbour1/current/neighbour2: variable in variables
        """

        if self.incomingArrowType(current, neighbour1) == self.tail or self.incomingArrowType(current, neighbour2) == self.tail:
            return True
        if neighbour2 in self.graph[current][neighbour1][1] and neighbour1 in self.graph[current][neighbour2][1]:
            return True
        else:
            return False

    def score(self, current, neighbour):
        """
        Returns the score of the arrow at [current] coming from [neighbour].
            :param current/neighbour: variable from variables
        """
        return self.graph[current][neighbour][2]

    def updateScore(self, current, neighbour, new_score):
        """
        Updates the score of the arrow at [current] coming from [neighbour].
            :param current/neighbour: variable from variables
        """
        self.graph[current][neighbour][2] = new_score

    def isCollider(self, neighbour1, current, neighbour2):
        if self.incomingArrowType(current, neighbour1) == self.head and self.incomingArrowType(current, neighbour2) == self.head:
            return True
        else:
            return False


    def adjacencies(self, variable):
        """
        Returns the adjacencies of [variable].
            :param variable: variable from variables
        """
        return list(self.graph[variable].keys())

    def parents(self, variable):
        """
        Returns the parents of [variable].
            :param variable: variable from variables
        """
        parents = []
        for neighbour in self.graph[variable].keys():
            if self.isDirectedEdge(neighbour, variable):
                parents.append(neighbour)
        return parents

    def print(self, variable):
        """
        Prints the adjacency variables of [variable] in the graph.
            :param variable: variable from variables
        """
        print(self.graph[variable])

    def isDirectedEdge(self, variable1, variable2):
        """
        Returns a boolean: True if and only if there is a directed edge from
        variable1 to variable2.
        """
        if self.incomingArrowType(variable1, variable2) == self.tail and self.incomingArrowType(variable2, variable1) == self.head:
            return True
        else:
            return False

    def addEdge(self, variable1, variable2, type1, type2, dnc1=[], dnc2=[], score1=None, score2=None):
        """
        Add a (new) edge: (variable1 type1-type2 variable2).
            variable1: variable from variables
            variable2: variable from variables
            type1:   type of edge on side of variable1
            type2:   type of edge on side of variable2
        """

        if score1 is None:
            score1=self.default_score

        if score2 is None:
            score2=self.default_score

        self.isCorrectType(type1)
        self.isCorrectType(type2)
        self.graph[variable1][variable2] = (type1, dnc1, score1)
        self.graph[variable2][variable1] = (type2, dnc2, score2)

        if self.track_directed_paths:
            if type1==self.head and type2==self.tail:
                self.updateDirectedEdgesAndPaths((variable2,variable1))
            elif type1==self.tail and type2==self.head:
                self.updateDirectedEdgesAndPaths((variable1,variable2))


    def updateEdge(self, edge_mark, current, neighbour, rule=None):
        """
        Update an existing edge.
            :param edge_mark:   edge_mark of edge (string); 't':tail or 'h':head
            :param current: variable from variables
            :param neighbour: variable from variables
            :param rule: CDC orientation rule (default=None)
        """
        self.isCorrectType(edge_mark)
        self.isWellDefinedEdge(current, neighbour)
        self.isDefinedEdge(current, neighbour)

        # update the edge in case it is currently unknown
        # otherwise, update it anyway
        if (self.assign_unknown and self.graph[current][neighbour][0]==self.unknown) or not self.assign_unknown:

            # update orientation track matrices for CDC
            if self.graph[current][neighbour][0]!=edge_mark:
                i = self.all_variables.index(neighbour)
                j = self.all_variables.index(current)
                if rule == 'CDC':
                    self.CDC_orientations[i,j] = self.possible_types.index(edge_mark) + 2
                elif rule == 'parent':
                    self.parent_orientations[i,j] = self.possible_types.index(edge_mark) + 2
                elif rule == 'FCI':
                    self.FCI_orientations[i,j] = self.possible_types.index(edge_mark) + 2

            if rule == 'parent':
                self.parent_orientations_all[i,j] = self.possible_types.index(edge_mark) + 2

            # update arrow edge_mark
            self.graph[current][neighbour][0] = edge_mark

            # in case we track the directed edges and paths, we have to update them
            if self.track_directed_paths:
                if self.incomingArrowType(current, neighbour)==self.head and self.incomingArrowType(neighbour, current)==self.tail:
                    self.updateDirectedEdgesAndPaths((neighbour,current))
                elif self.incomingArrowType(current, neighbour)==self.tail and self.incomingArrowType(neighbour, current)==self.head:
                    self.updateDirectedEdgesAndPaths((current,neighbour))

            if self.track_conv:
                self.trackConvergenceOrientation(current, neighbour, edge_mark)


    def updateCause(self, A, B, rule=None):
        """
        Update cause A->B.
            :param A: variable from variables
            :param B: variable from variables
            :param rule: CDC orientation rule (default=None)
        """
        self.updateEdge(self.head, B, A, rule)
        self.updateEdge(self.tail, A, B, rule)


    def deleteEdge(self, variable1, variable2, rule=None):
        """
        Delete an edge from the graph.
            :param variable1: variable from variables
            :param variable2: variable from variables
        """
        if variable1 in self.graph[variable2] and variable2 in self.graph[variable1]:
            del self.graph[variable1][variable2]
            del self.graph[variable2][variable1]

            i = self.all_variables.index(variable1)
            j = self.all_variables.index(variable2)

            # print('rule', rule)

            if rule == 'phase1':
                self.phase1_deletions[i,j] = 1
                self.phase1_deletions[j,i] = 1
            elif rule == 'phase2':
                self.phase2_deletions[i,j] = 1
                self.phase2_deletions[j,i] = 1
            elif rule == 'M0':
                self.M0_deletions[i,j] = 1
                self.M0_deletions[j,i] = 1
            elif rule == 'M1':
                self.M1_deletions[i,j] = 1
                self.M1_deletions[j,i] = 1
            elif rule == 'I0':
                self.I0_deletions[i,j] = 1
                self.I0_deletions[j,i] = 1
            elif rule == 'I1':
                self.I1_deletions[i,j] = 1
                self.I1_deletions[j,i] = 1
            elif rule == 'I2':
                self.I2_deletions[i,j] = 1
                self.I2_deletions[j,i] = 1

        else:
            raise ValueError("No such edge in graph.")


    def directedEdges(self):
        """
        Returns all directed edges.
        """
        if not self.track_directed_paths:
            raise TypeError("Directed edges are not tracked.")
        return list(self.directed_edges)


    def directedPaths(self):
        """
        Returns all directed paths.
        """
        if not self.track_directed_paths:
            raise TypeError("Directed paths are not tracked.")
        return list(self.directed_paths)


    def updateDirectedEdgesAndPaths(self, edge):
        """
        Updates the sets of directed edges and paths.
        Note that tracking works in case of PC algorithm.
        Due to reassigning and causal insufficienty, it does not work for FCI.
            :param edge: ordered tuple (X,Y) so that X->Y is a directed edge in the graph
        """

        if not self.track_directed_paths:
            raise TypeError("Directed edges and paths are not tracked.")

        self.directed_edges.add(edge)

        self.directed_paths.add(edge)

        done = False
        while not done:
            done = True
            new_directed_paths = []
            # loop over all pairs X->Y1 and Y2->Z
            for X, Y1 in self.directed_paths:
                for Y2, Z in self.directed_paths:
                    # in case the end vertex of first path is the vertex of the second paths
                    # we find a new path and we will add it to the path set
                    if Y1 == Y2 and (X,Z) not in self.directed_paths:
                        new_directed_paths.append((X,Z))
                        done = False
            self.directed_paths.update(new_directed_paths)


    def getCDCorientationMatrices(self):
        """
        Intializes adjacency matrices for the orientation rules in the CDC phase.
        """
        len_vars = len(self.all_variables)
        mat = np.zeros((len_vars, len_vars))
        self.CDC_orientations = np.zeros((len_vars, len_vars))
        self.parent_orientations = np.zeros((len_vars, len_vars))
        self.parent_orientations_all = np.zeros((len_vars, len_vars))
        self.FCI_orientations = np.zeros((len_vars, len_vars))

    def getDeletionMatrices(self):
        """
        Intializes adjacency matrices for the deletion rules in the m-FCI algorithm.
        """
        len_vars = len(self.all_variables)
        self.phase1_deletions = np.zeros((len_vars, len_vars))
        self.phase2_deletions = np.zeros((len_vars, len_vars))
        self.M0_deletions = np.zeros((len_vars, len_vars))
        self.M1_deletions = np.zeros((len_vars, len_vars))
        self.I0_deletions = np.zeros((len_vars, len_vars))
        self.I1_deletions = np.zeros((len_vars, len_vars))
        self.I2_deletions = np.zeros((len_vars, len_vars))

    def adjacencyMatrix(self):
        """
        Returns adjacency matrix.
        """
        len_vars = len(self.all_variables)
        adj_mat = np.zeros((len_vars, len_vars))

        for i in range(len_vars):
            for j in range(len_vars):
                # check if adjacent
                if self.all_variables[j] in self.adjacencies(self.all_variables[i]):
                    # check what kind of edge-mark we have here
                    incoming_arrow_type = self.incomingArrowType(self.all_variables[j], self.all_variables[i])
                    if self.unknown is not None and incoming_arrow_type == self.unknown:
                        adj_mat[i,j] = 1 # i*-oj
                    elif incoming_arrow_type == self.head:
                        adj_mat[i,j] = 2 # i*->j
                    elif incoming_arrow_type == self.tail:
                        adj_mat[i,j] = 3 # i*-j
        return adj_mat


    def getRealEdgeMatrix(self):
        """
        Returns matrix containing the edges that are real, i.e.,
        edges with score 0. 
        """
        len_vars = len(self.all_variables)
        real_edges = np.zeros((len_vars, len_vars))
        for i in range(len_vars):
            for j in range(len_vars):
                if self.all_variables[j] in self.adjacencies(self.all_variables[i]):
                    if self.graph[self.all_variables[i]][self.all_variables[j]][2] == 0:
                        incoming_arrow_type = self.incomingArrowType(self.all_variables[j], self.all_variables[i])
                        if self.unknown is not None and incoming_arrow_type == self.unknown:
                            real_edges[i,j] = 1 # i*-oj
                        elif incoming_arrow_type == self.head:
                            real_edges[i,j] = 2 # i*->j
                        elif incoming_arrow_type == self.tail:
                            real_edges[i,j] = 3 # i*-j

        return real_edges

    def getPNancestralMatrix(self):
        """
        Returns postitive and negative ancestral matrix.
        """
        len_vars = len(self.all_variables)
        pam = np.zeros((len_vars, len_vars))
        nam = np.zeros((len_vars, len_vars))
        pos_pam = np.zeros((len_vars, len_vars))

        for i in range(len_vars):
            for j in range(len_vars):
                # check if adjacent
                if self.all_variables[j] in self.adjacencies(self.all_variables[i]):
                    # check what kind of edge-mark we have here
                    incoming_arrow_type1 = self.incomingArrowType(self.all_variables[j], self.all_variables[i])
                    incoming_arrow_type2 = self.incomingArrowType(self.all_variables[i], self.all_variables[j])                    
                    if incoming_arrow_type1 == self.head:
                        nam[j,i] = 1
                        if incoming_arrow_type2 == self.tail:
                            # i -> j
                            pam[i,j] = 1
                        elif incoming_arrow_type2 == self.unknown:
                            pos_pam[i,j] = 1
                    elif incoming_arrow_type1 == self.unknown:
                        if incoming_arrow_type2 == self.unknown:
                            pos_pam[i,j] = 1

        anc_mat = copy.deepcopy(pam)
        for k in range(2,len_vars):
            pam = pam + np.linalg.matrix_power(anc_mat, k)
        pam = np.where(pam!=0, 1, 0)

        # get bigger negative ancestral matrix 
        if True:
            pos_pam = pos_pam + pam 
            anc_mat = copy.deepcopy(pos_pam)
            for k in range(2,len_vars):
                pos_pam = pos_pam + np.linalg.matrix_power(anc_mat, k)
            no_pos_pam = np.where(pos_pam!=0, 0, 1) # no possible positive ancestral matrix
            nam = nam + no_pos_pam
            nam = np.where(nam!=0, 1, 0)

        return pam, nam

    def trackConvergenceOrientation(self, V1, V2, edge_mark):
        if V1 in self.oracle.adjacencies(V2):
            if self.oracle.incomingArrowType(V1, V2) == edge_mark:
                self.conv_results[0][self.iteration] += 1
            else:
                self.conv_results[1][self.iteration] += 1

    def trackConvergenceDeletions(self, V1, V2):
        if not V1 in self.oracle.adjacencies(V2):
            self.conv_results[2][self.iteration] += 1
        else:
            self.conv_results[3][self.iteration] += 1


    def isCorrectType(self, type):
        """
        Checks if the type of edge end is correct.
            :param type:   type of edge (string); 't':tail or 'h':head
        """
        if type not in self.possible_types:
            raise ValueError("Type {} is not known. Choose type from the list {}.".format(type, self.possible_types))


    def isWellDefinedEdge(self, variable1, variable2):
        """
        Checks if the edge is well-defined, in other words,
        check if the edge is (not) saved in the dictionary of both variables.
            :param variable1: variable from variables
            :param variable2: variable from variables
        """
        # check for XOR
        if variable1 not in self.graph[variable2] != variable2 not in self.graph[variable1]:
            raise ValueError("Edge between {} and {} is not well defined.".format(variable1, variable2))


    def isDefinedEdge(self, variable1, variable2):
        """
        Checks if the edge exists.
            :param variable1: variable from variables
            :param variable2: variable from variables
        """
        # check for AND
        if variable1 not in self.graph[variable2] and variable2 not in self.graph[variable1]:
            raise ValueError("Edge between {} and {} does not exist.".format(variable1, variable2))
