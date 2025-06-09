import statistics
from causalGraph import CausalGraph
from kernelFunctions import *
from Fishers_z_transform import condIndFisherZ
from KCIT import independenceTest
import pandas as pd
import math
from boolFunctions import isSeparated1

class IndependenceTest(object):

    def __init__(self, alpha=0.05, indTest='Fisher', oracle=None, oracle_graph=None, ancestors=None, var_types=None, test='rf_pillai', binary=None, complete_value_indicator=None):

        self.alpha = alpha
        self.oracle_dic = oracle
        self.oracle_graph = oracle_graph
        self.effective_sample_size_ind = False
        self.ancestors = ancestors
        if not complete_value_indicator is None:
            self.complete_value_indicator = complete_value_indicator
            self.effective_sample_size_ind = True

        self.oracle_testing = False
        
        if indTest == 'Fisher':
            self.execute = self.condIndFisherZ_test
        
        elif indTest == 'kernel':
            self.execute = self.KCIT_test

        elif indTest == 'oracle' and not self.oracle_dic is None:
            self.execute = self.oracle_test_dic

        elif indTest == 'oracle' and not self.oracle_graph is None and not ancestors is None:
            self.execute = self.oracle_test
            self.oracle_testing = True

        else:
            raise ValueError('Choose setting for the independence test.')

        

    def oracle_test_dic(self, graph, X, Y, condition=[]):
        """
        Decide on independence making use of oracle_dic information.
            :param graph: CausalGraph object
            :param X, Y: variables
            :param condition: list of variables (default is empty list [])
        """
        independent = False
        if X in oracle_dic.keys() and Y in oracle_dic[X].keys():
            for sep_set in oracle_dic[X][Y]:
                if set(sep_set).issubset(set(condition)):
                    independent =  True
                    break
        elif Y in oracle_dic.keys() and X in oracle_dic[Y].keys():
            for sep_set in oracle_dic[Y][X]:
                if set(sep_set).issubset(set(condition)):
                    independent =  True
                    break

    def oracle_test(self, graph, X, Y, condition=[]):
        """
        Decide on independence making use of oracle information.
            :param graph: CausalGraph object
            :param X, Y: variables
            :param condition: list of variables (default is empty list [])
        """

        ind = isSeparated1(self.oracle_graph, X, Y, condition, self.ancestors)

        return ind


    def computeEffectiveSampleSize(self, X, Y, condition):
        """
        Returns effective sample size.
            :param X: variable
            :param Y: variable
            :param condition: list of variables (default is empty list [] )
        """

        # select the complete value indicators of the current subset
        subset = self.complete_value_indicator[[X] + [Y] + list(condition)]

        # number of rows that do not have a missing value in the
        # current set of variables we are observing
        effective_sample_size = pd.Series((subset != 0).all(axis=1)).sum()

        return effective_sample_size


    def condIndFisherZ_test(self, graph, X, Y, condition=[]):
        """
        Returns boolean if X and Y are independent given condition based on Fisher-Z test.
            :param graph: CausalGraph object
            :param X: variable
            :param Y: variable
            :param condition: list of variables (default is empty list [] )
        """

        if self.effective_sample_size_ind:
            # compute p-value with effective sample size (for example; when there are missing values)
            effective_sample_size = self.computeEffectiveSampleSize(X,Y,condition)
            p_val, _ = condIndFisherZ(X, Y, condition, graph.corr, effective_sample_size)
        else:
            # compute p-value with total sample size
            p_val, _ = condIndFisherZ(X, Y, condition, graph.corr, graph.data_size)

        # if the p-value is smaller than a value alpha, we will reject the null-hypothesis
        # and except the alternative hypothesis, i.e. the variables are dependent
        # otherwise we cannot reject the null-hypothesis
        if p_val > self.alpha:
            independent = True
        else:
            independent = False

        return independent


    def KCIT_test(self, graph, X, Y, condition=[], nr_samples=50000):
        """
        Returns boolean if X and Y are independent given condition based on:
        Kernel-based Conditional Independence Test and Application in Causal Discovery - Zhang et al. (2011)
            :param graph: CausalGraph object
            :param X: variable
            :param Y: variable
            :param condition: list of variables (default is empty list [] )
            :param nr_samples: number of samples to compute the statistics
        """
        return independenceTest(graph, X, Y, condition=condition, alpha=self.alpha, nr_samples=nr_samples)

