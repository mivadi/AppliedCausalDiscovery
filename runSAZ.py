import pandas as pd
import numpy as np
from FCI import FCI
from statistics import median


def experiments(data_size, exp_descr, path_cor, folder_boot_cor, num_boots, alpha=0.05, context = ['age', 'gender'], var_types=None, binary=None, effective_sample_size=True):

    print(exp_descr)

    # experiments with background information
    JCI='12'
    bkg_info = None 
    tiers = {'age':1, 'gender':1, 'creat':2, 'eGFR':2, 'map_scr':2, 'bmi':2, 'chronic_lung_disease':2, 'prev_cva':2, 'prev_card_surg':2, 'MCA':3, 'MCA_ok1':3, 'delirium':4, 'mortality':5, 'mech_ven':4, 'post_cva':4, 'rethorac_30':4, 'urg':2, 'ECA':2, 'multi_ves_dis':2, 'diabetes':2, 'nyha':2}
    complete_value_indicator = None

    corr = pd.read_csv(path_cor)
    variables = list(corr.columns)
    for context_var in context:
        variables.remove(context_var)
    V = corr.shape[0]

    av_skeleton = np.zeros((V,V))
    av_pam = np.zeros((V,V))
    av_nam = np.zeros((V,V))


    print("experiments with real world bootstrapped data")
    for i in range(1,num_boots+1):
        print(i)

        path = folder_boot_cor + "/cov_matrix_boot"+str(i)+".csv"
        corr = pd.read_csv(path)
        imp = None
        indTest = "Fisher"

        if effective_sample_size:
            complete_value_indicator = pd.read_csv(folder_boot_cor + "/complete_value_indicator"+str(i)+".csv")
        

        pag = FCI(variables, corr=corr, imp=imp, data_size=data_size, indTest=indTest, var_types=var_types, binary=binary, alpha=alpha, JCI=JCI, context=context, bkg_info=bkg_info, tiers=tiers, conservative_FCI=True, aug_PC=True, complete_value_indicator=complete_value_indicator)
        
        av_skeleton = av_skeleton + np.where(pag.adjacencyMatrix()>0, 1, 0)
        pam, nam = pag.getPNancestralMatrix()
        av_pam = av_pam + pam
        av_nam = av_nam + nam

    av_skeleton = av_skeleton / num_boots
    av_skeleton = pd.DataFrame(av_skeleton, index=pag.all_variables, columns=pag.all_variables)
    av_skeleton.to_csv("../../SAZ/my_results/FCI/" + exp_descr + "/av_skeleton.csv")

    av_pam = av_pam / num_boots
    av_pam = pd.DataFrame(av_pam, index=pag.all_variables, columns=pag.all_variables)
    av_pam.to_csv("../../SAZ/my_results/FCI/"+ exp_descr +"/av_pam.csv")

    av_nam = av_nam / num_boots
    av_nam = pd.DataFrame(av_nam, index=pag.all_variables, columns=pag.all_variables)
    av_nam.to_csv("../../SAZ/my_results/FCI/"+ exp_descr +"/av_nam.csv")


N = 142

num_boots=1000

dir_output = ''
dir_to_correlation_matrix_of_original_data = 'corr_matrix.csv'
dir_to_correlation_matrices_of_bootstrapped_data = ''

# experiments2(N, dir_output, dir_to_correlation_matrix_of_original_data, dir_to_correlation_matrices_of_bootstrapped_data, num_boots, context = ['age', 'gender'], effective_sample_size = True)

