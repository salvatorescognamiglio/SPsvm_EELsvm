

This folder contains the code for MATLAB implementation of the dual formulation of the Single Perturbation (SP) Support Vector Machine and the Extreme Empirical Loss (EEL) Support Vector Machine.

Reference Paper: Vali Asimit, Ioannis Kyriakou, Simone Santoni, Salvatore Scognamiglio and Rui Zhu. " Robust Classification via Support Vector Machines”. Working paper.

Please see sample_script.m for a sample implementation of the Single Perturbation (SP) SVM and the Extreme Empirical Loss (EEL) SVM on the sample dataset in statlog.csv.

The following are the files available:

    SPsvmtrain.m - This function implements the SP SVM formulation (dual) for binary classification.
    EELsvmtrain.m - This function implements the EEL SVM formulation (dual) for binary classification.
    EELsvmtune.m - This function can be used to tuning parameters of EEL-SVM with RBF kernel. It uses grid search for tuning.
    SPsvmtune.m - This function can be used to tuning parameters of SP-SVM with RBF kernel. It uses grid search for tuning.
    ker.m - This function computes linear or Radial Basis Function (RBF) kernel.
    statlog.csv - Sample dataset
    sample_script.m Sample code for running the SP and EEL SVMs with RBF kernels on statlog dataset.


