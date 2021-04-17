function [c_hat, gamma_hat, level_hat, cv] = EELsvmtune(x,y,all_D, all_gamma,all_level, nk,  kernel)
%EELsvmtune - This function implements the grid-search tuning of the  Extreme Empirical Loss
%SVM via k-fold cross-validation. 

% Inputs: 
% x_train: Training data (samplesXfeatures)
% y_train: Training labels (samplesX1) - should be +1/-1
% all_D: Range of values to test for the D (penalty) parameter
% all_gamma: Range of values to test for the kernel parameter
% all_level: Range of values to test for the alpha parameter 
% nk: Number of folds for the k-fold cross validation
% k_feature: column index for the single perturbation
% kernel: Kernel Type 'l': Linear, 'g':RBF

% Output: 
% c_hat: Selected value for the penalty parameter   
% gamma_hat: Selected value for the kernel parameter  
% level_hat: Selected value for the perturbation parameter
% cv: Matrix with all the hyperparameter combinations tested and the
% corresponding accuracy




%Define the hyperparameter combinations
[aa bb cc] = meshgrid(all_D, all_gamma,all_level);
all_test = [aa(:) bb(:) cc(:)];

%Run k fold validation
indices = crossvalind('Kfold',y,nk);
cp = classperf(y);
cv = ones(size(all_test,1),1);
for z = 1:length(cv)
    c = all_test(z,1);
    gamma = all_test(z,2);
    level = all_test(z,3);
e = ones(1,nk);
for i = 1:nk
    test = (indices == i); 
    train = ~test;
 e(i) =  (1- EELsvmtrain(x(find(train>0),:),y(find(train>0)),c*sum(train), level,  kernel, gamma, x(find(test>0),:),y(find(test>0))))*sum(test);    
end
cv(z) = 1-(sum(e)/length(y));
end
cv = horzcat(all_test, cv);
cv = sortrows(cv, 4, 'descend');

%Select the best hyperparameter values
  c_hat = cv(1,1);
  gamma_hat  = cv(1,2);
  level_hat  = cv(1,3);
end