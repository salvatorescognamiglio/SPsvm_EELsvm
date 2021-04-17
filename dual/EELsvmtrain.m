function [acc,model, y_hat, perf] =EELsvmtrain(x_train,y_train,D,level,kernel, gamma_k,x_test,y_test)
%EELsvmtrain - This function implements the Extreme Empirical Loss (EEL) SVM formulation (dual) for binary classification.
% Inputs: 
% x_train: Training data (samplesXfeatures)
% y_train: Training labels (samplesX1) - should be +1/-1
% D: Hyperparameters defining the penalty
% level: Hyperparameter defining the perturbation
% kernel: Kernel Type 'l': Linear, 'g':RBF
% gamma_k: Kernel Parameters: RBF width for kernel='g'
%           set gamma_k = 1 for kernel='l'
% x_test: Testing data (test_samplesXfeatures)
% y_test: Testing labels (test_samplesX1) - should be +1/-1

% Output: 
% acc: model accuracy on the testing set 
% model: structure array with some model's informations
%      (solution, computational time)
% y_hat: predictions on x_test
% perf: confusion matrix predictions Vs actual values 

n = size(x_train,1);
n_feature = size(x_train,2);

%define the objects  for optimisation
D1 = D/(n*(1-level));
Q = (y_train*y_train').*ker(x_train,x_train',kernel, gamma_k);
Z = zeros(n,n);
H = [Q, Z, Z;
     Z, Z, Z;
     Z, Z, Z];
  f = -[ones(1,n), zeros(1,2*n)]';
  AEQ = [eye(n),eye(n),eye(n);
    y_train', zeros(1,n*2);
    ones(1,2*n), zeros(1,n)];
beq = [D1*ones(n,1);0; D];
 lb = zeros(3*n,1);
 ub = [];
 A = [];
 b = [];
 
  %solve optimisation
 tic;
 sol_EEL = quadprog(H,f,A,b,AEQ,beq,lb,ub);
 times = toc;
 
 model.sol =  sol_EEL;
 model.time  =  times;
  
   %compute predictions 
 alpha_EEL = sol_EEL(1:n); % alpha of the paper
beta_EEL = sol_EEL((n+1):(2*n)); % beta of the paper
gamma_EEL = sol_EEL((2*n+1):(3*n)); % gamma of the paper
index_S3 = find(round(alpha_EEL.*beta_EEL.*gamma_EEL, 4) >0); % S3 set
index_S4 = find( (round(alpha_EEL.*beta_EEL,4)>0) & (round(gamma_EEL,4)== 0) );% S4 set
  
if (length(index_S3)>= length(index_S4))
    b = (sum(y_train(index_S3))-sum((alpha_EEL.*y_train).*ker(x_train, x_train(index_S4,:)', kernel,gamma_k), 'all'))/length(index_S3) ;
else
    b = (sum(y_train(index_S4))-sum((alpha_EEL.*y_train).*ker(x_train, x_train(index_S4,:)', kernel,gamma_k), 'all'))/length(index_S4);
end 
pred = (alpha_EEL.*y_train)'*ker(x_train, x_test', kernel,gamma_k)+ b;
y_hat = sign(pred);

model.pred = y_hat;
perf = confusionmat(y_test, y_hat);
acc = (perf(1,1) + perf(2,2))/length(y_test);
 
 
 
 
end