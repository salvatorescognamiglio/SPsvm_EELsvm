
function [acc,model, y_hat, perf] = SPsvmtrain(x_train,y_train,cost,k_feature,level, kernel,gamma,x_test,y_test)
%SPsvmtrain - This function implements the Single Perturbation SVM formulation (dual) for binary classification.
% Inputs: 
% x_train: Training data (samplesXfeatures)
% y_train: Training labels (samplesX1) - should be +1/-1
% cost: Hyperparameters defining the penalty
% k_feature: column index for the single perturbation
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
% perf: confusion matrix predictions VS actual values  


n = size(x_train,1);
n_feature = size(x_train,2);

%introduce the perturbation 
a_k = norminv(level)*std(x_train(:,k_feature));
aux = zeros(n,n_feature);
aux(:,k_feature) = a_k*ones(n,1);
lower = x_train - aux;
upper = x_train + aux;

%define the objects  for optimisation
T_00 = ker(x_train,x_train',kernel, gamma);
T_01 = ker(x_train,lower',kernel, gamma);
T_02 = ker(x_train,upper',kernel, gamma);
T_10 = ker(lower,x_train',kernel, gamma);
T_11 = ker(lower,lower',kernel, gamma);
T_12 = ker(lower,upper',kernel, gamma);
T_20 = ker(upper,x_train',kernel, gamma);
T_21 = ker(upper,lower',kernel, gamma);
T_22 = ker(upper,upper',kernel, gamma);
T = [T_00, T_01, T_02;
     T_10, T_11, T_12;
     T_20, T_21, T_22];
 full_y = [y_train; y_train; y_train];
 full_x = [x_train; lower; upper];
 Q = T.*(full_y*full_y');
 f = -ones(3*n,1);
 A = [eye(n),eye(n),eye(n)];
 B = cost*ones(n,1);
 aeq = full_y';
 lb = zeros(n*3,1);
 ub = [];
 beq = 0;
 
 %solve optimisation
tic
  sol_SP = quadprog(Q,f,A,B, aeq,beq, lb, ub);
  times = toc;
  full_x;
  full_y;
 
   model.sol =  sol_SP;
 model.time  =  times;
 

  %compute predictions 
   alpha_SP = sol_SP(1:n); % alpha of the paper
beta_SP = sol_SP((n+1):(2*n)); % beta of the paper
gamma_SP  = sol_SP((2*n+1):(3*n)); % gamma of the paper
delta_SP = cost - alpha_SP - beta_SP - gamma_SP;
S = find(round(alpha_SP.*delta_SP, 4) >0); % S set
S1 = find(round(beta_SP.*delta_SP, 4) >0); % S1 set
S2 = find(round(gamma_SP.*delta_SP, 4) >0); % S2 set
if (  length(S) == max([length(S), length(S1), length(S2)]))
        sol = alpha_SP; 
        S_star = S;
        x_train = full_x(1:n,:);
elseif ( length(S1) == max([length(S), length(S1), length(S2)]))
        sol = beta_SP;
        S_star = S1;
         x_train = full_x((n+1):(2*n),:);
else
        sol = gamma_SP;
        S_star = S2;
           x_train = full_x((2*n+1):(3*n),:);
end 

b= (sum(y_train(S_star))-sum((sol_SP.*full_y)'*ker(full_x,x_train(S_star,:)', kernel, gamma)))/length(S_star); 
pred = (sol_SP.*full_y)'*ker(full_x, x_test',kernel, gamma)+ b;
y_hat = sign(pred);
perf = confusionmat(y_test, y_hat);
acc = (perf(1,1) + perf(2,2))/length(y_test);
 model.full_x = full_x;
 model.full_y = full_y;
  
end