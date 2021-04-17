% Sample script for running Single Perturbation (SP) Support Vector Machine and
% the  Extreme Empirical Loss (EEL) Support Vector Machine. 
%
% Reference Paper: Vali Asimit, Ioannis Kyriakou, Simone Santoni, Salvatore Scognamiglio and Rui Zhu
% "Robust Classification via Support Vector Machines". 



clear all


% Load Read Data
name_dataset = 'statlog';
data = readtable(strjoin([  name_dataset,  ".csv"], ""));
data = table2array(data);
d = size(data, 2)-1;
n =size(data, 1);

% Separate training and test data (70:30 split)
n_test = 90  %n*1/3
n_train = n-n_test; 


% Normalize labels
data((data(:,14) == 2),14) = -1;


% Define x and y
x = data(:,1:13);
y = data(:,14);
tabulate(y)

% Normalize input
x = normalize(x, 'range', [-1,1]);

% Define training and test samples
rng(1234)
testing_index = randsample(n, n_test);
training_index =setdiff([1:n], testing_index);
x_train = x(training_index,:);
x_test = x(testing_index,:);
y_train = y(training_index);
y_test = y(testing_index);


%Run SP SVM 
kernel = 'g';
all_std = std(x);
% Define feature for perturbation
[max_std, index_max] = max(all_std);

% Define hyperparameter values
c_SP = 2^3; level_SP = 0.52; gamma_SP = 2^-2;

[accuracy_SP]  =  SPsvmtrain(x_train,y_train, c_SP,index_max,level_SP, kernel,gamma_SP,x_test,y_test );

%Run EEL SVM 
% Define hyperparameter values
c_EEL = 2^3*n_train; level_EEL = 0.05; gamma_EEL =  2^-2;

[accuracy_EEL]  = EELsvmtrain(x_train,y_train,(c_EEL),level_EEL,kernel, gamma_EEL,x_test, y_test);


disp(strjoin(['Accuracy of the SP SVM is ', round(accuracy_SP*100,2),  "%"], ""))
disp(strjoin(['Accuracy of the EEL SVM is ', round(accuracy_EEL*100,2),  "%"], ""))



