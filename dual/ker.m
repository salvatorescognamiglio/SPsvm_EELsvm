function ker=Ker(x1,x2,kernel, gamma_k)

% Ker - This function computes linear and RBF kernels
% Inputs:
% kernel: Kernel type ('l':linear, 'g':RBF
% x1: First sample of x for computing kernel (samples1Xfeatures)
% x2: Second Sample of x for computing kernel  (samples2Xfeatures)
% gamma_k: Kernel Parameters: RBF width for kernel='g'
%           set gamma_k = 1 for kernel='l' 
% Inputs:
% ker: Matrix containing the kernel transformation of the pairs (x_i, x_j) 
%       for x_i in x1 and x_j in x2 (samples1Xsample2).



n1 = size(x1,1);
n2 = size(x2,2);
ker = zeros(n1,n2);
if kernel=='g'    
for i=1:n1
    for j=1:n2
        ker(i,j) = exp(- gamma_k* norm(x1(i,:)-x2(:,j)')); %RBF Kernel
    end 
end    
elseif kernel=='l'
    for i=1:n1
    for j=1:n2
         ker(i,j) = x1(i,:)*x2(:,j); %Linear Kernel
    end 
end 

end