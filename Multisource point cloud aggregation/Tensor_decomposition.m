function T = Tensor_decomposition(F)

displog('Performing feature extraction and building feature tensor model');

T0 = tensor(F);


%% Unfolding 3rd-order tensor into mode-(1,2,3) and convert it to matrix
T01tm = tenmat(T0,1); T01 = double(T01tm);
T02tm = tenmat(T0,2); T02 = double(T02tm);
T03tm = tenmat(T0,3); T03 = double(T03tm);

%%% Perform initial SVD in the mode-1
displog('Performing initial SVD in the mode-1');
r1 = 1; % temporal/values rank
[T01_U,T01_S,T01_V] = svds(T01,r1);
T01_hat = (T01_U*T01_S*T01_V'); % norm(T01-T01_hat)
T = tensor(tenmat(T01_hat,T01tm.rdims,T01tm.cdims,T01tm.tsize));

% 
% F0_hat1_f1 = T0_hat1(:,:,1);
% F0_hat1_f2 = T0_hat1(:,:,2);
% F0_hat1_f3 = T0_hat1(:,:,3);
% F0_hat1_f4 = T0_hat1(:,:,4);
% F0_hat1_f5 = T0_hat1(:,:,5);
% F0_hat1_f6 = T0_hat1(:,:,6);
% F0_hat1_f7 = T0_hat1(:,:,7);
% F0_hat1_f8 = T0_hat1(:,:,8);

  

end
