function [p] = svdCovP(nnIdx, idx, Data, Seed)

nnPt = Data(:,nnIdx);
C = matrixCompute(nnPt,Seed(:,idx));
[U,S,~] = svd(C);
s_lam=diag(S);
lam=s_lam/sum(diag(S));

F1=lam(1);
F2=lam(2);
F3=lam(3);
F4=lam(4);
F5=lam(5);
F6=lam(6);
F7=lam(7);
F8=((lam(1)-lam(7))/lam(1)+(lam(2)-lam(7))/lam(2)+(lam(3)-lam(7))/lam(3)+(lam(4)-lam(7))/lam(4)+(lam(5)-lam(7))/lam(5)+(lam(6)-lam(7))/lam(6))/7.0;
F9=( F1 .* F2 .* F3 .* F4 .* F5 .* F6 .* F7).^(1/7);
F10=-( F1.*log(F1) + F2.*log(F2) +F3.*log(F3) + F4.*log(F4) +F5.*log(F5) + F6.*log(F6) +F7.*log(F7) )/7.0;

p =[F1;F2;F3;F4;F5;F6;F7;F8;F9;F10];
% s = diag(S)/sum(diag(S));
% n = sign(dot(U(:,3),-Seed(:,idx)))*U(:,3);
 end

% [s,n] = cellfun(@(x,y)svdCov(x,y,tarData,tarSeed),tarIdx,idx,'uni',false);
