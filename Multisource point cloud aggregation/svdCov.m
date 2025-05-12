function [s] = svdCovG(nnIdx, idx, Data, Seed)

nnPt = Data(:,nnIdx);
C = matrixCompute(nnPt,Seed(:,idx));
[U,S,~] = svd(C);
lam=diag(S);
F1=(lam(1)-lam(2))/lam(1);
F2=(lam(2)-lam(3))/lam(1);
F3=(lam(1)-lam(3))/lam(1);
F4=lam(1)/sum(diag(S));
F5=lam(2)/sum(diag(S));
F6=lam(3)/sum(diag(S));
F7=lam(3)/lam(1);
F8=lam(2)/lam(1);
F9=( F4 .* F5 .* F6 ).^(1/3);
F10=-(F4.*log(F4) + F5.*log(F5) +F6.*log(F6) );

s =[F1;F2;F3;F4;F5;F6;F7;F8;F9;F10];
% s = diag(S)/sum(diag(S));
% n = sign(dot(U(:,3),-Seed(:,idx)))*U(:,3);
end

% [s,n] = cellfun(@(x,y)svdCov(x,y,tarData,tarSeed),tarIdx,idx,'uni',false);
