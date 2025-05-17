 function [GeoF, PhyF] = feature_compute( P_Data, P_Seed, gridStep)
% Data=XYZ;
% Seed=XYZ;
% P_Data=P;
% P_Seed=P;
% gridStep=1;



PData=P_Data(:,1:7);
PhyData=PData';
PSeed=P_Seed(:,1:7);
PhySeed=PSeed';



radii = (1:0.5:2)*gridStep;

K = length(radii);
NS = createns(PhyData');
xyzIdx = rangesearch(NS,PhySeed',radii(1));
idxSz = cellfun(@length,xyzIdx,'uni',true);
M = length(idxSz);
idx = num2cell((1:M)');

p = cellfun(@(x,y)svdCovP(x,y,PhyData,PhySeed),xyzIdx,idx,'uni',false);
p = cell2mat(p);

for k = 2:K
    
    xyzIdx = rangesearch(NS,PhySeed',radii(k));
    
    pk = cellfun(@(x,y)svdCovP(x,y,PhyData,PhySeed),xyzIdx,idx,'uni',false);
    p = [p cell2mat(pk)];
    k
end

p1=p(:,1);
p2=p(:,2);
p3=p(:,3);


P1=reshape(p1,10,[]);
P2=reshape(p2,10,[]);
P3=reshape(p3,10,[]);


P1=P1';
P2=P2';
P3=P3';


PhyF=zeros(length(PhySeed),K,10);
PhyF(:,1,:)=P1;
PhyF(:,2,:)=P2;
PhyF(:,3,:)=P3;


% s = s';
 end

