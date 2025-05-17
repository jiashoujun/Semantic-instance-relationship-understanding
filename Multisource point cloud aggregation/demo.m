clc;clear;

% load('Corr_VL.mat');
% load('data1/Corr_SL.mat');

allVision = load('vision.txt');    
allSpectrum = load('data/spectrum.txt'); 
allLaser = load('data/laser.txt'); 

% find the correspondences
laserIdx=Corr_SL(:,1);
spectrumIdx=Corr_SL(:,2);

% overlapping points
XYZ=allSpectrum(spectrumIdx,1:3);


%% extract the physical information
I=allLaser(laserIdx,7)/200.0;
RGB=allLaser(laserIdx,4:6)/255.0;
LMH=allSpectrum(spectrumIdx,4:6)/255.0;

P=[I RGB LMH];


%% extract high-dimensional geometric and physical features
[GeoFeature, PhyFeature]  = feature_compute(XYZ, XYZ, P, P, 0.1);

R=RGB(:,1);
G=RGB(:,2);
NIR=LMH(:,3);

NDVI=(NIR-R)./(NIR+R);
NDWI=(G-NIR)./(G+NIR);
 

% construct high-dimensional tensor model
General_PointCloud=zeros(length(laserIdx),3,30);
General_PointCloud(:,1,1:10)=[XYZ P];
General_PointCloud(:,2,1:10)=[XYZ P];
General_PointCloud(:,3,1:10)=[XYZ P];
General_PointCloud(:,:,11:20)=GeoFeature;
General_PointCloud(:,:,21:28)=PhyFeature;
General_PointCloud(:,:,29)=[NDVI NDVI NDVI];
General_PointCloud(:,:,30)=[NDWI NDWI NDWI];


%%

