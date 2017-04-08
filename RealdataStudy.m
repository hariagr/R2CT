%% Real dataset reconstruction
% Please download the dataset APS02 from the following weblink
% ftp://ftp.xray.aps.anl.gov/pub/tomo-databank/Lorentz/aps/noisy/
%
% This script is mainly written to do reconstruction for APS02 dataset.
% 
% We are using ASTRA Toolbox.

clc;clear;close all;

% job setup
JOBID = getenv('PBS_JOBID');
if strcmp(JOBID,'')
    JOBID = datestr(now,'yymmddHHMMSS');
end
fprintf(1,'Job ID: %s\n',JOBID);
JOBNAME = getenv('PBS_JOBNAME');
if strcmp(JOBNAME,'')
    JOBNAME = mfilename;
end
fprintf(1,'Job name: %s\n',JOBNAME);

CUDADEV=getenv('CUDADEV');
GPUDev = gpuDevice;
if strcmp('CUDADEV','')
    GPU_ID = str2num(CUDADEV);
else
    GPU_ID = 0;
end
fprintf(1,'GPU ID: %i\n',GPU_ID);

filename = 'APS_13_BM.h5';

% Extract data from file
slice = 300;
F = double(squeeze(h5read(filename','/exchange/data_white',[1,slice,1],[960,1,12]))); % Flat-field measurements
Y = double(squeeze(h5read(filename','/exchange/data',[1,slice,1],[960,1,900])));      % Measurements with object
theta = double(h5read(filename,'/exchange/theta'))/180*pi;                            % Projection angles

% center shift correction 
Y = [Y(7:end,:) ; repmat(Y(end,:),6,1)];

% Reconstruction parameters
n = 768;            % Grid size (n x n)
r = size(Y,1);      % Number of detector elements
p = size(Y,2);      % Number of projections
s = size(F,2);      % Number of flat-field samples
dsz = 3.18*r*10^-4; % Domain size (cm)

% Reconstruction parameters
models = {'amap','jmap'};   % Reconstruction models
beta = [0 0];               % Flat-field reg. parameter
tv_reg = [0.01 0.01];       % Total variation reg. parameter
maxiters = [1000 1000];     % Number of iterations

%% Set up forward operator and generate problem data

% Object
vol_geom  = astra_create_vol_geom(n,n);
% Projection geometry
proj_geom = astra_create_proj_geom('parallel', n/r, r, theta);
% Create the Spot operator for ASTRA using the GPU.
P = opPermutation(reshape(reshape(1:r*p,r,p)',r*p,1));
A = opFoG(P',(dsz/n)*opTomo('cuda', proj_geom, vol_geom, GPU_ID));

%% Solve reconstruction problem
options = struct(...
    'u0',zeros(n*n,1),...
    'rho',1.8,...
    'tau',1e-2,...
    'lambda',tv_reg,...
    'maxiters',maxiters,...
    'uhold',[],...
    'mask',true,...
    'verbose',0);

N = length(models);
results = cell(N,2);
for k = 1:N
    options.model = models{k};
    options.beta = beta(k);
    options.lambda = tv_reg(k);
    options.maxiters = maxiters(k);
    
    fprintf(1,'Solving %s reconstruction problem..\n',models{k});
    [u,iterinfo] = poiss_gamma_nhtv(A,F,Y,options);
    results{k,1} = u;
    results{k,2} = iterinfo;
end
    
%% Save results
clear('A','iterinfo','P','mask','options','vol_geo','proj_geom');
astra_clear;
save([JOBNAME,'_astra_',JOBID,'.mat']);

%% Display Reconstruction Images

figure(1),clf
for k = 1:N
    
    subplot(1,2,k)
    img = reshape(results{k,1},n,n);
    imagesc(img,[0 10]);axis image off; colormap gray(256)
    title(upper(models{k}))
end


