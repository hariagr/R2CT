%% Example with real data
%
% This example computes AMAP and JMAP reconstructions from the APS02
% dataset (APS_13_BM.h5) which is avaliable here: 
%   
%   ftp://ftp.xray.aps.anl.gov/pub/tomo-databank/Lorentz/aps/noisy/
%
% For more information about the dataset, please refer to the problem 
% description which is avalable here: 
%
%   https://goo.gl/qaBMel
%
% Requirements: 
%
%   ASTRA Toolbox (http://astra-toolbox.com) and CUDA-compatible GPU 
%   Spot (http://www.cs.ubc.ca/labs/scl/spot)
%
clc;clear;close all;
filename = 'APS_13_BM.h5';

% Check that gd_recon.m is in the path
if exist('gd_recon') ~= 2
    if exist(fullfile('..','src','gd_recon.m')) == 2
        addpath(fullfile('..','src'));
    else
        error('Could not find gd_recon.m')
    end
end

% Check that data file exists
if exist(filename) ~= 2
    prompt = 'Could not find data file. Do you want to download it (151 MB)? Y/N [Y]: ';
    str = input(prompt,'s');
    if isempty(str)
        str = 'Y';
    end
    if strcmp(str, 'Y')
        % if ftp connection does not work, you may be behind firewall. We
        % recommend to use Passive Mode FTP. You can use the MATLAB toolbox
        % at https://goo.gl/x9yzbg
        fprintf(1,'Connecting to server (%s)\n','ftp.xray.aps.anl.gov');
        anl_ftp = ftp('ftp.xray.aps.anl.gov');
        cd(anl_ftp,'pub/tomo-databank/Lorentz/aps/noisy/');
        fprintf(1,'Downloading %s\n',filename);
        mget(anl_ftp,filename);
        fprintf(1,'Closing connection\n')
        close(anl_ftp);
    else
        return
    end
end

% Job setup
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

% Extract center slice from file
slice = 300;
F = double(squeeze(h5read(filename','/exchange/data_white',[1,slice,1],[960,1,12]))); % Flat-field measurements
Y = double(squeeze(h5read(filename','/exchange/data',[1,slice,1],[960,1,900])));      % Measurements with object
theta = double(h5read(filename,'/exchange/theta'))/180*pi;                            % Projection angles

% Center shift correction 
Y = [Y(7:end,:) ; repmat(Y(end,:),6,1)];

% Reconstruction parameters
n = 768;            % Grid size (n x n)
r = size(Y,1);      % Number of detector elements
p = size(Y,2);      % Number of projections
s = size(F,2);      % Number of flat-field samples
dsz = 3.18*r*10^-4; % Domain size (cm)

% Reconstruction parameters
models = {'amap','jmap'};   % Reconstruction models
beta = [0 200];             % Flat-field reg. parameter
tv_reg = [0.01 0.01];       % Total variation reg. parameter
maxiters = [1000 1000];     % Number of iterations

%% Set up forward operator and generate problem data

% Object and projection geometry
vol_geom  = astra_create_vol_geom(n,n);
proj_geom = astra_create_proj_geom('parallel', n/r, r, theta);
% Create Spot operator for ASTRA using GPU
P = opPermutation(reshape(reshape(1:r*p,r,p)',r*p,1));
A = opFoG(P',(dsz/n)*opTomo('cuda', proj_geom, vol_geom, GPU_ID));

%% Solve reconstruction problems
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
    [u,iterinfo] = gd_recon(A,F,Y,options);
    results{k,1} = u;
    results{k,2} = iterinfo;
end
    
%% Display Reconstruction Images

figure(1),clf
for k = 1:N
    subplot(1,2,k)
    img = reshape(results{k,1},n,n);
    imagesc(img,[0 10]);axis image off; colormap gray(256)
    title(upper(models{k}))
end

%% Save results
%clear('A','iterinfo','P','mask','options','vol_geo','proj_geom');
astra_clear;
save([JOBNAME,'_astra_',JOBID,'.mat']);


