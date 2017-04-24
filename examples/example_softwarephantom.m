%% Reconstruction of a software phantom
% 
% This example demonstrate the basic use case of toolbox. 
% 
%
% In this script, we have implemented two options to generate system matrix
% i.e. either using ASTRA toolbox or AIR Tool MATLAB Toolbox.
% 
% The software phantom is generated from AIR Tool MATLAB Toolbox.
% 
% The experimental settings are described in the following paper
%   Hari Om Aggrawal, Martin S. Andersen, Sean Rose, and Emil Sidky,
%   "A Convex Reconstruction Model for X-ray tomographic Imaging with
%   Uncertain Flat-fields", submitted to IEEE Transactions on
%   Computational Imaging, 2017. 
%
clc;clear;

% Check that gd_recon.m is in the path
if exist('gd_recon') ~= 2
    if exist(fullfile('..','src','gd_recon.m')) == 2
        addpath(fullfile('..','src'));
    else
        error('Could not find gd_recon.m')
    end
end

% Set RNG seed
rng(0,'twister');

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
if strcmp('CUDADEV','')
    GPU_ID = str2num(CUDADEV);
else
    GPU_ID = 0;
end
fprintf(1,'GPU ID: %i\n',GPU_ID);

% Flag - Astra: 1, AIRTOOL:0 
use_astra = 0;

% Simulation parameters
I0  = 5e2;         % Source intensity
n   = 128;         % Grid size (n x n)
r   = n;           % Number of detector elements
dw  = n;           % Detector width
p   = 720;         % Number of projections
s   = 5;           % Number of flat-field samples
dsz = 2;           % Domain size (cm)

% Reconstruction parameters
models = 'jmap'; % Reconstruction models {'baseline','amap','wls','swls','jmap'}
beta   = 0;      % Flat-field reg. parameter
u0     = 'amap'; % Initialization based on 'amap' OR zero

tv_reg   = 7;    % Total variation reg. parameter - 7
maxiters = 500;  % Number of iterations

%% Set up forward operator and generate problem data

if use_astra  
    % Object
    vol_geom = astra_create_vol_geom(n,n);
    % Projection geometry
    theta = linspace2(0,pi,p);
    proj_geom = astra_create_proj_geom('parallel', dw/r, r, theta);
    % Create the Spot operator for ASTRA using the GPU.
    P = opPermutation(reshape(reshape(1:r*p,r,p)',r*p,1));
    A = opFoG(P',(dsz/n)*opTomo('cuda', proj_geom, vol_geom, GPU_ID));

    % Generate problem data on fine grid to avoid inverse crime
    vol_geom_2n  = astra_create_vol_geom(2*n,2*n);
    proj_geom_2n = astra_create_proj_geom('parallel', 2*dw/r, r, theta);
    A_2n = opFoG(P',(dsz/(4*n))*opTomo('cuda', proj_geom_2n, vol_geom_2n, GPU_ID));

else
    % Use AIR Tools 'paralleltomo'
    theta = (0:p-1)*180/p;
    A = paralleltomo(n,theta,r,n);
    A = (dsz/n)*A;
    
    A_2n = paralleltomo(2*n,theta,r,2*n);
    A_2n = (dsz/(2*n))*A_2n;
end

% Phantom on the fine grid
xf = reshape(phantomgallery('grains',2*n,[],0),[],1);

% Phantom circular mask
[Xt,Yt] = meshgrid(linspace(-1,1,2*n),linspace(-1,1,2*n));
mask_radius = 0.8;
mask = Xt.^2 + Yt.^2 > mask_radius;
clear('Xt','Yt');
xf(mask) = 0;

% Phantom on the coarse grid 
x = reshape(imresize(reshape(xf,2*n,[]),0.5),[],1);

% Line integral
b = A_2n*xf;

% source intensity (Reference flat-field)
vref = poissrnd(I0*ones(r,1));

% Generate pseudo-random Poisson distributed measurements
F = poissrnd(I0*ones(r,s));
Y = poissrnd(repmat(vref,1,p).*reshape(exp(-b),r,p));

clear('A_2n','vol_geom_2n','proj_geom_2n');

%% Solve reconstruction problem

options = struct(...
    'u0',zeros(n*n,1),...
    'rho',1.8,...
    'tau',1e-2,...
    'tolf',1e-8,...
    'lambda',tv_reg,...
    'maxiters',maxiters,...
    'uhold',[50],...
    'uref',x,...
    'vref',vref,...
    'mask',true,...
    'verbose',1);

% Initialization u0 estimate
if strcmp(u0,'amap')
    options.model = 'amap';
    options.maxiters = 50;
    
    fprintf(1,'Estimating u0 as 50th iterate of AMAP reconstruction model\n');
    options.u0 = gd_recon(A,F,Y,options);
else
    options.u0 = zeros(n*n,1);
end

options.model = models;
options.beta = beta;
options.vprior = mean(F,2);
options.maxiters = maxiters;

fprintf(1,'Solving %s reconstruction problem..\n',models);
[u,iterinfo] = gd_recon(A,F,Y,options);


% AMAP reconstruction to compare with JMAP reconstruction
options.model = 'amap';
options.u0 = zeros(n*n,1);

fprintf(1,'Solving %s reconstruction problem..\n',options.model);
[u_amap,iterinfo_amap] = gd_recon(A,F,Y,options);

%% Ring Ratio and Ring images

% Circular Mask    
[XX,YY] = meshgrid(linspace(-1,1,n),linspace(-1,1,n));
mask = (XX(:).^2 + YY(:).^2 > 1);
clear('XX','YY');

if ~use_astra
    proj_geom = []; vol_geom=[];
end
rrInfo = struct(...
            'proj_geom',proj_geom,...
            'vol_geom',vol_geom,...
            'p', p,...
            'n',n,...
            'dsz',dsz,...
            'theta',theta,...
            'mask',mask,...
            'use_astra',use_astra);

% Ring Ratio error measures
vml = mean(F,2);
vest = iterinfo.vhu(:,iterinfo.maxiter);
[ringRatio, ringImage] = rr_errMeas(vref,vml,vest,rrInfo);

vml = mean(F,2);
vest = iterinfo_amap.vhu(:,iterinfo.maxiter);
[ringRatio_amap, ringImage_amap] = rr_errMeas(vref,vml,vest,rrInfo);

%% Display results

figure;
subplot(2,3,1);
imagesc(reshape(x,n,n),[0 max(x)]);colormap gray;axis off image;colorbar
title('Phantom')

subplot(2,3,2);
imagesc(reshape(u_amap,n,n),[0 max(x)]);colormap gray;axis off image;colorbar
title(['AMAP reconstruction']);

subplot(2,3,3);
imagesc(reshape(u,n,n),[0 max(x)]);colormap gray;axis off image;colorbar
title([upper(models) ' reconstruction']);

scale = max(abs(ringImage_amap(:)))*0.5;
subplot(2,3,5);
imagesc(abs(ringImage_amap),[0 scale]);colormap gray;axis off image;colorbar
title('Ring Image - AMAP');

subplot(2,3,6);
imagesc(abs(ringImage),[0 scale]);colormap gray;axis off image;colorbar
title('Ring Image - JMAP');

fprintf('Error measures\n');
fprintf('AMAP: Relative attenuation error = %.2f%%\n',100*iterinfo_amap.relerr(iterinfo_amap.maxiter));
fprintf('AMAP: Ring ratio = %.2f\n \n',ringRatio_amap);

fprintf('JMAP: Relative attenuation error = %.2f%%\n',100*iterinfo.relerr(iterinfo.maxiter));
fprintf('JMAP: Ring ratio = %.2f\n',ringRatio);

%% Save results

clear('A');
save([JOBNAME,'_',JOBID,'.mat']);

