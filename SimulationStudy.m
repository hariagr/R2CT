%% Simulation study - Reconstruction of a phantom
% 
% To generate system matrix either use ASTRA toolbox or AIR
% Tool MATLAB Toolbox.
% To generate phantom use AIR Tool MATLAB Toolbox
% 
% System geometry - Parallel beam
%

% Set RNG seed
rng(0,'twister');

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

if exist('use_astra') == 0
    use_astra = 0;
end

% Simulation parameters
I0 = 5e2;         % Source intensity
n = 512;          % Grid size (n x n)
r = n;            % Number of detector elements
dw = n;           % Detector width
p = 720;          % Number of projections
s = 5;            % Number of flat-field samples
dsz = 2;          % Domain size (cm)

% Reconstruction parameters
models = 'jmap'; % Reconstruction models {'baseline','amap','wls','swls','jmap'}
beta = 0;        % Flat-field reg. parameter

tv_reg = 3;       % Total variation reg. parameter
maxiters = 500;   % Number of iterations

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
    % Use modified AIR Tools 'paralleltomo'
    theta = (0:p-1)*180/p;
    A = paralleltomo(n,theta,r,n);
    A = (dsz/n)*A;
    
    A_2n = paralleltomo(2*n,theta,r,2*n);
    A_2n = (dsz/(4*n))*A_2n;
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
    'lambda',tv_reg,...
    'maxiters',maxiters,...
    'uhold',[50],...
    'uref',x,...
    'vref',vref,...
    'mask',true,...
    'verbose',1);

options.model = models;
options.beta = beta;
options.u0 = zeros(n*n,1);
options.vprior = mean(F,2);
        
fprintf(1,'Solving %s reconstruction problem..\n',models);
[u,iterinfo] = poiss_gamma_nhtv(A,F,Y,options);

%% Ring Ratio and Ring images

% Circular Mask    
[XX,YY] = meshgrid(linspace(-1,1,n),linspace(-1,1,n));
mask = (XX(:).^2 + YY(:).^2 > 1);
clear('XX','YY');
  
% Ring image from empirical mean estimate
z = (mean(F,2) - vref)./vref;
z = repmat(z,1,p);
if use_astra 
    psi_emp = astrafbp(proj_geom,vol_geom,n,dsz,z');
else
    psi_emp = iradon(z,theta,n);
end
psi_emp(mask) = 0;

% Ring image from reconstruction model estimates
z = (iterinfo.vhu(:,maxiters) - vref)./vref;
z = repmat(z,1,p);
if use_astra 
    psi = astrafbp(proj_geom,vol_geom,n,dsz,z');
else
    psi = iradon(z,theta,n);
end
psi(mask) = 0;

% Ring ratio
ring_ratio = norm(psi(:),2)/norm(psi_emp(:),2);

%% Display results

figure;
subplot(1,3,1);
imagesc(reshape(x,n,n),[0 max(x)]);colormap gray;axis off image
title('Phantom')

subplot(1,3,2);
imagesc(reshape(u,n,n),[0 max(x)]);colormap gray;axis off image
title([upper(models) ' reconstruction']);

subplot(1,3,3);
imagesc(psi);colormap gray;axis off image
title('Ring Image');

fprintf('Error measures\n');
fprintf('Relative attenuation error = %.2f%%\n',100*iterinfo.relerr(maxiters));
fprintf('Ring ratio = %.2f\n',ring_ratio);

%% Save results

clear('A');
save([JOBNAME,'_astra_',JOBID,'.mat']);

