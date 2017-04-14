function [u, iterinfo] = poiss_gamma_nhtv(A,F,Y,options)
% POISS_GAMMA_NHTV   MAP Estimation based on Poisson likelihood 
% with a Gamma flat-field prior and a nonnegativity + Huber-TV 
% attenuation prior. The general reconstruction model is given by
%
%    minimize    J(u,v) + lambda*TV(u)
%    subject to  u >= 0 
%
% where u is the attenuation image, v is the flat-field, and 
%
%    J(u,v) = v'*d(u) + Y(:)'*A*u - c'*log(v) 
%    c = sum(F,2) + sum(Y,2) + alpha - 1
%    d(u) = s + sum(reshape(exp(-A*u),r,p),2) + beta
%
% and where A is m-by-n, Y is r-by-p, and F is r-by-s.
%
% Reconstruction models:
%
%   * 'jmap' : Joint MAP estimation of u and v. (default)
%
%   * 'baseline' : Solves baseline reconstruction (inverse crime) where
%      v is replace by the true flat-field (i.e., v = options.vref).
%
%   * 'amap' : Solves approximate MAP estimatation problem where v is  
%      replaced by the ML estimate (i.e., v = mean(F,2)).
%
%   * 'wls' : Solves weighted least-squares approximation where v is  
%      replaced by the ML estimate (i.e., v = mean(F,2)).
%
%   * 'swls' : Solves stripe-weighted least-squares approximation where
%      v is replaced by the ML estimate (i.e., v = mean(F,2)).
%
% The optional input 'options' is a struct with one or more fields:
%
%    'model'      {'jmap','baseline','amap','wls','swls'}
%    'maxiters'   Maximum number of iterations  (default: 200)
%    'tolf'       Toleration relative objective value (default:1e-8)
%    'rho'        Gradient step multiplier      (default: 1.5)
%    'lambda'     TV-regularization parameter   (default: 0.0)
%    'tau'        TV-smoothing parameter        (default: 1e-2)
%    'alpha'      Flat-field hyperparameter     (default: 1.0)
%    'beta'       Flat-field hyperparameter     (default: 0.0)
%    'u0'         Starting point                (default: zeros(n,1))
%    'uref'       Reference image (for simulation studies only)
%    'vref'       Reference flat-field (for simulation studies only)
    
% Check/extract problem dimensions
[r,p] = size(Y);
n = size(A,2);
N = sqrt(n);
s = size(F,2);
assert(size(A,1) == r*p)
assert(size(F,1) == r)

% Reconstruction model
if isfield(options,'model')
    model = options.model;
else
    model = 'jmap';
end

if isfield(options,'verbose')
    verbose = options.verbose;
else
    verbose = 0;
end

% Attenuation hyperparameter (TV-regularization)
if isfield(options,'lambda')
    lambda = options.lambda;
    tv = 0.0;
else
    lambda = 0.0;
    tv = 0.0;
end

% Maximum numer of iterations
if isfield(options,'maxiters')
    maxiters = options.maxiters;
else
    maxiters = 200;
end

% Stopping tolerance for relative change in objective function
if isfield(options,'tolf')
    tolf = options.tolf;
else
    tolf = 1e-8;
end

% Save intermediate iterations
if isfield(options,'uhold')
    uhold_idx = options.uhold;
    uhold = zeros(n,length(uhold_idx));
end

% Step-size multiplier
if isfield(options,'rho')
    rho = options.rho;
    assert(rho > 0)
    if rho >= 2.0
        warning('rho >= 2.0: convergence not guaranteed')
    end
else
    rho = 1.5;
end

% Total variation smoothing
if isfield(options,'tau')
    tau = options.tau;
    assert(tau > 0)
else
    tau = 1e-2;
end

% Reference flat-field
if isfield(options,'vref')
    vref = (options.vref).*ones(r,1);
    vhu = zeros(r,maxiters+1);
end

% Flat-field hyperparameter
if isfield(options,'beta')
    beta = options.beta.*ones(r,1);
    assert(all(beta>=0))
else
    beta = zeros(r,1);
end

% Flat-field prior
if isfield(options,'vprior')
    vprior = options.vprior;
else
    vprior = mean(F,2);
end

% Flat-field hyperparameter
if isfield(options,'alpha')
    alpha = options.alpha.*ones(r,1);
    assert(all(alpha>=1))
else
    alpha = 1.0 + vprior.*beta;
end

% Flat-field ML estimate
vh = mean(F,2);
c = sum(F,2) + sum(Y,2) + alpha  - 1.0;
d = @(u) s + sum(exp(-reshape(A*u,r,p)),2) + beta;

% Check model and set up objective function
y = Y(:);
if strcmp(model,'wls')
    assert(all(y>0),'Y must have positive elements')
    b = -log(y./repmat(vh,p,1));
    J = @(u) Jwls(u,A,b,y);
elseif strcmp(model,'baseline')
    assert(exist('vref')==1);
    J = @(u) Jamap(u,A,y,vref);
elseif strcmp(model,'amap')
    J = @(u) Jamap(u,A,y,vh);
elseif strcmp(model,'jmap')
    J = @(u) Jjmap(u,A,y,c,d);
elseif strcmp(model,'swls')
    assert(all(y>0),'Y must have positive elements')
    b = -log(y./repmat(vh,p,1));
    J = @(u) Jswls(u,A,b,y,vh,s,alpha);
else
    error('Invalid reconstruction model')
end

% Starting point
if isfield(options,'u0')
    u = options.u0;
else
    u = zeros(n,1);
end

% Circular mask
if isfield(options,'mask')
    mask = options.mask;
else
    mask = false;
end
if mask    
    [XX,YY] = meshgrid(linspace(-1,1,N),linspace(-1,1,N));
    M = (XX(:).^2 + YY(:).^2 > 1);
    clear('XX','YY');
end    
    
% Reference image 
if isfield(options,'uref')
    uref = options.uref;
    assert(size(uref,1) == n)
    assert(size(uref,2) == 1)
    relerr = zeros(maxiters+1,1);
end

% Finite-difference matrix with Neumann boundary conditions
Dn = spdiags([ones(N-1,1),-ones(N-1,1);0,-1],[0,1],N,N);
Dy = kron(speye(N),Dn);
Dx = kron(Dn,speye(N));

% Compute Lipschitz constant (upper bound)
if ismatrix(A) & ~isa(A,'opFoG')
    if verbose
        fprintf(1,'Computing Lipschitz constant\n');
    end
    L = normest(spdiags(sqrt(y),0,r*p,r*p)*A)^2;
elseif isa(A,'opFoG')
    % Power iteration
    if verbose
        fprintf(1,'Estimating Lipschitz constant via power iteration\n');
    end
    tmp = 1/sqrt(n)*ones(n,1);
    for it = 1:20
        tmp = A'*(y.*(A*tmp/norm(tmp)));
    end
    L = norm(tmp);
    clear('tmp');
end
L = L + lambda*8/tau;
t = rho/L;    

% Gradient descent method
if verbose
    fprintf(1,'%4s %12s %8s %8s\n','it.','objval','opt','step'); 
end
fhist = zeros(maxiters+1,1);
fold = 0;
for it = 1:maxiters+1
    if exist('relerr','var') == 1
        % Compute relative error
        relerr(it) = norm(u-uref)/norm(uref);
    end
    if exist('uhold_idx','var')==1 && any(uhold_idx==it)
        uhold(:,uhold_idx==it) = u;
    end
    if exist('vhu','var') == 1
        % Compute relative flat-field error
        vhu(:,it) = c./d(u);
    end
    
    % Compute objective value and gradient
    [val,grad] = J(u);
    fhist(it) = val;    
    if lambda > 0
        [tv,tvgrad] = huber_tv(Dx,Dy,u,tau);
        grad = grad + lambda*tvgrad;
        fhist(it) = fhist(it) + lambda*tv;
    end
    if verbose && mod(it-1,50) == 0
        nopt = t*abs((u>0).*grad + (u<=0).*min(grad,0));
        if mask
            nopt(M) = 0;
        end
        fprintf(1,'%4i %12.6e %8.2e %8.2e\n',it-1,fhist(it),norm(nopt),t*norm(grad));
    end
    
    % Stopping criteria
    STOP(1) = abs(fhist(it)-fold)  <= tolf*(1+abs(fold));
    STOP(2) = (it >= maxiters + 1);
    %if it == maxiters+1
    if STOP(1) || STOP(2)
        break
    end
    fold = fhist(it); % update function value
    
    % Update u
    u = max(0,u-t*grad);
    if mask
        u(M) = 0;
    end
    
end

% Return info struct
iterinfo = struct('fhist',fhist);
if exist('relerr','var') == 1
    iterinfo.relerr = relerr;    
end
if exist('uhold','var') == 1
    iterinfo.uhold = uhold;
end
iterinfo.objval = [val, tv];
iterinfo.ngrad = grad*t;
if mask
    iterinfo.ngrad(M) = 0;
end
iterinfo.Au = A*u;
iterinfo.optu = abs((u>0).*iterinfo.ngrad + (u<=0).*min(iterinfo.ngrad,0));
if exist('vhu','var')==1
    iterinfo.vhu = vhu;
else
   iterinfo.vhu = c./d(u); 
end
iterinfo.theta = [s*ones(r,1) d(u)-beta-s beta]./repmat(d(u),1,3);
iterinfo.options = options;
iterinfo.maxiter = it-1;
iterinfo.relobjval = abs(fhist(it)-fold)/(1+abs(fold));

end

function [tv, grad] = huber_tv(Dx,Dy,u,tau)
% Huber total variation
g = [Dx*u Dy*u];
ng = sqrt(sum(g.^2,2));
tv = sum((ng <= tau).*(ng.^2/(2*tau)) + (ng > tau).*(ng - tau/2));
grad = Dx'*(g(:,1)./max(tau,ng)) + Dy'*(g(:,2)./max(tau,ng));
end

function [val, grad] = Jwls(u,A,b,y)
% Weighted least-squares approximation
r = A*u-b;
grad = A'*(y.*r);
val = 0.5*r'*(y.*r);
end

function [val, grad] = Jamap(u,A,y,v)
% Baseline/approximate MAP estimate
p = size(A,1)/size(v,1);
b = A*u;
yh = repmat(v,p,1).*exp(-b);
grad = A'*(y-yh);
val = sum(yh) + y'*b;
end

function [val, grad] = Jjmap(u,A,y,c,d)
% Joint MAP estimate
p = size(A,1)/size(c,1);
du = d(u);
b = A*u;
yh = repmat(c./du,p,1).*exp(-b);
grad = A'*(y-yh);
val = y'*b + c'*log(du);
end

function [val, grad] = Jswls(u,A,b,y,v,s,alpha)
% Stripe-weighted least-squares approximation
p = size(A,1)/size(v,1);
r = size(v,1);
R = reshape(A*u-b,r,[]);
SR = R;
for k = 1:r
    yk = y(k:r:end)';
    SR(k,:) = yk.*SR(k,:) - 1/(s*v(k)+sum(yk)+alpha(k)-1)*(yk*SR(k,:)')*yk;
end
val = R(:)'*SR(:);
grad = A'*SR(:);
end
