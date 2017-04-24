function [ringRatio, ringImage] = rr_errMeas(vref,vml,vest,rrInfo)
%Ring Ratio error measure
%
% This function calculate the ring ratio error measure.


% read parameter structure 
proj_geom = rrInfo.proj_geom;        
vol_geom = rrInfo.vol_geom;
p = rrInfo.p;
n = rrInfo.n;
dsz = rrInfo.dsz;
theta = rrInfo.theta;
use_astra = rrInfo.use_astra;
if isfield(rrInfo,'mask')
    mask = rrInfo.mask;
end

% Ring image from empirical mean estimate (ML estimate)
z = (vml - vref)./vref;
z = repmat(z,1,p);
if use_astra 
    psi_emp = astrafbp(proj_geom,vol_geom,n,dsz,z');
else
    psi_emp = (n/dsz)*iradon(z,theta,n);
end
if exist('mask','var')
    psi_emp(mask) = 0;
end

% Ring image from reconstruction model estimates
z = (vest - vref)./vref;
z = repmat(z,1,p);
if use_astra 
    psi = astrafbp(proj_geom,vol_geom,n,dsz,z');
else
    psi = (n/dsz)*iradon(z,theta,n);
end
if exist('mask','var')
    psi(mask) = 0;
end

% Ring ratio
ringRatio = norm(psi(:),2)/norm(psi_emp(:),2);

% Ring Image
ringImage = psi;

