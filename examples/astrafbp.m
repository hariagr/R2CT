function Xfbp = astrafbp(proj_geom,vol_geom,n,dsz,b)
% FBP Filtered backprojection from ASTRA Toolbox

scale_fact=n/dsz;

% Baselind FBP
sino_id = astra_mex_data2d('create', '-sino', proj_geom, b);
recon_id = astra_mex_data2d('create', '-vol', vol_geom, 0);
cfg = astra_struct('FBP_CUDA');
cfg.ProjectionDataId = sino_id;
cfg.ReconstructionDataId = recon_id;
fbp_id = astra_mex_algorithm('create', cfg);
astra_mex_algorithm('run', fbp_id);
Xfbp = astra_mex_data2d('get', recon_id);
astra_mex_data2d('delete', sino_id, recon_id);
astra_mex_algorithm('delete', fbp_id);

Xfbp =Xfbp*scale_fact;