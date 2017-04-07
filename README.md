# Convex Reconstruction Models for X-ray tomographic Imaging with Uncertain Flat-fields

This MATLAB package provides several reconstruction models for X-ray tomographic imaging. These models are based on a MAP estimation of Poisson likelihood functions with a Gamma flat-field prior and a Huber-TV prior with nonnegativity constraint on the attenuation function.

The general reconstruction model is given by

  minimize    J(u,v) + lambda*TV(u)
  subject to  u >= 0 

where u is the attenuation image, v is the flat-field.

Reconstruction models:

'JMAP' : Joint MAP estimation of u and v. (default)

'BASELINE' : Solves baseline reconstruction (inverse crime) where v is replace by the true flat-field.

'AMAP' : Solves approximate MAP estimatation problem where v is replaced by the flat-field ML estimate.

'WLS' : Solves weighted least-squares approximation where v is replaced by the flat-field ML estimate.

'SWLS' : Solves stripe-weighted least-squares approximation where v is replaced by the flat-field ML estimate.




