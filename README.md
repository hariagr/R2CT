# Convex Reconstruction Models for X-ray tomographic Imaging with Uncertain Flat-fields

This MATLAB package provides several reconstruction models for X-ray tomographic imaging. These models are based on a MAP estimation of Poisson likelihood functions with a Gamma flat-field prior and a Huber-TV prior with nonnegativity constraint on the attenuation function.

The general reconstruction model is given by

  minimize    J(u,v) + lambda*TV(u)
  subject to  u >= 0 

where u is the attenuation image, v is the flat-field, and 

    J(u,v) = v'*d(u) + Y(:)'*A*u - c'*log(v) 
    c = sum(F,2) + sum(Y,2) + alpha - 1
    d(u) = s + sum(reshape(exp(-A*u),r,p),2) + beta

and where A is m-by-n, Y is r-by-p, and F is r-by-s.

Reconstruction models:

'JMAP' : Joint MAP estimation of u and v. (default)

'BASELINE' : Solves baseline reconstruction (inverse crime) where
v is replace by the true flat-field (i.e., v = options.vref).

'AMAP' : Solves approximate MAP estimatation problem where v is  
      replaced by the ML estimate (i.e., v = mean(F,2)).

'WLS' : Solves weighted least-squares approximation where v is  
      replaced by the ML estimate (i.e., v = mean(F,2)).

'SWLS' : Solves stripe-weighted least-squares approximation where
      v is replaced by the ML estimate (i.e., v = mean(F,2)).



