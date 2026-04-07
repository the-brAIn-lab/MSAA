# SupdateIndiStep.py
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple

def supdate_indi_step(
    S: np.ndarray,
    XCtX: np.ndarray,
    CtXtXC: np.ndarray,
    muS: np.ndarray | float,
    numObs: int,
    niter: int,
    sigmaSq: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Written by Jesper L. Hinrich, Sophia E. Bardenfleth and Morten Mørup

    Copyright (C) 2016 Technical University of Denmark - All Rights Reserved
    You may use, distribute and modify this code under the
    terms of the Multisubject Archetypal Analysis Toolbox license.
    You should have received a copy of the Multisubject Archetypal Analysis Toolbox
    license with this file. If not, please write to: jesper dot hinrich at gmail dot com, 
    or visit : https://brainconnectivity.compute.dtu.dk/ (under software)

    Python equivalent of SupdateIndiStep.m (written by Alex Shepherd, UMT)
    -------------------------------------------------------------------------
    Individual subject voxels step size
    -------------------------------------------------------------------------
    In MS-AA with SPATIAL archetypes (i.e. X is Time x Voxels) and spatial
    (voxels) noise, the inputs are:
    S: (K, F) ndarray  
        components x Voxels (equivalently: components x features)
    XCtX: (K, F) ndarray    
        components x Voxels (equivalently: components x features)
    CtXtXC: (K, K) ndarray     
        components x components
    muS:        1 x Voxels
    numObs:     Is a scalar with the number of timesteps (i.e. Time)
    niter:      Is the number of line searches to perform
    sigmaSq:    Voxels x 1 (the noise variance is part of the SSt 
                            sufficient statistic)
 
    In MS-AA with TEMPORAL archetypes (i.e. X is Voxels x Time) and spatial
    (voxels) noise, the inputs are:
    S: (K, T) ndarray 
        components x Time
    XCtX: (K,T) ndarray
        components x Time
    CtXtXC: (K,K)
        components x components
    muS:        1 x Time
    numObs:     Is a scalar with the number of voxels (i.e. Features)
    niter:      Is the number of line searches to perform
    sigmaSq:    Should not be passed (as it not part of the SSt sufficient
                                      statistic)
    Note sigmaSq is not needed during the update, as it can be considered a
    scale difference, which is irrelevant when using individual stepsize for S

    Returns
    -------
    S : (K, F) ndarray
        Updated S.
    muS : (F,) ndarray
        Updated per-column step sizes.
    SSt : (K, K) ndarray
        Sufficient statistic (sigma**2-weighted if sigmaSq is provided).
    """
    S = np.asarray(S, dtype=float, order="F")
    XCtX = np.asarray(XCtX, dtype=float, order="F")
    CtXtXC = np.asarray(CtXtXC, dtype=float, order="F")

    K, F = S.shape

    # muS handling (MATLAB: if isscalar(muS) -> ones(1, numObs); else row)
    if np.isscalar(muS):
        muS = np.ones(numObs, dtype=float)
    else:
        muS = np.asarray(muS, dtype=float).reshape(-1)
        # In MATLAB they coerce to 1 x numObs, but algorithm uses one step per column (F).
        # We allow either len(muS)==F (preferred) or len(muS)==numObs; if the latter, expand/truncate.
        if muS.size != F:
            raise ValueError(f"muS must have length F={F} (got {muS.size}).")

    # Initial per-column cost: -2 * sum(S.*XCtX) + sum(S.*(CtXtXC*S))
    CtS = CtXtXC @ S                       # (K, F)
    cost = -2.0 * np.sum(S * XCtX, axis=0) + np.sum(S * CtS, axis=0)

    # Iterative accept/reject with individual step sizes
    k = 1
    rel_delta_cost = np.inf
    denom_eps = 1e-30  # guard against divide-by-zero in rel change

    while (k <= niter) and (rel_delta_cost > 1e-12):
        # Gradient (U_s2): (CtXtXC*S - XCtX) / (numObs * numFeature)
        g = (CtS - XCtX) / (numObs * F)

        # Project gradient onto tangent space of simplex per column:
        # g[:,j] -= (g[:,j] · S[:,j]) for each j
        col_dots = np.sum(g * S, axis=0)   # (F,)
        g = g - col_dots                   # broadcast over rows

        Sold = S.copy()

        # Update with per-column step sizes
        S = Sold - g * muS                 # (K,F) - (K,F)*(F,)
        # Enforce nonnegativity
        np.maximum(S, 0.0, out=S)
        # l1-normalize columns (U_s2)
        col_sums = S.sum(axis=0)
        # Avoid division by zero for empty columns
        safe_col_sums = np.where(col_sums > 0, col_sums, 1.0)
        S = S / safe_col_sums

        # Recompute cost and accept/reject per column
        CtS = CtXtXC @ S
        cost_new = -2.0 * np.sum(S * XCtX, axis=0) + np.sum(S * CtS, axis=0)

        idx = cost_new <= cost             # accept where cost decreased
        diff = cost_new - cost
        rel_delta_cost = float(np.sum(diff * diff) / max(np.sum(cost * cost), denom_eps))

        # Revert rejected columns
        if not np.all(idx):
            S[:, ~idx] = Sold[:, ~idx]
            CtS[:, ~idx] = (CtXtXC @ Sold)[:, ~idx]  # keep consistency for next iter

        # Update step sizes & costs
        muS[idx] *= 1.2
        muS[~idx] *= 0.5
        cost[idx] = cost_new[idx]

        k += 1
        # print(rel_delta_cost)  # optional debug

    # Sufficient statistic
    if sigmaSq is not None:
        sigmaSq = np.asarray(sigmaSq, dtype=float).reshape(-1)
        if sigmaSq.size != F:
            raise ValueError(f"sigmaSq must have length F={F} (got {sigmaSq.size}).")
        # SSt = S * bsxfun(@rdivide, S, sigmaSq')'  (MATLAB)
        #    = S @ (S / sigmaSq_row).T
        SSt = S @ (S / sigmaSq).T
    else:
        SSt = S @ S.T

    return S, muS, SSt
