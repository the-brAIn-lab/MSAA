from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import time

try:
    import cupy as cp
except Exception:
    cp = None

from mgetopt import mgetopt
from SupdateIndiStep import supdate_indi_step
from furthest_sum import furthest_sum
from furthest_sum_gpu import furthest_sum_gpu


@dataclass
class Subject:
    """Container matching the MATLAB 'subj' struct."""
    X:  np.ndarray          # shape (T, V)
    sX: np.ndarray          # shape (T, sV)
    T:            Any   = field(default=None)
    SST:          Any   = field(default=None)
    sigmaSq:      Any   = field(default=None)
    S:            Any   = field(default=None)
    muS:          Any   = field(default=None)
    sXC:          Any   = field(default=None)
    XCtX:         Any   = field(default=None)
    CtXtXC:       Any   = field(default=None)
    SSt:          Any   = field(default=None)
    XSt:          Any   = field(default=None)
    SST_sigmaSq:  float = field(default=0.0)
    NLL:          float = field(default=0.0)


def _to_numpy(a, runGPU: bool) -> np.ndarray:
    if runGPU and hasattr(a, 'get'):
        try:
            return a.get()
        except Exception:
            return np.array(a)
    return np.asarray(a)


def _to_float(a, runGPU: bool) -> float:
    if runGPU and hasattr(a, 'get'):
        return float(a.get())
    return float(a)


def multi_subject_aa(
    subj: List[Subject],
    noc: int,
    opts: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, float, float]:
    if opts is None:
        opts = {}

    conv_crit       = mgetopt(opts, 'conv_crit',        1e-6)
    maxiter         = int(mgetopt(opts, 'maxiter',       100))
    fix_var_iter    = int(mgetopt(opts, 'fix_var_iter',  5))
    runGPU          = bool(mgetopt(opts, 'use_gpu',      False))
    voxelVariance   = bool(mgetopt(opts, 'heteroscedastic', True))
    numCstep        = int(mgetopt(opts, 'numCstep',      10))
    numSstep        = int(mgetopt(opts, 'numSstep',      20))
    sort_crit       = mgetopt(opts, 'sort_crit',         'corr')
    init_type       = mgetopt(opts, 'init',              'FurthestSum')
    initial_S_steps = int(mgetopt(opts, 'initSstep',     250))
    rngSEED         = mgetopt(opts, 'rngSEED',           None)

    if runGPU and cp is None:
        raise ImportError("opts['use_gpu']=True but CuPy is not installed.")

    rng = np.random.default_rng(rngSEED)

    V      = subj[0].X.shape[1]
    sV     = subj[0].sX.shape[1]
    B      = len(subj)
    T_list = [s.sX.shape[0] for s in subj]

    xp = cp if runGPU else np

    # ------------------------------------------------------------------ #
    #  Initialise C                                                        #
    # ------------------------------------------------------------------ #
    if init_type.lower() == 'furthestsum':
        total_T   = sum(T_list)
        Xcombined = np.zeros((total_T, sV), dtype=float)
        offset    = 0
        for s in subj:
            T = s.sX.shape[0]
            Xcombined[offset:offset + T, :] = np.asarray(s.sX, dtype=float)
            offset += T

        seed = int(rng.integers(low=0, high=sV))

        if runGPU and cp is not None:
            print(f"[init] Running furthest_sum_gpu (K={noc}, N={total_T})...")
            idx = furthest_sum_gpu(
                Xcombined,          # (32400, 700) — same as CPU, no transpose
                noc=noc,
                i=seed,
                exclude=None,
                treat_as_kernel=False,
                one_based=False,
            )

        else:
            print(f"[init] Running furthest_sum CPU (K={noc}, N={total_T})...")
            idx = furthest_sum(
                Xcombined, noc=noc, i=seed,
                exclude=None, treat_as_kernel=False,
            )

        idx = np.asarray(idx, dtype=int)
        C   = xp.zeros((sV, noc), dtype=float)
        C[idx, xp.arange(noc)] = 1.0

    else:
        print(f"[init] Random initialisation (K={noc})...")
        C_rand  = xp.asarray(rng.random((sV, noc)), dtype=float)
        C_rand /= C_rand.sum(axis=0, keepdims=True) + xp.finfo(float).eps
        C       = C_rand

    muC = xp.array(1.0)

    # ------------------------------------------------------------------ #
    #  Move subject arrays to GPU and precompute SST                      #
    # ------------------------------------------------------------------ #
    for s in subj:
        if runGPU:
            s.X  = cp.asarray(s.X,  dtype=float)
            s.sX = cp.asarray(s.sX, dtype=float)
        s.T   = s.sX.shape[0]
        s.SST = (s.X * s.X).sum()

    SST = float(sum(_to_float(s.SST, runGPU) for s in subj))

    # ------------------------------------------------------------------ #
    #  Initialise per-subject quantities                                   #
    # ------------------------------------------------------------------ #
    for s in subj:
        if voxelVariance:
            s.sigmaSq = xp.ones((V, 1), dtype=float) * (SST / (sum(T_list) * V))
        else:
            s.sigmaSq = xp.ones((V, 1), dtype=float)

        U    = xp.asarray(rng.random((noc, V)), dtype=float)
        s.S  = -xp.log(U + xp.finfo(float).tiny)
        s.S /= s.S.sum(axis=0, keepdims=True) + xp.finfo(float).eps

        s.muS = xp.ones((1,), dtype=float)

        s.sXC    = s.sX @ C
        s.XCtX   = s.sXC.T @ s.X
        s.CtXtXC = s.sXC.T @ s.sXC
        s.SSt    = s.S @ s.S.T
        s.XSt    = s.X @ s.S.T

        S_np, muS_np, SSt_np = supdate_indi_step(
            _to_numpy(s.S,                 runGPU),
            _to_numpy(s.XCtX,              runGPU),
            _to_numpy(s.CtXtXC,            runGPU),
            np.ones(s.S.shape[1],          dtype=float),
            int(s.T),
            int(initial_S_steps),
            _to_numpy(s.sigmaSq.squeeze(), runGPU),
        )
        s.S   = xp.asarray(S_np)
        s.muS = xp.ones((s.S.shape[1],), dtype=float)
        s.SSt = xp.asarray(SSt_np)

    # ------------------------------------------------------------------ #
    #  Initial NLL                                                         #
    # ------------------------------------------------------------------ #
    NLL         = 0.0
    SST_sigmaSq = 0.0
    for s in subj:
        s.XSt = s.X @ (s.S / s.sigmaSq.T).T
        s.SSt = s.S @ (s.S / s.sigmaSq.T).T

        s.SST_sigmaSq = _to_float(
            (s.X * (s.X / s.sigmaSq.T)).sum(), runGPU
        )
        s.NLL = float(
            0.5 * s.SST_sigmaSq
            - _to_float((s.sXC    * s.XSt).sum(),        runGPU)
            + 0.5 * _to_float((s.CtXtXC * s.SSt).sum(),  runGPU)
            + (s.T / 2.0) * (
                V * np.log(2.0 * np.pi)
                + _to_float(xp.log(s.sigmaSq).sum(),      runGPU)
            )
        )
        NLL         += s.NLL
        SST_sigmaSq += s.SST_sigmaSq

    t_start  = time.perf_counter()
    cost_fun = np.zeros((maxiter,), dtype=float)

    noise_threshold_opt = mgetopt(opts, 'noise_threshold', None)
    var_threshold = (
        float(noise_threshold_opt)
        if noise_threshold_opt is not None
        else (SST / (sum(T_list) * V)) * 1e-2
    )

    # ------------------------------------------------------------------ #
    #  Main optimisation loop                                              #
    # ------------------------------------------------------------------ #
    iter_ = 0
    dNLL  = np.inf

    while (
        (abs(dNLL) >= conv_crit * abs(NLL)) or (fix_var_iter >= iter_)
    ) and (iter_ < maxiter):

        iter_  += 1
        NLL_old = NLL
        cost_fun[iter_ - 1] = NLL

        # ---- C update ----
        C, muC, NLL = _Cupdate_multi_subjects(
            subj, C, muC, NLL, numCstep, runGPU
        )

        # Update sufficient statistics that depend on C
        for s in subj:
            s.sXC    = s.sX @ C
            s.XCtX   = s.sXC.T @ s.X
            s.CtXtXC = s.sXC.T @ s.sXC
            s.NLL    = float(
                0.5 * s.SST_sigmaSq
                - _to_float((s.XCtX   * (s.S / s.sigmaSq.T)).sum(), runGPU)
                + 0.5 * _to_float((s.CtXtXC * s.SSt).sum(),         runGPU)
                + (s.T / 2.0) * (
                    V * np.log(2.0 * np.pi)
                    + _to_float(xp.log(s.sigmaSq).sum(),             runGPU)
                )
            )

        # ---- S update ----
        NLL         = 0.0
        SST_sigmaSq = 0.0
        for s in subj:
            S_np, muS_np, SSt_np = supdate_indi_step(
                _to_numpy(s.S,                 runGPU),
                _to_numpy(s.XCtX,              runGPU),
                _to_numpy(s.CtXtXC,            runGPU),
                _to_numpy(s.muS,               runGPU),
                int(s.T),
                int(numSstep),
                _to_numpy(s.sigmaSq.squeeze(), runGPU),
            )
            s.S   = xp.asarray(S_np)
            s.muS = xp.asarray(muS_np)
            s.SSt = xp.asarray(SSt_np)

            if voxelVariance and (iter_ > fix_var_iter):
                resid         = s.X - (s.sXC @ s.S)
                s.sigmaSq     = (resid * resid).sum(axis=0, keepdims=True).T / float(s.T)
                s.sigmaSq     = xp.maximum(s.sigmaSq, var_threshold)
                s.XSt         = s.X @ (s.S / s.sigmaSq.T).T
                s.SSt         = s.S @ (s.S / s.sigmaSq.T).T
                s.SST_sigmaSq = _to_float(
                    (s.X * (s.X / s.sigmaSq.T)).sum(), runGPU
                )
            else:
                s.XSt = s.X @ (s.S / s.sigmaSq.T).T

            s.NLL = float(
                0.5 * s.SST_sigmaSq
                - _to_float((s.sXC    * s.XSt).sum(),        runGPU)
                + 0.5 * _to_float((s.CtXtXC * s.SSt).sum(),  runGPU)
                + (s.T / 2.0) * (
                    V * np.log(2.0 * np.pi)
                    + _to_float(xp.log(s.sigmaSq).sum(),      runGPU)
                )
            )
            NLL         += s.NLL
            SST_sigmaSq += s.SST_sigmaSq

        dNLL = NLL_old - NLL

        if iter_ % 5 == 0:
            print(
                f"  iter {iter_:4d} | NLL {NLL:.4e} "
                f"| dNLL/NLL {dNLL / abs(NLL):.4e}"
            )

        if (dNLL / abs(NLL) < 0) and (abs(dNLL / NLL) > conv_crit):
            pass

    # ------------------------------------------------------------------ #
    #  Wrap-up                                                             #
    # ------------------------------------------------------------------ #
    elapsed = time.perf_counter() - t_start

    SSE = []
    for s in subj:
        sXC_S = s.sXC @ s.S
        sse   = float(
            (cp if runGPU else np).linalg.norm(s.X - sXC_S, ord='fro') ** 2
        )
        SSE.append(sse)

    varexpl = (SST - sum(SSE)) / SST

    ind = np.arange(noc)
    if sort_crit.lower() == 'corr' and (sum(T_list) == max(T_list) * B):
        arch      = np.zeros((T_list[0], B))
        mean_corr = np.zeros((noc,))
        for j in range(noc):
            for bi, s in enumerate(subj):
                arch[:, bi] = _to_numpy(s.sXC[:, j], runGPU)
            Ccorr        = np.corrcoef(arch, rowvar=False)
            iu           = np.triu_indices(B, k=1)
            mean_corr[j] = float(Ccorr[iu].mean())
        ind = np.argsort(mean_corr)[::-1]

    if not np.array_equal(ind, np.arange(noc)):
        C = C[:, ind]
        for s in subj:
            s.S   = s.S[ind, :]
            s.sXC = s.sXC[:, ind]

    C_np = _to_numpy(C, runGPU)
    results_subj: List[Dict[str, Any]] = []
    for bi, s in enumerate(subj):
        results_subj.append({
            "S":           _to_numpy(s.S,       runGPU),
            "sXC":         _to_numpy(s.sXC,     runGPU),
            "sigmaSq":     _to_numpy(s.sigmaSq, runGPU).reshape(-1, 1),
            "NLL":         s.NLL,
            "SSE":         float(SSE[bi]),
            "SST":         _to_float(s.SST,  runGPU),
            "SST_sigmaSq": s.SST_sigmaSq,
        })

    return results_subj, C_np, cost_fun[:iter_], float(varexpl), float(elapsed)


# ------------------------------------------------------------------ #
#  C update                                                            #
# ------------------------------------------------------------------ #

def _Cupdate_multi_subjects(subj, C, muC, NLL, niter, runGPU):
    xp      = cp if runGPU else np
    sV, noc = C.shape
    V       = subj[0].X.shape[1]
    total_T = sum(int(s.T) for s in subj)

    for s in subj:
        if not isinstance(s.SST_sigmaSq, float):
            s.SST_sigmaSq = _to_float(s.SST_sigmaSq, runGPU)
        if not isinstance(s.NLL, float):
            s.NLL = _to_float(s.NLL, runGPU)

    # sXtsX precomputed once — sX never changes
    sXtsX_list = [s.sX.T @ s.sX for s in subj]   # (sV, sV)

    # Log-variance — one sync per subject, done once
    log_var_terms = [
        float(s.T) / 2.0 * (
            V * np.log(2.0 * np.pi)
            + _to_float(xp.log(s.sigmaSq).sum(), runGPU)
        )
        for s in subj
    ]

    for _ in range(niter):
        NLL_old = NLL
        if float(muC) < 1e-12:
            muC = xp.array(1e-4)

        # XtXSt depends on XSt which changes each outer iter — recompute here
        XtXSt_list = [s.sX.T @ s.XSt for s in subj]

        g = xp.zeros((sV, noc), dtype=float)
        for sXtsX, XtXSt, s in zip(sXtsX_list, XtXSt_list, subj):
            g += sXtsX @ (C @ s.SSt) - XtXSt
        g /= total_T * sV
        g  = g - (g * C).sum(axis=0, keepdims=True)

        Cold     = C.copy()
        stop     = False
        ls_steps = 0

        while not stop:
            C = Cold - muC * g
            xp.maximum(C, 0.0, out=C)
            C /= C.sum(axis=0, keepdims=True) + xp.finfo(float).eps

            NLL_gpu = xp.array(0.0)
            for log_var, s in zip(log_var_terms, subj):
                sXC_new = s.sX @ C
                CtXtXC  = sXC_new.T @ sXC_new
                NLL_gpu = (
                    NLL_gpu
                    + 0.5 * s.SST_sigmaSq
                    - (sXC_new * s.XSt).sum()
                    + 0.5 * (CtXtXC * s.SSt).sum()
                    + log_var
                )

            NLL_tmp  = _to_float(NLL_gpu, runGPU)
            ls_steps += 1

            if NLL_tmp <= NLL_old * (1.0 + 1e-9):
                muC  = muC * 1.2
                NLL  = NLL_tmp
                stop = True
            elif ls_steps >= 50:
                C   = Cold
                NLL = NLL_old
                stop = True
                muC = xp.array(1e-4)
            else:
                muC = muC / 2.0

    for s in subj:
        s.sXC    = s.sX @ C
        s.CtXtXC = s.sXC.T @ s.sXC

    return C, muC, NLL