# use the firt k day from simulated data and estimated range of reproduction number to predict overall outbreak probability 
# Also output conditional probability, outbreak probability given first three cases, outbreak probability given reproduction number 
"""
Generalised, asks user to input number of weeks.
Will need to change plotting functions in next cell to allow this as well.
"""
import gradio as gr
import numpy as np
import scipy 
from scipy.stats import gamma
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.optimize import root_scalar
import math
from scipy.special import lambertw


# --- parameters from paper ---
MEAN_SI_DAYS = 15.3
SD_SI_DAYS = 9.3
R_MIN, R_MAX = 0.0, 10.0

# --- weekly discretisation of continuous gamma SI using triangular kernel ---
def weekly_w(max_weeks=50, mean=MEAN_SI_DAYS, sd=SD_SI_DAYS):
    shape = (mean/sd)**2
    scale = sd**2 / mean
    def g(x):
        return gamma.pdf(x, a=shape, scale=scale)
    w = np.zeros(max_weeks)
    for k in range(1, max_weeks+1):
        left = max(0.0, 7*(k-1))
        right = 7*(k+1)
        integrand = lambda u: (1.0 - abs(u - 7*k)/7.0) * g(u) if abs(u - 7*k) <= 7 else 0.0
        val, _ = quad(integrand, left, right, epsabs=1e-9, epsrel=1e-9)
        w[k-1] = val
    # ogi gittins method
    w[0] = 1.0 - w[1:].sum()
    w /= w.sum()
    return w

# --- compute log-likelihood P([I1,I2,I3]|R) under Poisson renewal ---
def log_likelihood_I(I_seq, R, w):
    I = np.asarray(I_seq, dtype=float)
    T = len(I)
    loglike = 0.0
    for t in range(1, T):
        max_s = min(t, len(w))
        infectious = sum(I[t-s] * w[s-1] for s in range(1, max_s+1))
        lam = R * infectious
        if lam <= 0:
            if I[t] == 0:
                continue
            return -np.inf
        # Poisson log pmf: I[t]*log(lam) - lam - log(I[t]!)
        # use direct expression; factorial term is constant with respect to R but keep it for exact posterior
        loglike += I[t] * np.log(lam) - lam - np.log(math.factorial(int(I[t])))
    return loglike




# --- extinction probability q (smallest nonneg root of q = exp(R(q-1))) ---
def extinction_q(R):
    if R <= 1:
        return 1.0
    z = -R * np.exp(-R)
    return -lambertw(z).real / R


# --- conditional PMO given R and three-week counts (equation 4) ---
# def PMO_given_R_three_weeks(I_seq, R, w):
#     q = extinction_q(R)
#     w1 = w[0] if len(w) > 0 else 0.0
#     w2 = w[1] if len(w) > 1 else 0.0
#     term1 = np.exp(R * (1.0 - w1 - w2) * (q - 1.0))
#     term2 = np.exp(R * (1.0 - w1) * (q - 1.0))
#     term3 = q
#     none_prob = (term1 ** I_seq[0]) * (term2 ** I_seq[1]) * (term3 ** I_seq[2])
#     return 1.0 - none_prob
def PMO_given_R_general(I_seq, R, w):
    """
    For observed weeks t = 1..T with counts I_seq[t-1],
    remaining infectiousness fraction for a case in week k is:
      remaining_k = 1 - sum_{s=1}^{T-k} w_s
    (sum only up to len(w)).
    Then none_prob = prod_k exp(R * remaining_k * (q-1))^{I_k}
    which reduces to exp( (q-1) * R * sum_k I_k * remaining_k ).
    PMO = 1 - none_prob.
    """
    I = np.asarray(I_seq, dtype=float)
    T = len(I)
    q = extinction_q(R)
    # precompute cumulative sums of w
    # cumulative_w_s = sum_{s=1..m} w_s (1-based s)
    cumulative_w = np.cumsum(w)
    total = 0.0
    for k in range(1, T + 1):
        # number of observed future weeks after week k within our window
        m = T - k
        if m <= 0:
            sum_w = 0.0
        else:
            # sum min(m, len(w)) of w[0:m]
            m_use = min(m, len(w))
            sum_w = np.sum(w[:m_use])
        remaining = 1.0 - sum_w
        total += I[k - 1] * remaining
    # log none prob = R * (q-1) * total
    log_none = R * (q - 1.0) * total
    # guard against under/overflow
    if log_none < -700:
        none_prob = 0.0
    elif log_none > 700:
        none_prob = float('inf')  # theoretically impossible, handle below
    else:
        none_prob = math.exp(log_none)
    # If none_prob >=1 due to numerical issues clamp to 1
    if not math.isfinite(none_prob) or none_prob > 1.0:
        none_prob = 1.0
    pmor = 1.0 - none_prob
    # clamp
    pmor = max(0.0, min(1.0, pmor))
    return pmor
def PMO_general(I_seq, w=None, nR=2001, R_min=R_MIN, R_max=R_MAX):
    if w is None:
        w = weekly_w(max_weeks=max(40, len(I_seq) + 5))
    # ensure w is long enough
    if len(w) < len(I_seq):
        w = weekly_w(max_weeks=len(I_seq) + 10)
    R_grid = np.linspace(R_min, R_max, nR)
    delta = R_grid[1] - R_grid[0]
    loglikes = np.array([log_likelihood_I(I_seq, R, w) for R in R_grid])
    if not np.any(np.isfinite(loglikes)):
        return 0.0, R_grid, np.full_like(R_grid, -np.inf), None
    logpost_unnorm = loglikes  # uniform prior
    logpost_norm = logpost_unnorm - logsumexp(logpost_unnorm + np.log(delta))
    post = np.exp(logpost_norm)
    pmogivenR = np.array([PMO_given_R_general(I_seq, R, w) for R in R_grid])
    PMO_val = np.sum(pmogivenR * post * delta)
    return PMO_val, R_grid, loglikes, pmogivenR

# --- Example usage ---
if __name__ == "__main__":
    # ask user how many weeks they want to provide
    while True:
        try:
            n_weeks = int(input("How many weeks of observed early data will you enter? (integer >=1): "))
            if n_weeks < 1:
                print("Please enter a positive integer.")
                continue
            break
        except ValueError:
            print("Please enter an integer.")

    print(f"Enter the observed counts for weeks 1..{n_weeks}. Press enter after each integer.")
    I_seq = []
    for k in range(1, n_weeks + 1):
        while True:
            try:
                v = int(input(f"I_{k}: "))
                if v < 0:
                    print("Please enter a non-negative integer.")
                    continue
                I_seq.append(v)
                break
            except ValueError:
                print("Please enter an integer.")

    # compute w with adequate length
    w = weekly_w(max_weeks=max(50, n_weeks + 10))
    #print_weekly_weights(w, n_print=min(10, len(w)))

    # compute PMO
    PMO_val, R_grid, loglikes, pmogivenR = PMO_general(I_seq, w=w, nR=4001)
    print("\nObserved sequence:", I_seq)
    print("Estimated PMO (integrated over R in [{},{}]) = {:.3f}".format(R_MIN, R_MAX, PMO_val))

# --- integrate over R on a fine linear grid [0,5] with uniform prior ---
# def PMO(I1, I2, I3, w=None, nR=2001):
#     if w is None:
#         w = weekly_w(max_weeks=40)
#     I_seq = [int(I1), int(I2), int(I3)]
#     R_grid = np.linspace(R_MIN, R_MAX, nR)
#     delta = R_grid[1] - R_grid[0]
#     loglikes = np.array([log_likelihood_I(I_seq, R, w) for R in R_grid])
#     # handle case where all loglikes are -inf
#     if not np.any(np.isfinite(loglikes)):
#         return 0.0
#     # posterior (log) up to const; prior is uniform so ignored in log space
#     logpost_unnorm = loglikes
#     # Normalize posterior accounting for grid spacing (delta)
#     logpost_norm = logpost_unnorm - logsumexp(logpost_unnorm + np.log(delta))
#     post = np.exp(logpost_norm)
#     # compute PMO(R) on grid and integrate
#     pmogivenR = np.array([PMO_given_R_three_weeks(I_seq, R, w) for R in R_grid])
#     PMO_val = np.sum(pmogivenR * post * delta)
#     return PMO_val

# --- Example usage ---
# if __name__ == "__main__":
#     w = weekly_w(max_weeks=40)
#     I1 = int(input("I1: "))
#     I2 = int(input("I2: "))
#     I3 = int(input("I3: "))
#     pmo = PMO(I1, I2, I3, w=w, nR=2001)
#     print("PMO = {:.6f}".format(pmo))
# print("Weekly serial interval weights (w_s):")
# for i, val in enumerate(w[:10], start=1):
#     print(f"w{i} = {val:.6f}")

#Was returning PMO=0 for all inputs, used this to see why:
def diagnostics(I_seq, w, R_grid, loglikes, post, pmogivenR):
    I1, I2, I3 = I_seq
    delta = R_grid[1] - R_grid[0]

    # 1) summary of w
    print("w1, w2, w3 =", w[0], w[1], w[2])
    print("sum(w) =", w.sum())

    # 2) likelihood / posterior summary
    finite_mask = np.isfinite(loglikes)
    print("loglikes finite fraction:", finite_mask.mean())
    if not np.any(finite_mask):
        print("All log-likelihoods are -inf; nothing fits. Stop here.")
        return

    max_idx = np.nanargmax(loglikes)
    R_map = R_grid[max_idx]
    print(f"MAP (by loglike) at R = {R_map:.6f} (loglike = {loglikes[max_idx]:.3f})")
    # posterior mass near extremes
    print("Posterior mass at R grid endpoints:",
          post[0]*delta, post[-1]*delta,
          "sum(post*delta)=", (post*delta).sum())

    # 3) show lambda_t at MAP R
    def compute_lambdas(I_seq, R, w):
        I = np.array(I_seq, dtype=float)
        T = len(I)
        lambdas = []
        for t in range(1, T):
            max_s = min(t, len(w))
            infectious = sum(I[t-s] * w[s-1] for s in range(1, max_s+1))
            lambdas.append(R * infectious)
        return lambdas

    lambdas_map = compute_lambdas(I_seq, R_map, w)
    print("lambda_t at MAP R:", lambdas_map, "observed I (t=2..):", I_seq[1:])

    # 4) print pmogivenR at a few R values
    check_Rs = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    print("PMO|R at sample R values:")
    for r in check_Rs:
        # find nearest grid idx if using grid
        idx = (np.abs(R_grid - r)).argmin() if R_grid is not None else None
        val = pmogivenR[idx] if idx is not None else None
        print(f" R={r:>4}  PMO|R={val}")

    from scipy.optimize import root_scalar
    def extinction_q(R):
        if R <= 1.0:
            return 1.0
        f = lambda q: q - math.exp(R*(q - 1.0))
        sol = root_scalar(f, bracket=[0.0, 1.0], method='bisect')
        return sol.root

    log_none_probs = np.full_like(R_grid, -np.inf, dtype=float)
    for i,R in enumerate(R_grid):
        if not np.isfinite(loglikes[i]):
            continue
        q = extinction_q(R)
        w1 = w[0]; w2 = w[1]
        log_term1 = R * (1.0 - w1 - w2) * (q - 1.0)
        log_term2 = R * (1.0 - w1) * (q - 1.0)
        log_term3 = math.log(q) if q>0 else -np.inf
        log_none = I1*log_term1 + I2*log_term2 + I3*log_term3
        log_none_probs[i] = log_none

    # show range of log_none (none_prob = exp(log_none))
    finite_none = np.isfinite(log_none_probs)
    if np.any(finite_none):
        print("log-none-prob range:", log_none_probs[finite_none].min(), "(min) to", log_none_probs[finite_none].max(), "(max)")
        # if the max is near 0, none_prob close to 1 -> PMO tiny
        # show where none_prob is near 1
        top_indices = np.argsort(log_none_probs)[-5:]
        print("Top 5 R (by log-none) with their log-none and PMO|R:")
        for idx in top_indices:
            R = R_grid[idx]
            ln = log_none_probs[idx]
            none = math.exp(ln) if ln > -700 else 0.0
            pmor = 1.0 - none
            print(f" R={R:.6f} log-none={ln:.4f} none_prob={none:.6g}  PMO|R={pmor:.6g}")
    else:
        print("No finite none_prob computed.")

# Example invocation:
#diagnostics([2,4,6], w, R_grid, loglikes, post, pmogivenR)
