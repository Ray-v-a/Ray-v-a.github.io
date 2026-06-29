
# Converted to a Hugging Face Space friendly Gradio app.
# Enter observed weekly counts as comma-separated integers (e.g. "2,0,1,3").

import io
import math
import numpy as np
from scipy.stats import gamma
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.special import lambertw
import gradio as gr
import matplotlib.pyplot as plt

# --- parameters from paper ---
MEAN_SI_DAYS = 15.3
SD_SI_DAYS = 9.3
R_MIN, R_MAX = 0.0, 10.0


# --- weekly discretisation of continuous gamma SI using triangular kernel ---
def weekly_w(max_weeks=50, mean=MEAN_SI_DAYS, sd=SD_SI_DAYS):
    shape = (mean / sd) ** 2
    scale = sd ** 2 / mean

    def g(x):
        return gamma.pdf(x, a=shape, scale=scale)

    w = np.zeros(max_weeks)
    for k in range(1, max_weeks + 1):
        left = max(0.0, 7 * (k - 1))
        right = 7 * (k + 1)

        def integrand(u):
            return (1.0 - abs(u - 7 * k) / 7.0) * g(u) if abs(u - 7 * k) <= 7 else 0.0

        val, _ = quad(integrand, left, right, epsabs=1e-9, epsrel=1e-9)
        w[k - 1] = val

    # adjust first bin so sum(w)=1 (Gittins-like adjustment)
    w[0] = max(0.0, 1.0 - w[1:].sum())
    w /= w.sum()
    return w


# --- compute log-likelihood P([I1,I2,...]|R) under Poisson renewal ---
def log_likelihood_I(I_seq, R, w):
    I = np.asarray(I_seq, dtype=float)
    T = len(I)
    loglike = 0.0
    for t in range(1, T):
        max_s = min(t, len(w))
        infectious = sum(I[t - s] * w[s - 1] for s in range(1, max_s + 1))
        lam = R * infectious
        if lam <= 0:
            if I[t] == 0:
                continue
            return -np.inf
        # Poisson log pmf
        # factorial part is constant in R but we keep it for exactness
        loglike += I[t] * np.log(lam) - lam - math.lgamma(int(I[t]) + 1)
    return loglike


# --- extinction probability q (smallest nonneg root of q = exp(R(q-1))) ---
def extinction_q(R):
    if R <= 1:
        return 1.0
    z = -R * np.exp(-R)
    # lambertw may return complex with tiny imag parts; take real
    return -lambertw(z).real / R


# --- conditional PMO given R and observed I sequence ---
def PMO_given_R_general(I_seq, R, w):
    I = np.asarray(I_seq, dtype=float)
    T = len(I)
    q = extinction_q(R)

    total = 0.0
    for k in range(1, T + 1):
        m = T - k  # future weeks within observation window after week k
        if m <= 0:
            sum_w = 0.0
        else:
            m_use = min(m, len(w))
            sum_w = np.sum(w[:m_use])
        remaining = 1.0 - sum_w
        total += I[k - 1] * remaining

    log_none = R * (q - 1.0) * total

    if log_none < -700:
        none_prob = 0.0
    elif log_none > 700:
        none_prob = 1.0
    else:
        none_prob = math.exp(log_none)

    none_prob = min(1.0, max(0.0, none_prob))
    pmor = 1.0 - none_prob
    pmor = min(1.0, max(0.0, pmor))
    return pmor


def PMO_general(I_seq, w=None, nR=2001, R_min=R_MIN, R_max=R_MAX):
    if w is None:
        w = weekly_w(max_weeks=max(40, len(I_seq) + 5))
    if len(w) < len(I_seq):
        w = weekly_w(max_weeks=len(I_seq) + 10)

    R_grid = np.linspace(R_min, R_max, nR)
    delta = R_grid[1] - R_grid[0]
    loglikes = np.array([log_likelihood_I(I_seq, R, w) for R in R_grid])
    if not np.any(np.isfinite(loglikes)):
        # nothing fits data: return zeros and diagnostics placeholders
        return 0.0, R_grid, loglikes, np.zeros_like(R_grid), w

    logpost_unnorm = loglikes  # uniform prior
    logpost_norm = logpost_unnorm - logsumexp(logpost_unnorm + np.log(delta))
    post = np.exp(logpost_norm)
    pmogivenR = np.array([PMO_given_R_general(I_seq, R, w) for R in R_grid])
    PMO_val = np.sum(pmogivenR * post * delta)
    return PMO_val, R_grid, loglikes, pmogivenR, w


# ---------- Gradio interface functions ----------

def parse_counts(text_counts, n_weeks):
    """Parse comma/space separated counts and validate length."""
    if text_counts.strip() == "":
        return None, "Please enter observed counts (comma separated)."
    # allow commas, spaces, semicolons
    parts = [p.strip() for p in text_counts.replace(";", ",").replace("\n", ",").split(",") if p.strip() != ""]
    try:
        ints = [int(float(x)) for x in parts]  # allow "2.0"
    except Exception:
        return None, "Counts must be integers (comma separated)."
    if any(v < 0 for v in ints):
        return None, "Counts must be non-negative integers."
    if len(ints) != n_weeks:
        return None, f"Number of counts ({len(ints)}) does not match n_weeks ({n_weeks})."
    return ints, None


def run_pmo(text_counts, n_weeks, nR=2001, plot=True):
    parsed, err = parse_counts(text_counts, n_weeks)
    if err:
        return err, None
    I_seq = parsed
    w = weekly_w(max_weeks=max(50, n_weeks + 10))
    PMO_val, R_grid, loglikes, pmogivenR, w = PMO_general(I_seq, w=w, nR=nR, R_min=0.0, R_max=10.0)
    out_text = f"Observed sequence: {I_seq}\nEstimated PMO (integrated over R in [0,5]) = {PMO_val:.6f}"

    plot_img = None
    if plot:
        # create a figure showing posterior-weighted PMO(R) and posterior
        delta = R_grid[1] - R_grid[0]
        finite_mask = np.isfinite(loglikes)
        if np.any(finite_mask):
            logpost_unnorm = loglikes
            logpost_norm = logpost_unnorm - logsumexp(logpost_unnorm + np.log(delta))
            post = np.exp(logpost_norm)

            fig, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(R_grid, pmogivenR, label="PMO|R")
            ax1.set_xlabel("R")
            ax1.set_ylabel("PMO|R")
            ax1.set_ylim(0, 1.05)
            ax2 = ax1.twinx()
            ax2.fill_between(R_grid, 0, post, alpha=0.3)
            ax2.set_ylabel("Posterior density (unnormalised w.r.t. plot scale)")
            ax1.legend(loc='upper left')
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            plt.close(fig)
            plot_img = buf.read()

    return out_text, plot_img


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# PMO calculator — weekly early data")
    gr.Markdown(
        "Enter how many weeks you will provide and paste the observed counts as comma-separated integers.\n"
        "Example: `1,2,0` for weeks 1..3."
    )

    with gr.Row():
        n_weeks_input = gr.Slider(1, 12, value=3, step=1, label="Number of weeks (n_weeks)")
        nR_input = gr.Slider(501, 8001, value=4001, step=500, label="R-grid points (increase for smoother integration)")

    counts_input = gr.Textbox(lines=2, value="1,2,0", label="Observed counts (comma separated)")
    run_btn = gr.Button("Compute PMO")

    output_text = gr.Textbox(label="Result / diagnostics")
    output_image = gr.Image(label="PMO|R and posterior", type="pil")

    def on_click(counts, nw, nr):
        text, img = run_pmo(counts, int(nw), nR=int(nr), plot=True)
        if img is not None:
            # return text and PIL image built from bytes
            from PIL import Image
            import io as _io
            pil = Image.open(_io.BytesIO(img))
            return text, pil
        else:
            return text, None

    run_btn.click(on_click, inputs=[counts_input, n_weeks_input, nR_input], outputs=[output_text, output_image])

    gr.Markdown("**Notes:**\n- This app integrates PMO over a grid of R with a uniform prior.\n- If you need a narrower R-range, edit `R_min` / `R_max` in the code or modify `PMO_general` call.")

if __name__ == "__main__":
    demo.launch()

