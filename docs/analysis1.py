import io
import math
import base64

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import gamma
from scipy.integrate import quad
from scipy.special import logsumexp, lambertw

from js import document
from pyodide.ffi import create_proxy


MEAN_SI_DAYS = 15.3
SD_SI_DAYS = 9.3

R_MIN = 0.0
R_MAX = 10.0


# -----------------------------
# DOM helpers
# -----------------------------

def by_id(element_id):
    return document.getElementById(element_id)


def set_text(element_id, text):
    el = by_id(element_id)
    if el is not None:
        el.textContent = text


def show_plot_from_bytes(png_bytes):
    encoded = base64.b64encode(png_bytes).decode("utf-8")

    image = by_id("pmo-plot-image")
    placeholder = by_id("pmo-plot-placeholder")

    if image is not None:
        image.src = f"data:image/png;base64,{encoded}"
        image.style.display = "block"

    if placeholder is not None:
        placeholder.style.display = "none"


def clear_plot():
    image = by_id("pmo-plot-image")
    placeholder = by_id("pmo-plot-placeholder")

    if image is not None:
        image.removeAttribute("src")
        image.style.display = "none"

    if placeholder is not None:
        placeholder.style.display = "block"
        placeholder.textContent = "Plot will appear here after computation."


def set_button_busy(is_busy):
    button = by_id("compute-pmo-button")
    if button is not None:
        button.disabled = is_busy
        button.textContent = "Computing..." if is_busy else "Compute PMO"


# -----------------------------
# Model code
# -----------------------------

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
            if abs(u - 7 * k) <= 7:
                return (1.0 - abs(u - 7 * k) / 7.0) * g(u)
            return 0.0

        val, _ = quad(integrand, left, right, epsabs=1e-9, epsrel=1e-9)
        w[k - 1] = val

    if len(w) > 1:
        w[0] = max(0.0, 1.0 - w[1:].sum())

    total = w.sum()
    if total > 0:
        w /= total

    return w


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

        loglike += I[t] * np.log(lam) - lam - math.lgamma(int(I[t]) + 1)

    return loglike


def extinction_q(R):
    if R <= 1:
        return 1.0
    z = -R * np.exp(-R)
    return float((-lambertw(z).real / R))


def PMO_given_R_general(I_seq, R, w):
    I = np.asarray(I_seq, dtype=float)
    T = len(I)

    q = extinction_q(R)
    total = 0.0

    for k in range(1, T + 1):
        m = T - k
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

    return min(1.0, max(0.0, pmor))


def PMO_general(I_seq, w=None, nR=2001, R_min=R_MIN, R_max=R_MAX):
    if w is None:
        w = weekly_w(max_weeks=max(40, len(I_seq) + 5))

    if len(w) < len(I_seq):
        w = weekly_w(max_weeks=len(I_seq) + 10)

    R_grid = np.linspace(R_min, R_max, nR)
    delta = R_grid[1] - R_grid[0]

    loglikes = np.array([log_likelihood_I(I_seq, R, w) for R in R_grid])

    if not np.any(np.isfinite(loglikes)):
        zeros = np.zeros_like(R_grid)
        return 0.0, R_grid, loglikes, zeros, zeros, w

    logpost_norm = loglikes - logsumexp(loglikes + np.log(delta))
    posterior = np.exp(logpost_norm)

    pmogivenR = np.array([PMO_given_R_general(I_seq, R, w) for R in R_grid])
    PMO_val = np.sum(pmogivenR * posterior * delta)

    return PMO_val, R_grid, loglikes, posterior, pmogivenR, w


# -----------------------------
# Input parsing
# -----------------------------

def parse_counts(text_counts, n_weeks):
    if text_counts.strip() == "":
        return None, "Please enter observed counts."

    cleaned = (
        text_counts
        .replace(";", ",")
        .replace("\n", ",")
        .replace(" ", ",")
    )

    parts = [p.strip() for p in cleaned.split(",") if p.strip() != ""]

    try:
        counts = [int(float(x)) for x in parts]
    except Exception:
        return None, "Counts must be integers, for example: 1,2,0"

    if any(v < 0 for v in counts):
        return None, "Counts must be non-negative integers."

    if len(counts) != n_weeks:
        return None, f"Number of counts ({len(counts)}) does not match number of weeks ({n_weeks})."

    return counts, None


# -----------------------------
# Plot creation
# -----------------------------

def make_plot_png(R_grid, pmogivenR, posterior):
    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.plot(R_grid, pmogivenR, label="PMO|R")
    ax1.set_xlabel("R")
    ax1.set_ylabel("PMO|R")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.fill_between(R_grid, 0, posterior, alpha=0.3)
    ax2.set_ylabel("Posterior density")

    ax1.legend(loc="upper left")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    buf.seek(0)
    return buf.read()


# -----------------------------
# UI actions
# -----------------------------

def update_n_weeks_value(event=None):
    slider = by_id("n-weeks-slider")
    if slider is not None:
        set_text("n-weeks-value", slider.value)


def update_r_grid_value(event=None):
    slider = by_id("r-grid-slider")
    if slider is not None:
        set_text("r-grid-value", slider.value)


def compute_pmo(event=None):
    if event is not None:
        event.preventDefault()

    clear_plot()
    set_button_busy(True)

    try:
        n_weeks = int(float(by_id("n-weeks-slider").value))
        nR = int(float(by_id("r-grid-slider").value))
        text_counts = by_id("observed-counts-input").value

        counts, err = parse_counts(text_counts, n_weeks)
        if err:
            set_text("pmo-result-output", err)
            return

        set_text(
            "pmo-result-output",
            "Computing PMO...\n"
            f"Observed sequence: {counts}\n"
            f"R-grid points: {nR}"
        )

        w = weekly_w(max_weeks=max(50, n_weeks + 10))

        PMO_val, R_grid, loglikes, posterior, pmogivenR, w = PMO_general(
            counts,
            w=w,
            nR=nR,
            R_min=R_MIN,
            R_max=R_MAX,
        )

        result_text = (
            f"Observed sequence: {counts}\n"
            f"Estimated PMO integrated over R in [{R_MIN:g}, {R_MAX:g}] = {PMO_val:.6f}\n"
            f"Number of weeks: {n_weeks}\n"
            f"R-grid points: {nR}"
        )
        set_text("pmo-result-output", result_text)

        if np.any(np.isfinite(loglikes)):
            png_bytes = make_plot_png(R_grid, pmogivenR, posterior)
            show_plot_from_bytes(png_bytes)
        else:
            set_text("pmo-result-output", result_text + "\n\nNo finite likelihood values were found.")

    except Exception as exc:
        clear_plot()
        set_text("pmo-result-output", f"An error occurred while computing PMO:\n{type(exc).__name__}: {exc}")

    finally:
        set_button_busy(False)


# -----------------------------
# Bind events
# -----------------------------

submit_proxy = create_proxy(compute_pmo)
weeks_proxy = create_proxy(update_n_weeks_value)
rgrid_proxy = create_proxy(update_r_grid_value)


def bind_events():
    form = by_id("pmo-form")
    n_weeks_slider = by_id("n-weeks-slider")
    r_grid_slider = by_id("r-grid-slider")

    if form is not None:
        form.addEventListener("submit", submit_proxy)

    if n_weeks_slider is not None:
        n_weeks_slider.addEventListener("input", weeks_proxy)

    if r_grid_slider is not None:
        r_grid_slider.addEventListener("input", rgrid_proxy)

    update_n_weeks_value()
    update_r_grid_value()
    set_text("pmo-result-output", "Ready.")


bind_events()
