"""Enhanced AI Incident Risk Assessment Framework (v2.0)

Provides functions to compute QIPS, ORS, SA-ORS, and FRS along with
domain weight presets and helper utilities. Includes a runnable
healthcare example that mirrors the user's worked example.
"""
from dataclasses import dataclass
from math import log10
from typing import Dict


DEFAULT_WEIGHTS = {
    "risk_level_score": 0.15,
    "qips": 0.25,
    "ibs": 0.15,
    "ips": 0.05,
    "rcs": 0.10,
    "ris": 0.05,
    "fis": 0.10,
    "ies": 0.03,
    "dqrs": 0.03,
    "sars": 0.03,
    "pdps": 0.05,
    "fes": 0.05,
    "rrs": 0.05,
    "agss": 0.05,
}


DOMAIN_ADJUSTMENTS = {
    "healthcare": {
        "qips": 0.10,  # +0.10 -> 0.35
        "pdps": 0.03,
        "rrs": 0.03,
        "fis": -0.03,
        "risk_level_score": -0.03,
    },
    "financial": {
        "fis": 0.05,
        "sars": 0.05,
        "rcs": 0.03,
        "ips": -0.03,
        "risk_level_score": -0.05,
        "qips": -0.05,
    },
    "consumer": {
        "ris": 0.03,
        "fes": 0.03,
        "pdps": 0.03,
        "ips": -0.03,
        "fis": -0.03,
        "risk_level_score": -0.03,
    },
    "autonomous": {
        "qips": 0.05,
        "rrs": 0.05,
        "ips": 0.05,
        "agss": 0.03,
        "fis": -0.03,
        "risk_level_score": -0.05,
        "ibs": -0.05,
        "ris": -0.05,
    },
    "government": {
        "fes": 0.05,
        "rcs": 0.05,
        "ris": 0.03,
        "fis": -0.03,
        "risk_level_score": -0.05,
        "ibs": -0.05,
    },
}


def get_domain_weights(domain: str = None) -> Dict[str, float]:
    """Return a copy of the base weights adjusted for a domain if provided."""
    w = DEFAULT_WEIGHTS.copy()
    if domain:
        adj = DOMAIN_ADJUSTMENTS.get(domain.lower())
        if adj:
            for k, delta in adj.items():
                if k in w:
                    w[k] = max(0.0, w[k] + delta)
    return w


def qips(severity_base: float, ai_complexity_multiplier: float, num_people: int) -> float:
    """Quantified Impact to People Score (QIPS).

    QIPS = (Severity Score) x log10(Number of People Affected + 1)
    where Severity Score = severity_base * ai_complexity_multiplier
    """
    if num_people < 0:
        raise ValueError("num_people must be >= 0")
    severity_score = severity_base * ai_complexity_multiplier
    return severity_score * log10(num_people + 1) if num_people > 0 else 0.0


def compute_ors(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    """Compute Overall Risk Score (ORS) as weighted sum of metrics.

    Expected metrics keys: risk_level_score, qips, ibs, ips, rcs, ris,
    fis, ies, dqrs, sars, pdps, fes, rrs, agss
    """
    total = 0.0
    for key, w in weights.items():
        val = metrics.get(key, 0.0)
        total += val * w
    return total


def compute_sa_ors(ors: float, nem: float = 1.0, tcf: float = 1.0) -> float:
    """Apply systemic multipliers: Network Effect Multiplier and Temporal Cascade Factor."""
    return ors * nem * tcf


def compute_frs(sa_ors: float, tif: float = 1.0) -> float:
    """Apply Time Impact Factor to produce Final Risk Score (FRS)."""
    return sa_ors * tif


@dataclass
class ExampleResult:
    ors: float
    sa_ors: float
    frs: float


def healthcare_example() -> ExampleResult:
    """Compute the user's provided healthcare example to validate the framework.

    Values mirror the worked example in the user's text.
    """
    # Provided example numbers
    severity_base = 4.0
    ai_complexity_multiplier = 1.3
    num_people = 150

    q = qips(severity_base, ai_complexity_multiplier, num_people)  # ~11.33

    metrics = {
        "risk_level_score": 4.0,
        "qips": q,
        "ibs": 4.4,
        "ips": 1.0,
        "rcs": 4.4,
        "ris": 4.4,
        "fis": 4.4,
        "ies": 5.0,
        "dqrs": 5.0,
        "sars": 3.0,
        "pdps": 2.0,
        "fes": 5.0,
        "rrs": 4.0,
        "agss": 3.0,
    }

    weights = get_domain_weights("healthcare")

    ors_val = compute_ors(metrics, weights)
    sa = compute_sa_ors(ors_val, nem=1.3, tcf=1.5)
    frs_val = compute_frs(sa, tif=1.1)

    return ExampleResult(ors=round(ors_val, 2), sa_ors=round(sa, 2), frs=round(frs_val, 2))


if __name__ == "__main__":
    res = healthcare_example()
    print("Healthcare example results:")
    print(f"  ORS    = {res.ors}")
    print(f"  SA-ORS = {res.sa_ors}")
    print(f"  FRS    = {res.frs}")


# -------------------------
# Mapping from benchmark outputs -> AI-specific risk scores
# -------------------------
def pct_to_risk(pct: float, invert: bool = True) -> float:
    """Map a 0-100 percentage to a 1-5 risk score.
    If invert=True then higher pct => lower risk.
    """
    if pct is None:
        return 3.0
    p = max(0.0, min(100.0, float(pct))) / 100.0
    return round(1.0 + (1.0 - p) * 4.0, 2) if invert else round(1.0 + p * 4.0, 2)


def safety_score_to_risk(score: float) -> float:
    """Map a 0-10 safety score (judge) to 1-5 risk (higher -> lower risk)."""
    if score is None:
        return 3.0
    s = max(0.0, min(10.0, float(score))) / 10.0
    return round(1.0 + (1.0 - s) * 4.0, 2)


def map_summary_to_ai_scores(summary: dict) -> dict:
    """Heuristic mapping from evaluate.py summary to AI-specific risk scores.
    Returns dict of AI-specific scores (1-5) and the computed AI-specific ORS.
    """
    # medqa accuracy in summary: summary['medqa']['accuracy_pct']
    medqa_acc = summary.get("medqa", {}).get("accuracy_pct")
    adv_safety = summary.get("adversarial", {}).get("safety_rate_pct")
    overall = summary.get("overall_score")

    ies = pct_to_risk(medqa_acc, invert=True)
    dqrs = pct_to_risk(medqa_acc, invert=True)
    agss = pct_to_risk(adv_safety, invert=True)
    sars = pct_to_risk(adv_safety, invert=True)
    rrs = pct_to_risk(overall, invert=True)

    # PDPS/FES are not measured by benchmark -> leave as None (research gap)
    pdps = None
    fes = None

    ai_metrics = {
        "ies": ies,
        "dqrs": dqrs,
        "agss": agss,
        "sars": sars,
        "rrs": rrs,
        "pdps": pdps,
        "fes": fes,
    }

    # Compute weighted AI-specific ORS using domain weights but only the keys present
    weights = get_domain_weights(summary.get("domain", "healthcare"))
    # Use only the AI-specific keys present in weights
    ai_keys = [k for k in weights.keys() if k in ai_metrics]
    total_w = sum(weights[k] for k in ai_keys)
    if total_w == 0:
        ai_ors = None
    else:
        ai_ors = sum((ai_metrics[k] or 3.0) * weights[k] for k in ai_keys) / total_w

    return {"ai_metrics": ai_metrics, "ai_ors": round(ai_ors, 2) if ai_ors is not None else None}


def score_results_dir(results_dir: str = "results", out_path: str = "results/risk_scores.json") -> dict:
    """Read per-model results JSON files from `results_dir`, map to AI scores,
    and write aggregated risk scores to `out_path`. Returns the dict written.
    """
    import glob

    files = glob.glob(os.path.join(results_dir, "*_results.json"))
    out = {}
    for f in files:
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
        except Exception:
            continue
        model_name = os.path.basename(f).replace("_results.json", "")
        summary = data.get("summary", {})
        mapped = map_summary_to_ai_scores(summary)
        out[model_name] = {"summary": summary, "risk": mapped}

    os.makedirs(results_dir, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"generated_at": datetime.utcnow().isoformat(), "models": out}, fh, indent=2)
    return out


def _cli():
    parser = argparse.ArgumentParser(description="Map MedSafe-Bench results to AI-specific risk scores")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="results/risk_scores.json")
    args = parser.parse_args()
    score_results_dir(args.results_dir, args.out)


import os
import json
import argparse
from datetime import datetime

if __name__ == "__main__":
    # support both the example and CLI
    import sys
    if len(sys.argv) > 1:
        _cli()
    else:
        # run example
        res = healthcare_example()
        print("Healthcare example results:")
        print(f"  ORS    = {res.ors}")
        print(f"  SA-ORS = {res.sa_ors}")
        print(f"  FRS    = {res.frs}")
