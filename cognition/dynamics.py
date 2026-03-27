from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Sequence, Tuple


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_inhibition(activations: Dict[str, float]) -> float:
    return float(sum(max(0.0, v) for v in activations.values()))


def activation_propagation(
    node_ids: Sequence[str],
    activation: Dict[str, float],
    incoming: Dict[str, List[Tuple[str, float]]],
    value_field: Dict[str, float],
    mission_proj: Dict[str, float],
    decay_lambda: float,
    beta: float,
    noise_std: float,
    gamma: float,
    kappa: float,
    inhibition: float,
    rng: random.Random,
) -> Dict[str, float]:
    nxt: Dict[str, float] = {}
    for nid in node_ids:
        transmission = 0.0
        for src, w in incoming.get(nid, []):
            transmission += w * math.tanh(beta * activation.get(src, 0.0))
        val = (
            (1.0 - decay_lambda) * activation.get(nid, 0.0)
            + transmission
            + rng.gauss(0.0, noise_std)
            + gamma * mission_proj.get(nid, value_field.get(nid, 0.0))
            - kappa * inhibition
        )
        nxt[nid] = clamp(val, -1.0, 1.0)
    return nxt


def attention_update(activation: Dict[str, float], value: Dict[str, float], uncertainty: Dict[str, float], eta_a: float, eta_v: float, eta_u: float) -> Dict[str, float]:
    raw = {}
    for nid in activation:
        s = eta_a * activation.get(nid, 0.0) + eta_v * value.get(nid, 0.0) - eta_u * uncertainty.get(nid, 0.5)
        raw[nid] = math.exp(max(-20.0, min(20.0, s)))
    z = sum(raw.values())
    if z <= 1e-12 and raw:
        return {k: 1.0 / len(raw) for k in raw}
    return {k: v / max(z, 1e-12) for k, v in raw.items()}


def value_update(value: Dict[str, float], mission_proj: Dict[str, float], external_signal: Dict[str, float], lambda_v: float, beta_m: float, beta_s: float) -> Dict[str, float]:
    out = {}
    for nid in value:
        out[nid] = clamp((1.0 - lambda_v) * value.get(nid, 0.0) + beta_m * mission_proj.get(nid, 0.0) + beta_s * external_signal.get(nid, 0.0), -1.0, 1.0)
    return out


def uncertainty_update(uncertainty: Dict[str, float], contradiction_signal: Dict[str, float], evidence_signal: Dict[str, float], test_signal: Dict[str, float], rho_c: float, rho_e: float, rho_t: float) -> Dict[str, float]:
    out = {}
    for nid in uncertainty:
        u = uncertainty.get(nid, 0.5) + rho_c * contradiction_signal.get(nid, 0.0) - rho_e * evidence_signal.get(nid, 0.0) - rho_t * test_signal.get(nid, 0.0)
        out[nid] = clamp(u, 0.0, 1.0)
    return out


def stability_update(stability: Dict[str, float], attention: Dict[str, float], activation: Dict[str, float], uncertainty: Dict[str, float], delta: float, rho_sigma: float, rho_u: float) -> Dict[str, float]:
    out = {}
    for nid in stability:
        s = (1.0 - delta) * stability.get(nid, 0.0) + rho_sigma * attention.get(nid, 0.0) * abs(activation.get(nid, 0.0)) - rho_u * uncertainty.get(nid, 0.5)
        out[nid] = max(0.0, s)
    return out


def plasticity_update(weight: float, a_i: float, a_j: float, alpha: float, mu: float) -> float:
    return max(0.0, min(2.0, weight + alpha * a_i * a_j - mu * weight))


def confidence_update(confidence: float, support_signal: float, contradiction_signal: float, xi_s: float, xi_k: float) -> float:
    return clamp(confidence + xi_s * support_signal - xi_k * contradiction_signal, 0.0, 1.0)


def novelty_signal(current: Dict[str, float], previous: Dict[str, float]) -> float:
    keys = set(current) | set(previous)
    return math.sqrt(sum((current.get(k, 0.0) - previous.get(k, 0.0)) ** 2 for k in keys))


def contradiction_tension(pairs: Iterable[Tuple[str, str, float]], activation: Dict[str, float]) -> float:
    return sum(activation.get(i, 0.0) * activation.get(j, 0.0) * conf for i, j, conf in pairs)


def uncertainty_mass(uncertainty: Dict[str, float]) -> float:
    return float(sum(uncertainty.values()))
