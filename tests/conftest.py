"""Gemeinsame Fixtures: der Wagen mit R = 1 (Mini-Instanz, an der jede Einheit von Hand geprüft wird, DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import res_dynamics as D  # noqa: E402


@pytest.fixture
def wagen_r1():
    """Wagen (Ts = 1, Euler-Form), Q = I, R = 1: (A, B, P, K). Handwert: K = (0,42208; 1,24393), Pole 0,378 ± 0,188 i."""
    a, b = D.wagen(1.0, "euler")
    p, k = D.dlqr(a, b, np.eye(2), 1.0)
    return a, b, p, k
