"""Exakte Handrechnung in Q(√5): Zahlen a + b·√5 mit rationalen a, b (`fractions`), und die Handbeispiele des Skalars und des Doyle-Falls in Brüchen.

Herkunft: die kleine Klasse `QS` ist neu geschrieben nach dem Muster des Plan-Skripts `handbeispiele_04.py` (nicht importiert, nur gelesen). Alles hier ist
exakt (Brüche), damit die Tests „Bruch gleich Bruch“ prüfen können (DEMO-PLAYBOOK §12, Orakel mit `fractions`); die App benutzt es für die Anzeige des
Handbeispiels (Skalar a = 1 und a = 2). Konvention wie überall: u = −K·x, Q = R = 1, Eingang b = 1, Schleife am Streckeneingang geöffnet, L(z) = K/(z − a)."""

import math
from fractions import Fraction as F


class QS:
    """a + b·√5 mit rationalen a, b."""

    def __init__(self, a=0, b=0):
        self.a = F(a)
        self.b = F(b)

    @staticmethod
    def _lift(o):
        return o if isinstance(o, QS) else QS(o)

    def __add__(self, o):
        o = self._lift(o)
        return QS(self.a + o.a, self.b + o.b)

    __radd__ = __add__

    def __sub__(self, o):
        o = self._lift(o)
        return QS(self.a - o.a, self.b - o.b)

    def __rsub__(self, o):
        return QS(o) - self

    def __mul__(self, o):
        o = self._lift(o)
        return QS(self.a * o.a + 5 * self.b * o.b, self.a * o.b + self.b * o.a)

    __rmul__ = __mul__

    def inv(self):
        n = self.a * self.a - 5 * self.b * self.b
        return QS(self.a / n, -self.b / n)

    def __truediv__(self, o):
        return self * self._lift(o).inv()

    def __rtruediv__(self, o):
        return QS(o) * self.inv()

    def __neg__(self):
        return QS(-self.a, -self.b)

    def __eq__(self, o):
        o = self._lift(o)
        return self.a == o.a and self.b == o.b

    def __hash__(self):
        return hash((self.a, self.b))

    def __float__(self):
        return float(self.a) + float(self.b) * math.sqrt(5)

    def sign(self) -> int:
        """Exaktes Vorzeichen von a + b·√5 (Vergleich der Quadrate bei gemischten Vorzeichen)."""
        if self.a == 0 and self.b == 0:
            return 0
        if self.a >= 0 and self.b >= 0:
            return 1
        if self.a <= 0 and self.b <= 0:
            return -1
        d = self.a * self.a - 5 * self.b * self.b
        if self.a > 0:  # b < 0
            return 1 if d > 0 else -1
        return -1 if d > 0 else 1  # a < 0, b > 0

    def __lt__(self, o):
        return (self - o).sign() < 0

    def __gt__(self, o):
        return (self - o).sign() > 0

    def __repr__(self):
        return f"({self.a} + {self.b}·√5 = {float(self):.6f})"


S5 = QS(0, 1)


def qs_str(x: QS) -> str:
    """Darstellung einer Zahl a + b·√5 für die Anzeige: „1/2 + 1/2·√5“, „2 + √5“, „3/2“."""
    def frac(f: F) -> str:
        return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"

    a, b = x.a, x.b
    if b == 0:
        return frac(a).replace("-", "−")
    root = "√5" if abs(b) == 1 else f"{frac(abs(b))}·√5"
    if a == 0:
        return ("−" if b < 0 else "") + root
    return f"{frac(a)} {'−' if b < 0 else '+'} {root}".replace("-", "−")


def scalar_exact(a: int) -> dict:
    """Skalar x⁺ = a·x + u, Q = R = 1, für a = 1 oder a = 2, in Q(√5).

    DARE: p² − a²·p − 1 = 0, p = (a² + √(a⁴ + 4))/2, K = a·p/(1 + p), Pol c = a − K = a/(1 + p). Verstärkungsfenster |a − g·K| < 1:
    g von (a − 1)/K (nur wenn a > 1, sonst 0) bis (a + 1)/K. Kleinster Abstand bei z = −1: |1 + L(−1)| = (1 + c)/(1 + a).
    Rückführdifferenz-Gleichung bei z = −1: (1 + p)·|1 + L(−1)|² = 1 + 1/(1 + a)²."""
    if a == 1:
        p = (QS(1) + S5) / 2
    elif a == 2:
        p = QS(2) + S5
    else:
        raise ValueError("nur a = 1 und a = 2 haben eine einfache Darstellung in Q(√5)")
    k = a * p / (1 + p)
    c = a - k
    g_lo = QS(0) if a == 1 else (a - 1) / k
    g_hi = (a + 1) / k
    min_ret = (1 + c) / (1 + a)
    r0_squared = 1 / (1 + p)
    return {"a": a, "p": p, "K": k, "pole": c, "g_lo": g_lo, "g_hi": g_hi, "min_return": min_ret, "r0_squared": r0_squared,
            "residual_lhs": (1 + p) * min_ret * min_ret, "residual_rhs": F(1) + F(1, (1 + a) ** 2)}


def scalar_phase_margin_degrees(a: int) -> float:
    """Phasenreserve des Skalars (a = 1: genau 72°, a = 2: 29,8°) aus der Durchtrittsfrequenz |e^{jω} − a| = K, als Zahl (nicht exakt)."""
    ex = scalar_exact(a)
    k = float(ex["K"])
    cos_w = (1 + a * a - k * k) / (2 * a)
    w = math.acos(cos_w)
    return math.degrees(math.atan2(math.sin(w), a - cos_w))


def doyle5_polynomial(u) -> tuple:
    """Doyle mit q = σ = 5 (f = d = 5): Koeffizienten (c3, c2, c1, c0) von s⁴ + c3·s³ + c2·s² + c1·s + c0 bei m = 1 + u, in Brüchen."""
    u = F(u)
    return F(6), F(11), F(6) + 50 * u, F(1) - 25 * u


def doyle5_window() -> tuple:
    """Hurwitz-Fenster der Streckenverstärkung bei q = σ = 5: ((43 − 9√5)/25, 26/25), exakt."""
    return (43 - 9 * S5) / 25, QS(F(26, 25))
