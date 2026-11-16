"""Plotly-Abbildungen (alle Achsen gesperrt: fixedrange, damit Touch-Scrollen auf der Seite nicht am Diagramm hängen bleibt)."""

import math

import numpy as np
import plotly.graph_objects as go

import res_constants as C

LEGEND_BELOW = dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0)
CLIP = 2.5                       # Ortskurven werden außerhalb dieses Betrags abgeschnitten (die Kurve läuft für ω → 0 ins Unendliche)


def _base(fig: go.Figure, height: int = 360, title_x: str = "", title_y: str = "", legend: bool = True) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=60 if legend else 30), legend=LEGEND_BELOW if legend else None, showlegend=legend)
    fig.update_xaxes(fixedrange=True, title=title_x)
    fig.update_yaxes(fixedrange=True, title=title_y)
    return fig


def _circle(center: float, radius: float, n: int = 181):
    t = np.linspace(0.0, 2.0 * math.pi, n)
    return center + radius * np.cos(t), radius * np.sin(t)


def _clipped(z: np.ndarray):
    keep = np.abs(z) <= CLIP
    return z.real[keep], z.imag[keep]


def build_nyquist(view: dict) -> go.Figure:
    """Ortskurve L(e^{jω}) des Wagens (gestrichelt: ω < 0), Punkt −1, Kreis vom Radius des kleinsten Abstands, Kreis der Schranke r₀ und die stetige Kurve mit ihrem Kreis vom Radius 1."""
    fig = go.Figure()
    xs, ys = _clipped(view["Lc"])
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="stetiger Entwurf (gleiches R)", line=dict(color=C.COLOR_ORANGE, width=1.5), hoverinfo="skip"))
    cx, cy = _circle(-1.0, 1.0)
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", name="Abstand 1 (stetige Garantie)", line=dict(color=C.COLOR_ORANGE, width=1, dash="dash"), hoverinfo="skip"))
    xs, ys = _clipped(view["L"])
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="abgetastet: L(e^{jω})", line=dict(color=C.COLOR_MAIN, width=2.5), hovertemplate="Re %{x:.3f}, Im %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=-ys, mode="lines", showlegend=False, line=dict(color=C.COLOR_MAIN, width=1, dash="dot"), hoverinfo="skip"))
    cx, cy = _circle(-1.0, view["min_return"])
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", name=f"kleinster Abstand {view['min_return']:.3f}".replace(".", ","), line=dict(color=C.COLOR_BAD, width=1.5), hoverinfo="skip"))
    cx, cy = _circle(-1.0, view["r0"])
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", name=f"Schranke r₀ = {view['r0']:.3f}".replace(".", ","), line=dict(color=C.COLOR_GREY, width=1.5, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[-1.0], y=[0.0], mode="markers", name="kritischer Punkt −1", marker=dict(symbol="x", size=11, color=C.COLOR_BAD), hoverinfo="skip"))
    # unsichtbare Eckpunkte erzwingen den Ausschnitt; scaleanchor nur mit autorange (DEMO-PLAYBOOK §3)
    fig.add_trace(go.Scatter(x=[-3.2, 1.2], y=[-2.2, 2.2], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))
    _base(fig, height=520, title_x="Re L", title_y="Im L")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def build_skalar(view: dict) -> go.Figure:
    fig = go.Figure()
    w = view["w"]
    lw = view["L"]
    xs, ys = _clipped(lw)
    cx, cy = _circle(-1.0, 1.0)
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", name="Abstand 1", line=dict(color=C.COLOR_ORANGE, width=1, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="L(e^{jω})", line=dict(color=C.COLOR_MAIN, width=2.5), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=-ys, mode="lines", showlegend=False, line=dict(color=C.COLOR_MAIN, width=1, dash="dot"), hoverinfo="skip"))
    cx, cy = _circle(-1.0, view["min_return"])
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", name=f"kleinster Abstand {view['min_return']:.3f}".replace(".", ","), line=dict(color=C.COLOR_BAD, width=1.5), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[-1.0], y=[0.0], mode="markers", name="−1", marker=dict(symbol="x", size=11, color=C.COLOR_BAD), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[-3.2, 1.2], y=[-2.2, 2.2], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))
    _base(fig, height=380, title_x="Re L", title_y="Im L")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def build_rho(view: dict, m: float, r: float) -> go.Figure:
    """Spektralradius über der Streckenverstärkung m: unter 1 stabil; Marke bei m, Rand des Fensters."""
    from res_evaluation import M_GRID
    fig = go.Figure()
    rho = np.minimum(view["rho_curve"], 1.6)
    fig.add_trace(go.Scatter(x=M_GRID, y=rho, mode="lines", name="Spektralradius ρ(m)", line=dict(color=C.COLOR_MAIN, width=2.5)))
    fig.add_trace(go.Scatter(x=[C.M_MIN, C.M_MAX], y=[1.0, 1.0], mode="lines", name="ρ = 1 (Grenze)", line=dict(color=C.COLOR_BAD, width=1.5, dash="dash"), hoverinfo="skip"))
    if view["window"] is not None and view["window"][1] < C.M_MAX:
        fig.add_vline(x=view["window"][1], line=dict(color=C.COLOR_GREY, dash="dot"))
    rho_here = min(view["rho"], 1.6)
    fig.add_trace(go.Scatter(x=[m], y=[rho_here], mode="markers", name="gewähltes m", marker=dict(size=11, color=C.COLOR_ORANGE, line=dict(width=1, color="white"))))
    _base(fig, height=320, title_x="Streckenverstärkung m (Modell: m = 1)", title_y="Spektralradius ρ")
    fig.update_yaxes(range=[0, 1.65])
    fig.update_xaxes(range=[C.M_MIN, C.M_MAX])
    return fig


def build_trajectory(view: dict, m: float) -> go.Figure:
    """Position der freien Bahn aus x(0) = (1, 0); Werte über ±3 werden an der Achse abgeschnitten."""
    tr = np.clip(view["traj"][:, 0], -3.0, 3.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(tr)), y=tr, mode="lines+markers", name=f"Position, m = {m:.2f}".replace(".", ","), line=dict(color=C.COLOR_MAIN), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=[0, len(tr) - 1], y=[0, 0], mode="lines", showlegend=False, line=dict(color=C.COLOR_GREY, width=1), hoverinfo="skip"))
    _base(fig, height=320, title_x="Periode k", title_y="Position x₁(k)")
    fig.update_yaxes(range=[-3.1, 3.1])
    return fig


def build_ts(data: dict, ts_cur: float) -> go.Figure:
    rows = data["rows"]
    xs = [r["ts"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["min_euler"] for r in rows], mode="lines+markers", name="Euler-Form B = (0, Ts)ᵀ", line=dict(color=C.COLOR_MAIN, width=2.5)))
    fig.add_trace(go.Scatter(x=xs, y=[r["min_halten"] for r in rows], mode="lines+markers", name="Halte-Form B = (Ts²/2, Ts)ᵀ", line=dict(color=C.COLOR_LINE, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=[min(xs), max(xs)], y=[1.0, 1.0], mode="lines", name="stetig: Abstand 1", line=dict(color=C.COLOR_ORANGE, dash="dash"), hoverinfo="skip"))
    fig.add_vline(x=ts_cur, line=dict(color=C.COLOR_GREY, dash="dot"))
    _base(fig, height=340, title_x="Abtastzeit Ts (logarithmisch)", title_y="kleinster Abstand min |1 + L|")
    fig.update_xaxes(type="log")
    fig.update_yaxes(range=[0.3, 1.05])
    return fig


def build_held(data: dict) -> go.Figure:
    held = data["held"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[h["ts"] for h in held], y=[min(h["rho"], 3.0) for h in held], mode="lines+markers", name="Spektralradius des gehaltenen stetigen Gewinns",
                             line=dict(color=C.COLOR_MAIN, width=2.5)))
    fig.add_trace(go.Scatter(x=[held[0]["ts"], held[-1]["ts"]], y=[1.0, 1.0], mode="lines", name="ρ = 1", line=dict(color=C.COLOR_BAD, dash="dash"), hoverinfo="skip"))
    fig.add_vline(x=data["ts_max"], line=dict(color=C.COLOR_ORANGE, dash="dot"))
    _base(fig, height=300, title_x="Abtastzeit Ts (logarithmisch)", title_y="Spektralradius ρ (abgeschnitten bei 3)")
    fig.update_xaxes(type="log")
    fig.update_yaxes(range=[0, 3.1])
    return fig


def build_doyle(rows: list, q: float, dm: float) -> go.Figure:
    qs = [r["q"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qs, y=[r["hi"] for r in rows], mode="lines+markers", name="obere Kante des Fensters", line=dict(color=C.COLOR_MAIN, width=2)))
    fig.add_trace(go.Scatter(x=qs, y=[r["lo"] for r in rows], mode="lines+markers", name="untere Kante (Hurwitz)", line=dict(color=C.COLOR_LINE, width=2), fill="tonexty", fillcolor="rgba(47,107,101,0.15)"))
    fig.add_trace(go.Scatter(x=[min(qs), max(qs)], y=[dm, dm], mode="lines", name="gewähltes m", line=dict(color=C.COLOR_ORANGE, dash="dash")))
    fig.add_vline(x=q, line=dict(color=C.COLOR_GREY, dash="dot"))
    _base(fig, height=340, title_x="q = σ (logarithmisch)", title_y="Streckenverstärkung m")
    fig.update_xaxes(type="log")
    fig.update_yaxes(range=[0.89, 1.11])
    return fig


def build_lqg_traj(view: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(view["traj_lqr"])), y=np.clip(view["traj_lqr"], -3, 3), mode="lines", name="Zustands-LQR", line=dict(color=C.COLOR_LINE, width=2)))
    fig.add_trace(go.Scatter(x=np.arange(len(view["traj_lqg"])), y=np.clip(view["traj_lqg"], -3, 3), mode="lines", name="LQG (mit Schätzer)", line=dict(color=C.COLOR_MAIN, width=2.5)))
    _base(fig, height=320, title_x="Periode k", title_y="Position x₁(k), abgeschnitten bei ±3")
    fig.update_yaxes(range=[-3.1, 3.1])
    return fig


def build_lqg90(rows: list) -> go.Figure:
    """Nyquist-Abstand der 90 vorgerechneten Entwürfe über der Rauschstärke W; gestrichelt: der Zustands-LQR des jeweiligen Reglers."""
    colors = {("I", 1.0): C.COLOR_MAIN, ("I", 0.1): C.COLOR_LINE, ("rang1", 1.0): C.COLOR_ORANGE}
    names = {("I", 1.0): "Q = I, R = 1", ("I", 0.1): "Q = I, R = 0,1", ("rang1", 1.0): "Q = 100·(1,1)ᵀ(1,1), R = 1"}
    fig = go.Figure()
    seen = set()
    for key, color in colors.items():
        sel = [r for r in rows if (r["lq"], r["r"]) == key]
        for form in C.LQG_FORMS:
            for gk in C.G_KEYS:
                line = [r for r in sel if r["form"] == form and r["g"] == gk]
                line.sort(key=lambda r: r["w"])
                legend = key not in seen
                seen.add(key)
                fig.add_trace(go.Scatter(x=[r["w"] for r in line], y=[r["min_return"] for r in line], mode="lines", name=names[key], legendgroup=str(key), showlegend=legend,
                                         line=dict(color=color, width=1, dash="solid" if form == "schaetz" else "dot"), opacity=0.8,
                                         hovertemplate=f"{names[key]}, {form}, G {gk}: %{{y:.3f}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[C.LQG_W_GRID[0], C.LQG_W_GRID[-1]], y=[sel[0]["lqr_min"]] * 2, mode="lines", showlegend=False, line=dict(color=color, width=2, dash="dash"), hoverinfo="skip"))
    _base(fig, height=360, title_x="Rauschstärke W (logarithmisch)", title_y="kleinster Abstand min |1 + L|")
    fig.update_xaxes(type="log")
    fig.update_yaxes(range=[0.0, 0.65])
    return fig


def build_bound(rows: list) -> go.Figure:
    rs = [r["r"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rs, y=[r["min_return"] for r in rows], mode="lines+markers", name="gemessen: min |1 + L|", line=dict(color=C.COLOR_MAIN, width=2.5)))
    fig.add_trace(go.Scatter(x=rs, y=[r["r0"] for r in rows], mode="lines+markers", name="Schranke r₀ = √(R/(R + BᵀPB))", line=dict(color=C.COLOR_BAD, width=2, dash="dash")))
    _base(fig, height=340, title_x="R (logarithmisch)", title_y="Abstand zu −1")
    fig.update_xaxes(type="log")
    fig.update_yaxes(range=[0.0, 1.05])
    return fig
