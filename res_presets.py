"""Permalink (Begrenzen und Einrasten), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben) und Zufalls-Seed-Button."""

import math
import random
from dataclasses import dataclass

import streamlit as st

import res_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None
    digits: int | None = None      # Rundung der Schieberegler-Werte auf ihre Schrittweite


SETTING_SPECS = {
    "r_select": SettingSpec("r", float, C.DEFAULTS["r"], options=C.R_OPTIONS),
    "ts_select": SettingSpec("ts", float, C.DEFAULTS["ts"], options=C.TS_OPTIONS),
    "bform_radio": SettingSpec("b", str, C.DEFAULTS["bform"], options=C.BFORMS),
    "a_select": SettingSpec("a", float, C.DEFAULTS["a"], options=C.A_OPTIONS),
    "m_slider": SettingSpec("m", float, C.DEFAULTS["m"], C.M_MIN, C.M_MAX, digits=2),
    "q_select": SettingSpec("q", float, C.DEFAULTS["q"], options=C.Q_OPTIONS),
    "dm_slider": SettingSpec("dm", float, C.DEFAULTS["dm"], C.DM_MIN, C.DM_MAX, digits=3),
    "w_select": SettingSpec("w", float, C.DEFAULTS["w"], options=C.W_OPTIONS),
    "lqgform_radio": SettingSpec("lf", str, C.DEFAULTS["lform"], options=C.LQG_FORMS),
    "g_select": SettingSpec("g", str, C.DEFAULTS["g"], options=C.G_KEYS),
    "lqgq_select": SettingSpec("lq", str, C.DEFAULTS["lq"], options=C.LQ_KEYS),
    "lqgm_slider": SettingSpec("lm", float, C.DEFAULTS["lm"], C.LQM_MIN, C.LQM_MAX, digits=2),
    "seed_input": SettingSpec("seed", int, C.DEFAULTS["seed"], 0, C.SEED_MAX),
    "rs_slider": SettingSpec("n", int, C.DEFAULTS["n"], C.RS_MIN, C.RS_MAX),
}
PRESET_KEYS = {"r": "r_select", "ts": "ts_select", "bform": "bform_radio", "a": "a_select", "m": "m_slider", "q": "q_select", "dm": "dm_slider",
               "w": "w_select", "lform": "lqgform_radio", "g": "g_select", "lq": "lqgq_select", "lm": "lqgm_slider", "seed": "seed_input", "n": "rs_slider"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _snap(spec: SettingSpec, value):
    """Wert auf die Reglergrenzen begrenzen bzw. auf die nächste erlaubte Stufe einrasten; None, wenn nicht verwendbar."""
    if spec.options is not None:
        if spec.caster is str:
            return value if value in spec.options else None
        if not math.isfinite(value):
            return None
        return min(spec.options, key=lambda o: abs(o - value))
    if not math.isfinite(value):
        return None
    value = max(spec.lo, min(spec.hi, value))
    return round(value, spec.digits) if spec.digits is not None else value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt (Stufen eingerastet)."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            value = _snap(spec, value)
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State.
    Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS, dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
