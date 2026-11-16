"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink (Begrenzen und Einrasten) und die Wirkung jedes Presets
(DEMO-PLAYBOOK §11a Punkt 7: kein Preset ohne Effekt meldet einen Effekt; die Zahlen der Hilfetexte stehen an den Messwerten)."""

import pytest

import res_constants as C
import res_evaluation as ev
import res_presets as P

DEFAULT = next(iter(C.PRESETS))


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_ranges_and_options():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            value = preset[key]
            if spec.options is not None:
                assert value in spec.options, (name, key, value)
            else:
                assert spec.lo <= value <= spec.hi, (name, key, value)


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[DEFAULT]) == ev.Settings()
    for key, state_key in P.PRESET_KEYS.items():
        assert P.SETTING_SPECS[state_key].default == C.PRESETS[DEFAULT][key] == C.DEFAULTS[key]


def test_setting_specs_cover_every_preset_key_and_presets_are_pairwise_different():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)


def test_every_preset_changes_something_except_the_default():
    for name, preset in C.PRESETS.items():
        if name != DEFAULT:
            assert any(preset[k] != C.DEFAULTS[k] for k in C.DEFAULTS), name


def test_permalink_snaps_to_options_and_clamps_sliders():
    spec_r, spec_m, spec_b, spec_n = (P.SETTING_SPECS[k] for k in ("r_select", "m_slider", "bform_radio", "rs_slider"))
    assert P._snap(spec_r, 0.05) in (0.03, 0.1)                       # nächste Stufe
    assert P._snap(spec_r, 1e9) == 10.0 and P._snap(spec_r, float("nan")) is None
    assert P._snap(spec_m, 5.0) == C.M_MAX and P._snap(spec_m, 0.0) == C.M_MIN and P._snap(spec_m, 1.2345) == 1.23
    assert P._snap(spec_b, "halten") == "halten" and P._snap(spec_b, "heun") is None
    assert P._snap(spec_n, 10**6) == C.RS_MAX


# --- Wirkung der Presets -----------------------------------------------------------------------------------------------------------------

def _p(name_part):
    name = next(n for n in C.PRESETS if name_part in n)
    return name, C.PRESETS[name]


def test_default_preset_reports_the_wagen_numbers_of_its_help():
    name, p = _p("Voreinstellung")
    view = ev.nyquist_view(p["r"], p["ts"], p["bform"])
    assert round(view["min_return"], 3) == 0.484 and "0,484" in C.PRESET_HELP[name]
    kind, text = ev.verdict_nyquist(p["r"], p["ts"], view["min_return"], view["w_min"], view["cont_min"], 3)
    assert kind == "warning" and "**0,484**" in text


def test_skalar_a2_preset_matches_its_help():
    name, p = _p("Skalar a = 2")
    sv = ev.skalar_view(p["a"])
    assert round(sv["K"], 3) == 1.618 and round(sv["window"][0], 3) == 0.618 and round(sv["window"][1], 3) == 1.854 and round(sv["min_return"], 3) == 0.461
    assert round(sv["pm_deg"], 1) == 29.8
    for token in ("1,618", "0,618 < g < 1,854", "0,461", "29,8°"):
        assert token in C.PRESET_HELP[name]


def test_aggressive_preset_matches_its_help_and_the_chosen_m_is_inside_the_window():
    name, p = _p("aggressiv")
    view = ev.nyquist_view(p["r"], p["ts"], p["bform"])
    assert round(view["min_return"], 3) == 0.348 and round(view["window"][1], 3) == 1.535
    assert p["m"] < view["window"][1]
    gv = ev.gain_view(p["r"], p["ts"], p["bform"], p["m"])
    assert gv["stable"]
    for token in ("0,348", "1,535", "m = 1,5"):
        assert token in C.PRESET_HELP[name]


def test_fine_sampling_preset_is_close_to_the_continuous_value_but_not_one():
    name, p = _p("Fein abgetastet")
    view = ev.nyquist_view(p["r"], p["ts"], p["bform"])
    assert round(view["min_return"], 3) == 0.958 < 1.0 and "0,958" in C.PRESET_HELP[name] and "0,484" in C.PRESET_HELP[name]
    kind, _ = ev.verdict_nyquist(p["r"], p["ts"], view["min_return"], view["w_min"], view["cont_min"], 3)
    assert kind == "info"


def test_doyle_preset_reports_instability_at_the_chosen_m():
    name, p = _p("Doyle")
    dv = ev.doyle_view(p["q"], p["dm"])
    assert round(dv["window"][0], 3) == 0.938 and round(dv["window"][1], 4) == 1.0067 and not dv["stable_lqg"] and dv["stable_lq"]
    kind, text = ev.verdict_doyle(p["q"], p["dm"], dv["window"], dv["lq_lower"], dv["return0"], dv["stable_lqg"], dv["stable_lq"])
    assert kind == "warning" and "instabil" in text
    for token in ("0,938 < m < 1,0067", "m = 1,02"):
        assert token in C.PRESET_HELP[name]
    assert ev.doyle_view(5.0, 1.02)["stable_lqg"]                  # Voreinstellung (q = σ = 5): dasselbe m ist hier stabil


def test_lqg_preset_matches_its_help():
    name, p = _p("LQG am Wagen")
    lv = ev.lqg_view(p["r"], p["ts"], p["bform"], p["lq"], p["g"], p["w"], p["lform"], p["lm"])
    assert round(lv["window"][0], 3) == 0.335 and round(lv["window"][1], 3) == 1.531 and round(lv["min_return"], 4) == 0.3433
    assert lv["window"][1] < lv["lqr_window"][1] and round(lv["lqr_window"][1], 3) == 1.936
    assert "0 bis 1,936" in C.PRESET_HELP[name]
    kind, _ = ev.verdict_lqg(lv, p["w"], p["lform"], ev.load_frozen()["lqg90_summary"])
    assert kind == "warning"
