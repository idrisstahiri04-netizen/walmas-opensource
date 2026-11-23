import numpy as np, pytest
from pathlib import Path
from walmas import analysis as an
from walmas.core.pipeline import load_original, run_consensus, CRITERIA, PROFILES

DATA = Path(__file__).parents[1] / "data" / "original"

@pytest.fixture(scope="module")
def raw(): return load_original(DATA)

def test_iqr_rejects_80(raw):
    assert an.iqr_table(raw).rejected.sum() == 80

def test_convergence_iterations_and_initial_W(raw):
    W_bar = run_consensus(raw)["W_bar"]
    df, _ = an.convergence(W_bar, [0.3, 0.5, 0.7])
    assert df.iterations.tolist() == [4, 5, 9] and df.converged.all()
    ki = an.kendall_initial(W_bar)
    assert ki["standard"] == pytest.approx(0.433, abs=1e-3) and ki["legacy"] == pytest.approx(0.86, abs=0.01)

def test_final_weights(raw):
    fw = an.final_weights(run_consensus(raw, 0.5)).set_index("criterion").w_star.round(2)
    expected = {"Regulatory Compliance": 7.56, "Settlements and Urban Centers": 7.11,
                "Urban Development and Infrastructure Location": 6.80, "Ecological Sensitivity": 6.28,
                "Slope": 5.88, "Climate and Weather Conditions": 5.09, "Aquifers and Surface Water": 4.90,
                "Geology": 4.40, "Topography": 4.03}
    assert fw.to_dict() == pytest.approx(expected, abs=1e-9)
    assert list(fw.index) == list(expected)          # ordre par rang

def test_profile_sensitivity(raw):
    d = an.max_abs_delta(an.profile_sensitivity(raw)).round(2)
    assert d.to_dict() == pytest.approx({"sans_ENR": 1.04, "sans_GGIS": 0.67, "sans_CEUP": 0.52, "sans_ES": 0.77})

def test_no_negotiation_gap(raw):
    assert an.no_negotiation_gap(run_consensus(raw)) < 1e-10

def test_synthetic_bootstrap_and_self_comparison():
    rng = np.random.default_rng(1)
    raw = {(a, e): rng.integers(1, 10, size=(9, 8)).astype(float) for a in ["A", "B"] for e in PROFILES}
    b = an.bootstrap(raw, B=30)
    assert ((b.ic95_bas <= b.w_star + 1e-9) & (b.w_star <= b.ic95_haut + 1e-9)).all()
    assert b.part_rang_conserve.between(0, 1).all()
    stats, table = an.compare(b.w_star, b.w_star)
    assert stats["spearman_rho"] == pytest.approx(1) and stats["mae"] == 0 and len(table) == len(CRITERIA)


def test_position_bias(tmp_path):
    import json
    from walmas.analysis import position_bias, position_summary
    from walmas.core.pipeline import CRITERIA
    rev = CRITERIA[::-1]
    recs = [dict(model="m", profile="ENR", agent=0, valid=True, response=json.dumps({c: 9 - i for i, c in enumerate(CRITERIA)})),
            dict(model="m", profile="ENR", agent=1, valid=True, order=rev,     # poids decroissants dans l'ordre du prompt
                 response=json.dumps({c: 9 - i for i, c in enumerate(rev)})),
            dict(model="m", profile="ENR", agent=2, valid=False, response="x")]
    p = tmp_path / "r.jsonl"; p.write_text("\n".join(json.dumps(r) for r in recs))
    pb = position_bias(p)
    assert len(pb) == 18 and (pb.rho == 1).all()
    assert pb[pb.agent == 1].set_index("position").criterion[1] == CRITERIA[-1]
    assert position_summary(pb).loc["m", "part_rho_positif"] == 1


def test_criterion_position_r2_separates_effects():
    import pandas as pd
    from walmas.analysis import criterion_position_r2
    rng = np.random.default_rng(0); rows = []
    for a in range(40):                                  # poids = effet critere seul, ordre aleatoire
        for p, c in enumerate(rng.permutation(9)):
            rows.append(dict(model="m", criterion=c, position=p + 1, weight=1 + c % 9))
    r = criterion_position_r2(pd.DataFrame(rows)).loc["m"]
    assert r.R2_critere == pytest.approx(1) and r.R2_position < 0.3
