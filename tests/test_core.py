import numpy as np, pytest
from pathlib import Path
from walmas.core.iqr import iqr_filter_mean
from walmas.core.kendall import kendall_w, kendall_w_legacy
from walmas.core.negotiation import negotiate
from walmas.core.pipeline import load_original, aggregate_levels, run_consensus

DATA = Path(__file__).parents[1] / "data" / "original"
RNG = np.random.default_rng(0)

@pytest.fixture(scope="module")
def raw(): return load_original(DATA)

def test_data_shape(raw):
    assert len(raw) == 12 and all(v.shape == (9, 20) for v in raw.values())
    assert all(((v >= 1) & (v <= 9)).all() for v in raw.values())

# ---- Kendall ----
def test_kendall_bounds_and_extremes():
    same = np.tile(np.arange(9.0)[:, None], (1, 4))
    assert kendall_w(same) == pytest.approx(1.0)
    rev = np.stack([np.arange(9.0), np.arange(9.0)[::-1]], axis=1)   # 2 profils opposes
    assert kendall_w(rev) == pytest.approx(0.0, abs=1e-12)

def test_kendall_matches_scipy_free_reference():
    M = RNG.random((9, 4)); n, L = M.shape
    r = np.argsort(np.argsort(-M, axis=0), axis=0) + 1.0      # sans egalites
    R = r.sum(axis=1); S = ((R - R.mean()) ** 2).sum()
    assert kendall_w(M) == pytest.approx(12 * S / (L**2 * (n**3 - n)))

def test_kendall_ties_use_average_ranks():
    M = np.array([[5., 5.], [5., 1.], [1., 1.]])
    assert 0 <= kendall_w(M) <= 1
    assert kendall_w(M, tie_correction=True) >= kendall_w(M)

def test_initial_W_on_published_weights(raw):
    Wb, _ = aggregate_levels(raw)
    assert kendall_w(Wb) == pytest.approx(0.43, abs=0.01)      # note 23 du rapport
    assert kendall_w_legacy(Wb) == pytest.approx(0.86, abs=0.01)

# ---- IQR ----
def test_iqr_removes_outlier():
    w = np.array([[2, 3, 3, 3, 4, 3, 4, 9.]])          # Q1=3, Q3=4 -> bornes [1.5, 5.5]
    assert iqr_filter_mean(w)[0] == pytest.approx(np.mean([2, 3, 3, 3, 4, 3, 4]))

def test_iqr_zero_spread_rejects_every_deviation():
    """Piege : si IQR=0 (agents quasi unanimes), TOUT ecart (meme 4 vs 3) est rejete."""
    w = np.array([[3, 3, 3, 3, 4, 3, 3, 9.]])
    assert iqr_filter_mean(w)[0] == pytest.approx(3.0)

def test_iqr_constant_and_never_empty():
    assert iqr_filter_mean(np.full((2, 20), 7.0)).tolist() == [7.0, 7.0]
    m = RNG.integers(1, 10, size=(9, 20)).astype(float)
    _, mask = iqr_filter_mean(m, return_mask=True)
    assert (mask.sum(axis=1) >= 1).all()

# ---- Negociation : proprietes (11) et (12) ----
@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7])
def test_mean_invariance_and_geometric_decay(raw, alpha):
    M0, _ = aggregate_levels(raw)
    res = negotiate(M0, alpha=alpha, eps=0, t_max=15)           # eps=0 : force 15 iterations
    mean0 = M0.mean(axis=1, keepdims=True)
    for t, Mt in enumerate(res.M_history):
        np.testing.assert_allclose(Mt.mean(axis=1, keepdims=True), mean0, atol=1e-12)   # (11)
        np.testing.assert_allclose(Mt - mean0, alpha**t * (M0 - mean0), atol=1e-12)     # (12)

@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7])
def test_final_weights_equal_simple_mean_and_W_to_one(raw, alpha):
    out = run_consensus(raw, alpha=alpha)
    assert out["result"].converged
    np.testing.assert_allclose(out["w_star"], out["W_bar"].mean(axis=1), atol=1e-12)    # w* = moyenne simple
    assert out["result"].w_history[-1] > 0.99
    assert out["w_hat"].sum() == pytest.approx(1.0)
    print(f"alpha={alpha}: T={out['result'].iterations}, W0={out['result'].w_history[0]:.3f}")

def test_degenerate_cases():
    M = np.full((9, 4), 5.0)
    res = negotiate(M); assert res.converged and np.allclose(res.M_final, 5.0)
    one = np.arange(1.0, 10.0)[:, None]                          # un seul profil
    assert negotiate(one).w_star == pytest.approx(one[:, 0])

def test_invariance_to_input_permutation():
    """Quel que soit l'input, W -> 1 : W_K ne valide rien (argument de la discussion)."""
    M = RNG.uniform(1, 9, size=(9, 4))
    assert negotiate(M, alpha=0.5, eps=1e-9).w_history[-1] == pytest.approx(1.0)
