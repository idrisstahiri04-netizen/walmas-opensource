import json, pytest
from pathlib import Path
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest
from walmas.core.pipeline import load_original, run_consensus
from walmas.llm.agents import OllamaClient

APP = str(Path(__file__).parents[1] / "app" / "streamlit_app.py")

@pytest.fixture(autouse=True)
def fake_ollama(monkeypatch):
    """Aucun appel reel a Ollama : /api/tags et /api/chat simules (poids 1..9 dans l'ordre du schema)."""
    def call(self, path, payload=None):
        if path == "/api/tags": return {"models": [{"name": "fake:1b"}]}
        keys = payload["format"]["required"]
        return {"message": {"content": json.dumps({c: i % 9 + 1 for i, c in enumerate(keys)})}}
    monkeypatch.setattr(OllamaClient, "_call", call)

def run(at):
    at.run(timeout=120); assert not at.exception, at.exception
    return at

def click(at, label):
    next(b for b in at.button if b.label == label).click(); return run(at)

def test_app_datasets_and_random():
    at = run(AppTest.from_file(APP, default_timeout=120))
    # w* affiche (parametres par defaut, k=1.5) = pipeline.run_consensus
    raw = load_original(Path(APP).parents[1] / "data" / at.selectbox(key="dataset").value)
    shown = next(d.value for d in at.dataframe if "w*" in d.value.columns and "ŵ" in d.value.columns)
    exp = sorted(run_consensus(raw)["w_star"], reverse=True)
    assert shown["w*"].tolist() == pytest.approx(exp, abs=1e-9)
    for name in at.selectbox(key="dataset").options:
        at.selectbox(key="dataset").select(name); run(at)
    click(at, "Entrées aléatoires")
    assert any("W_K initial → final" in m.label for m in at.metric)

def test_live_tab_with_fake_ollama():
    at = run(AppTest.from_file(APP, default_timeout=120))
    at.checkbox[-1].check()                                    # ordre des criteres aleatoire
    click(at, "Interroger les agents")
    assert not at.error and len(at.session_state.live) == 1 and at.session_state.live[0]["w"]
    at.slider[-1].set_value(1)
    click(at, "Lancer le mini-panel")
    raw_p, err = at.session_state.panel
    assert err is None and len(raw_p) == 4 and not at.error

def test_live_tab_ollama_down(monkeypatch):
    import urllib.error
    def down(self, path, payload=None): raise urllib.error.URLError("connexion refusee")
    monkeypatch.setattr(OllamaClient, "_call", down)
    at = run(AppTest.from_file(APP, default_timeout=120))
    click(at, "Interroger les agents")
    assert any("Réponse inutilisable" in e.value for e in at.error)
