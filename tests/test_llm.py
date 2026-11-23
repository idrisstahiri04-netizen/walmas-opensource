import json, pytest
import numpy as np
from walmas.core.pipeline import CRITERIA, load_original
from walmas.llm.agents import build_prompt, parse_weights, collect_profile, run_collection, PROBLEM

GOOD = {c: (i % 9) + 1 for i, c in enumerate(CRITERIA)}

class FakeClient:
    """Repond selon une liste de reponses (cyclique) ; enregistre les appels. Aucun reseau."""
    def __init__(self, replies=None): self.replies, self.calls = replies or [json.dumps(GOOD)], []
    def chat(self, model, prompt, schema, temperature, seed):
        self.calls.append((model, seed)); r = self.replies[(len(self.calls) - 1) % len(self.replies)]
        if isinstance(r, Exception): raise r
        return r
    def show(self, model): return {"tag": model, "digest": "x", "quantization_level": "Q4", "parameter_size": "8B"}
    def version(self): return "0.0-fake"

def test_prompt_content():
    p = build_prompt("Geography and GIS")
    assert "Geography and GIS" in p and PROBLEM in p and all(c in p for c in CRITERIA)

def test_parse_valid_and_normalised():
    d = {f"  {c.upper()} ": v for c, v in GOOD.items()}
    d[f"  {CRITERIA[0].upper()} "] = 7.0; d[f"  {CRITERIA[1].upper()} "] = "7"
    assert parse_weights(json.dumps(d)) == [7, 7] + list(GOOD.values())[2:]

@pytest.mark.parametrize("bad", ["missing", "extra", 0, 10, 7.5, "high", True])
def test_parse_rejects(bad):
    d = dict(GOOD)
    if bad == "missing": del d[CRITERIA[0]]
    elif bad == "extra": d["Noise"] = 5
    else: d[CRITERIA[0]] = bad
    with pytest.raises(ValueError): parse_weights(json.dumps(d))

def test_parse_rejects_non_json():
    with pytest.raises(ValueError): parse_weights("Here are my weights: ...")

def test_retry_changes_seed(tmp_path):
    c = FakeClient(["not json", json.dumps(GOOD)])
    log = tmp_path / "log.jsonl"
    W = collect_profile(c, "fake", "ENR", 1, 0.7, 5, 3, log)
    assert W.shape == (9, 1) and [s for _, s in c.calls] == [5, 1005]
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    assert [(r["valid"], r["seed"]) for r in recs] == [(False, 5), (True, 1005)]

def test_fails_after_retries(tmp_path):
    c = FakeClient([ConnectionError("down"), "{}"])
    log = tmp_path / "log.jsonl"
    with pytest.raises(RuntimeError): collect_profile(c, "fake", "ES", 2, 0.7, 0, 3, log)
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(recs) == 3 and not any(r["valid"] for r in recs) and recs[0]["error"].startswith("ConnectionError")

def test_run_collection_and_resume(tmp_path):
    models = {"FakeA": "a:1b", "FakeB": "b:1b"}
    c = FakeClient(); run_collection(c, models, tmp_path, m=3)
    raw = load_original(tmp_path)
    assert len(raw) == 8 and all(v.shape == (9, 3) and ((v >= 1) & (v <= 9)).all() for v in raw.values())
    assert len(c.calls) == 8 * 3
    cfg = json.loads((tmp_path / "run_config.json").read_text())
    assert cfg["m"] == 3 and cfg["models"]["FakeA"]["tag"] == "a:1b"
    c2 = FakeClient(); run_collection(c2, models, tmp_path, m=3)
    assert c2.calls == []

def test_shuffle_permutes_prompt_but_keeps_csv_order(tmp_path):
    """Ordre aleatoire par agent dans le prompt et le schema ; la sortie reste dans l'ordre CRITERIA."""
    class Echo(FakeClient):
        def chat(self, model, prompt, schema, temperature, seed):
            self.calls.append((list(schema["properties"]), prompt))
            return json.dumps({c: i + 1 for i, c in enumerate(schema["properties"])})   # poids = position
    c, log = Echo(), tmp_path / "log.jsonl"
    W = collect_profile(c, "fake", "ENR", 3, 0.7, 0, 3, log, shuffle=True)
    orders = [o for o, _ in c.calls]
    assert len({tuple(o) for o in orders}) > 1 and all(sorted(o) == sorted(CRITERIA) for o in orders)
    for s, o in enumerate(orders):
        assert W[:, s].tolist() == [o.index(cr) + 1 for cr in CRITERIA]
        assert ", ".join(o) in c.calls[s][1]
    assert [json.loads(l)["order"] for l in log.read_text().splitlines()] == orders
