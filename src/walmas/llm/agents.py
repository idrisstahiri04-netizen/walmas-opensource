"""Niveau 0 : agents LLM (Ollama) qui attribuent des poids 1..9 aux criteres."""
import json, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from ..core.pipeline import CRITERIA, PROFILES

PROFILE_NAMES = {"ENR": "Environment and Natural Resources", "GGIS": "Geography and GIS",
                 "CEUP": "Civil Engineering and Urban Planning", "ES": "Economics and Society"}
PROBLEM = "the selection of a landfill site"
TEMPLATE = ("You are an expert acting as a {profile} specialist in a spatial multi-criteria decision analysis "
            "for {problem}. Considering the criteria {criteria}, assign to each criterion an integer weight "
            "from 1 (low importance) to 9 (extreme importance), according to its importance from the point "
            "of view of your expertise. Other expert groups will provide their own weights, so your answer "
            "must reflect the priorities specific to your field. Reply only with a JSON object mapping each "
            "criterion name to its integer weight.")

def build_prompt(profile_name, criteria=CRITERIA, problem=PROBLEM) -> str:
    return TEMPLATE.format(profile=profile_name, problem=problem, criteria=", ".join(criteria))

def _to_int(v):
    """Entier 1..9 : accepte 7, 7.0, "7" ; rejette bool, 7.5, "high"."""
    if isinstance(v, bool): raise ValueError(f"valeur booleenne {v!r}")
    if isinstance(v, float) and v.is_integer(): v = int(v)
    elif isinstance(v, str) and v.strip().isdigit(): v = int(v.strip())
    if not isinstance(v, int): raise ValueError(f"valeur non entiere {v!r}")
    if not 1 <= v <= 9: raise ValueError(f"valeur hors 1..9 : {v}")
    return v

def parse_weights(text, criteria=CRITERIA) -> list:
    """JSON {critere: poids} -> poids dans l'ordre de criteria. ValueError si invalide."""
    try: obj = json.loads(text)
    except (json.JSONDecodeError, TypeError) as e: raise ValueError(f"JSON invalide : {e}") from None
    if not isinstance(obj, dict): raise ValueError("la reponse n'est pas un objet JSON")
    norm = {str(k).strip().casefold(): v for k, v in obj.items()}
    want = {c.strip().casefold(): c for c in criteria}
    missing, extra = set(want) - set(norm), set(norm) - set(want)
    if missing or extra or len(norm) != len(obj):
        raise ValueError(f"cles invalides : manquantes={sorted(missing)} en trop={sorted(extra)}")
    try: return [_to_int(norm[k]) for k in want]
    except ValueError as e: raise ValueError(f"poids invalide : {e}") from None

def weights_schema(criteria=CRITERIA) -> dict:
    """Schema JSON passe a Ollama (champ format) pour contraindre la sortie."""
    return {"type": "object", "required": list(criteria), "additionalProperties": False,
            "properties": {c: {"type": "integer", "minimum": 1, "maximum": 9} for c in criteria}}

class OllamaClient:
    def __init__(self, url="http://localhost:11434", timeout=300):
        self.url, self.timeout = url.rstrip("/"), timeout

    def _call(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def chat(self, model, prompt, schema, temperature, seed) -> str:
        return self._call("/api/chat", {"model": model, "messages": [{"role": "user", "content": prompt}],
                                        "stream": False, "format": schema,
                                        "options": {"temperature": temperature, "seed": seed}})["message"]["content"]

    def show(self, model) -> dict:
        d = self._call("/api/show", {"model": model}).get("details", {})
        # /api/show ne donne pas le digest (Ollama 0.40) : on le prend dans /api/tags
        digest = next((t["digest"] for t in self._call("/api/tags")["models"] if t["name"] == model), None)
        return {"tag": model, "digest": digest, "quantization_level": d.get("quantization_level"),
                "parameter_size": d.get("parameter_size")}

    def version(self) -> str:
        return self._call("/api/version")["version"]

def _log(log_path, **rec):
    with open(log_path, "a") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), **rec}) + "\n")

def collect_profile(client, tag, profile, m, temperature, base_seed, retries, log_path, criteria=CRITERIA,
                    shuffle=False):
    """m agents pour (modele, profil) -> array (n, m), lignes dans l'ordre de criteria. Tentative k :
    seed = base_seed + s + 1000*k (seed fixe => meme reponse, il faut la changer pour reessayer).
    shuffle : ordre des criteres (prompt et schema) tire au hasard par agent (controle du biais de position).
    RuntimeError si un agent echoue."""
    cols = []
    for s in range(m):
        order = [str(c) for c in np.random.default_rng(base_seed + s).permutation(criteria)] if shuffle else list(criteria)
        prompt, schema = build_prompt(PROFILE_NAMES[profile], order), weights_schema(order)
        for k in range(retries):
            seed, text, err = base_seed + s + 1000 * k, None, None
            try:
                text = client.chat(tag, prompt, schema, temperature, seed)
                w = parse_weights(text, criteria)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            _log(log_path, model=tag, profile=profile, agent=s, attempt=k, seed=seed, temperature=temperature,
                 prompt=prompt, order=order, response=text, valid=err is None, error=err)
            if err is None: cols.append(w); break
        else:
            raise RuntimeError(f"{tag} {profile} agent {s} : echec apres {retries} tentatives ({err})")
    return np.array(cols, dtype=int).T

def run_collection(client, models, out_dir, m=20, temperature=0.7, base_seed=0, retries=3, force=False, shuffle=False):
    """models {nom_court: tag_ollama} -> out_dir/{Nom}_G{i}.csv (9 x m, format data/original).
    Reprise : un CSV existant est saute sauf si force. Modele par modele (evite les swaps en RAM)."""
    out = Path(out_dir); (out / "raw").mkdir(parents=True, exist_ok=True)
    log_path = out / "raw" / "responses.jsonl"
    config = {"date": datetime.now(timezone.utc).isoformat(), "ollama_version": client.version(),
              "models": {short: client.show(tag) for short, tag in models.items()},
              "temperature": temperature, "base_seed": base_seed, "shuffle": shuffle, "m": m, "retries": retries,
              "problem": PROBLEM, "profiles": PROFILE_NAMES, "criteria": CRITERIA, "prompt_template": TEMPLATE}
    (out / "run_config.json").write_text(json.dumps(config, indent=2))
    for short, tag in models.items():
        for i, profile in enumerate(PROFILES, 1):
            path = out / f"{short}_G{i}.csv"
            if path.exists() and not force:
                print(f"{short} {profile} : deja present, saute"); continue
            t0 = time.time()
            W = collect_profile(client, tag, profile, m, temperature, base_seed, retries, log_path, shuffle=shuffle)
            np.savetxt(path, W, fmt="%d", delimiter=",")
            print(f"{short} {profile} : {m} agents, {time.time() - t0:.0f} s", flush=True)
