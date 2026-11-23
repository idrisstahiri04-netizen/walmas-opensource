"""Collecte niveau 0 : poids des criteres par des LLM open source via Ollama."""
import argparse
from walmas.llm.agents import OllamaClient, run_collection

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--models", nargs="+", default=["Llama31=llama3.1:8b", "Qwen25=qwen2.5:7b", "Mistral=mistral:7b"],
               help="paires NOM=tag_ollama")
p.add_argument("--m", type=int, default=20)
p.add_argument("--temperature", type=float, default=0.7)
p.add_argument("--base-seed", type=int, default=0)
p.add_argument("--retries", type=int, default=3)
p.add_argument("--out", default="data/opensource")
p.add_argument("--force", action="store_true")
p.add_argument("--shuffle", action="store_true", help="ordre des criteres aleatoire par agent (controle du biais de position)")
a = p.parse_args()
models = dict(s.split("=", 1) for s in a.models)
run_collection(OllamaClient(), models, a.out, a.m, a.temperature, a.base_seed, a.retries, a.force, a.shuffle)
