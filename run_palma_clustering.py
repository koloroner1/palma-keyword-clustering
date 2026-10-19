#!/usr/bin/env python3
"""Headless semantic clustering of Palma product-mapped keywords using the local gte-large model.
Zero API cost. Reads keywords-clean.csv, writes palma-semantic-clusters.csv.
"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.semantic_clustering import SemanticClusteringEngine
from collections import Counter

SRC = r"C:/Users/nyand/OneDrive/Desktop/Palma/work/marketing-decision-dataset/keywords-clean.csv"
OUT = r"C:/Users/nyand/OneDrive/Desktop/Palma/work/marketing-decision-dataset/palma-semantic-clusters.csv"

# unique product-mapped keywords carrying volume (highest signal)
seen = {}
with open(SRC, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if not r["handle"]:
            continue
        v = (r.get("volume") or "").strip()
        if not v.lstrip("-").isdigit():
            continue
        seen[r["keyword_norm"]] = max(seen.get(r["keyword_norm"], 0), int(v))
keywords = list(seen.keys())
print(f"Loaded {len(keywords)} unique product-mapped keywords with volume", flush=True)

engine = SemanticClusteringEngine()
if not engine.check_dependencies():
    print("ERROR: sentence-transformers/torch not available", flush=True)
    sys.exit(1)

def cb(m):
    print("  " + m, flush=True)

model_key = "Balanced (CPU/GPU) - 2GB+ VRAM - gte-large"
if not engine.load_model(model_key, cb):
    print("FAILED to load model", flush=True); sys.exit(1)

emb = engine.generate_embeddings(keywords, cb)
if emb is None:
    print("FAILED embeddings", flush=True); sys.exit(1)

clusters = engine.perform_clustering(emb, similarity_threshold=0.85, min_cluster_size=2, progress_callback=cb)
df = engine.format_clusters(keywords, clusters)
print(f"Clusters: {len(clusters)}  rows: {len(df)}", flush=True)

if not df.empty:
    df.to_csv(OUT, index=False)
    print(f"Saved -> {OUT}", flush=True)
    sizes = df.groupby("Parent Keyword").size().sort_values(ascending=False)
    print("Top clusters:", flush=True)
    for parent, n in sizes.head(15).items():
        children = df[df["Parent Keyword"] == parent]["Child Keyword"].tolist()
        print(f"  [{n}] {parent} :: {', '.join(children[:5])}", flush=True)
else:
    print("No clusters formed (threshold too high?)", flush=True)
