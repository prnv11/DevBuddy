"""
Experiments for parts 1 and 3. Parameters are tuned on the dev split and
results are reported on the test split only.

    python -m eval.run part1     # BM25 / BM25F / dense / hybrid
    python -m eval.run part3     # + PageRank / personalised PageRank / HITS
    python -m eval.run all

Outputs: eval/results/<part>.json and eval/results/<part>.md
"""
from __future__ import annotations

import itertools
import json
import sys
import time

import numpy as np

from eval.common import RESULTS, compare, evaluate, fmt_table, load_qrels, load_queries, splits
from ir.engine import Engine

METRICS = ["nDCG@10", "P@5", "R@10", "MAP", "MRR"]


def run_system(eng: Engine, system: str, queries: list[dict], k: int = 100, **params):
    return {q["qid"]: [eng.chunks[d]["id"] for d, _ in eng.run(system, q["query"], k=k, **params)]
            for q in queries}


def judged_sets(qrels, source):
    """With human (pooled) judgments every judged pair is in qrels, including grade 0."""
    return {q: set(r) for q, r in qrels.items()} if source.startswith("human") else None


def score(eng, system, queries, qrels, metric="nDCG@10", **params):
    run = run_system(eng, system, queries, **params)
    mean, per = evaluate(run, qrels, [q["qid"] for q in queries])
    return mean[metric], mean, per


def grid(eng, system, dev, qrels, space: dict, base: dict | None = None):
    base = dict(base or {})
    best, best_p = -1.0, None
    keys = list(space)
    for vals in itertools.product(*(space[k] for k in keys)):
        p = {**base, **dict(zip(keys, vals))}
        s, _, _ = score(eng, system, dev, qrels, **p)
        if s > best + 1e-9:
            best, best_p = s, p
    return best_p, best


def by_type(per: dict, queries: list[dict], metric="nDCG@10") -> dict[str, float]:
    out = {}
    for t in ("api", "howto", "cross"):
        qs = [q["qid"] for q in queries if q["type"] == t and q["qid"] in per]
        out[t] = float(np.mean([per[q][metric] for q in qs])) if qs else float("nan")
    return out


def write(name: str, payload: dict, md: str):
    RESULTS.mkdir(exist_ok=True, parents=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2, default=float))
    (RESULTS / f"{name}.md").write_text(md)
    print(md)


def sig_line(name_a, name_b, c):
    star = "**" if c["p_bootstrap"] < 0.05 else ""
    return (f"| {name_b} vs {name_a} | {c['delta']:+.3f} | {star}{c['p_bootstrap']:.4f}{star} | "
            f"{c['p_wilcoxon']:.4f} | {c['wins']}/{c['losses']} |")


SIG_HEAD = "| comparison (nDCG@10) | delta | p (bootstrap) | p (Wilcoxon) | wins/losses |\n|:---|---:|---:|---:|---:|"


# ---------------------------------------------------------------- part 1
def part1(eng: Engine):
    queries = load_queries()
    qrels, source = load_qrels()
    sp = splits(queries)
    dev, test = sp["dev"], sp["test"]
    t0 = time.time()
    tuned = {}
    tuned["bm25"], _ = grid(eng, "bm25", dev, qrels, {"k1": [0.6, 0.9, 1.2, 1.5, 1.8, 2.1],
                                                     "b": [0.3, 0.5, 0.75, 0.9]})
    p, _ = grid(eng, "bm25f", dev, qrels, {"k1": [0.6, 0.9, 1.2, 1.5, 1.8, 2.1], "b": [0.3, 0.5, 0.75, 0.9]})
    space_w = []
    for wt, wh, wc in itertools.product([1.0, 2.0, 3.0], [1.0, 2.5, 4.0], [0.5, 1.2, 2.0]):
        space_w.append({"title": wt, "heading": wh, "body": 1.0, "code": wc})
    tuned["bm25f"], _ = grid(eng, "bm25f", dev, qrels, {"weights": space_w}, base=p)
    eng.params.update(tuned["bm25f"])            # lexical side of hybrid uses tuned BM25F
    dense_names = ["lsa"] + (["minilm"] if eng.use_minilm else [])
    for dn in dense_names:
        tuned[f"hybrid_{dn}"], _ = grid(eng, f"hybrid_{dn}", dev, qrels,
                                       {"rrf_w": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]})
    tune_s = time.time() - t0

    rows, pers = [], {}
    configs = [("BM25 (k1=1.2, b=0.75)", "bm25", {"k1": 1.2, "b": 0.75}),
               ("BM25 tuned", "bm25", tuned["bm25"]),
               ("BM25F (default weights)", "bm25f", {"k1": 1.2, "b": 0.75,
                                                    "weights": {"title": 2.0, "heading": 2.5, "body": 1.0, "code": 1.2}}),
               ("BM25F tuned", "bm25f", tuned["bm25f"])]
    for dn in dense_names:
        configs.append((f"Dense {dn.upper() if dn == 'lsa' else 'MiniLM'}", dn, {}))
    for dn in dense_names:
        configs.append((f"Hybrid RRF (BM25F + {dn.upper() if dn == 'lsa' else 'MiniLM'})", f"hybrid_{dn}",
                        tuned[f"hybrid_{dn}"]))
    lat = {}
    for label, system, params in configs:
        t = time.time()
        run = run_system(eng, system, test, **params)
        lat[label] = (time.time() - t) / len(test) * 1000
        mean, per = evaluate(run, qrels, [q["qid"] for q in test], judged_sets(qrels, source))
        pers[label] = per
        bt = by_type(per, test)
        rows.append({"system": label, **{m: mean[m] for m in METRICS},
                     "api": bt["api"], "howto": bt["howto"], "cross": bt["cross"], "ms/q": lat[label],
                     **({"judged@10": mean["judged@10"]} if "judged@10" in mean else {})})
    comps = [("BM25 tuned", "BM25F tuned")]
    for dn in dense_names:
        dl = f"Dense {dn.upper() if dn == 'lsa' else 'MiniLM'}"
        hl = f"Hybrid RRF (BM25F + {dn.upper() if dn == 'lsa' else 'MiniLM'})"
        comps += [("BM25F tuned", dl), ("BM25F tuned", hl), (dl, hl)]
    sig = {f"{b} vs {a}": compare(pers[a], pers[b]) for a, b in comps}
    # per-type significance for the headline comparisons
    type_sig = {}
    for t in ("api", "howto", "cross"):
        qs = {q["qid"] for q in test if q["type"] == t}
        for a, b in comps[1:]:
            pa = {k: v for k, v in pers[a].items() if k in qs}
            pb = {k: v for k, v in pers[b].items() if k in qs}
            type_sig[f"[{t}] {b} vs {a}"] = compare(pa, pb)

    md = [f"# Part 1: lexical, dense and hybrid retrieval\n",
          f"Judgments: **{source}**. Test split: {len(test)} queries "
          f"({', '.join(f'{t}={sum(q['type'] == t for q in test)}' for t in ('api', 'howto', 'cross'))}); "
          f"tuned on dev ({len(dev)} queries, {tune_s:.0f}s grid search).\n",
          "Columns api / howto / cross are nDCG@10 per query type.\n",
          fmt_table(rows, ["system"] + METRICS + ["api", "howto", "cross", "ms/q"]
                    + (["judged@10"] if "judged@10" in rows[0] else [])),
          "\n## Tuned parameters (dev)\n", "```json", json.dumps(tuned, indent=1), "```",
          "\n## Paired significance (test)\n", SIG_HEAD]
    md += [sig_line(*k.split(" vs ")[::-1], v) for k, v in sig.items()]
    md += ["\n### By query type\n", SIG_HEAD]
    md += [sig_line(*k.split(" vs ")[::-1], v) for k, v in type_sig.items()]
    if not eng.use_minilm:
        md.append("\n_MiniLM rows are missing because sentence-transformers is not installed here; "
                  "run `pip install sentence-transformers && python -m eval.run part1` to add them._")
    write("part1", {"judgments": source, "rows": rows, "tuned": tuned, "significance": sig,
                    "type_significance": type_sig, "dev_qids": [q["qid"] for q in dev],
                    "test_qids": [q["qid"] for q in test]}, "\n".join(md) + "\n")
    return tuned


# ---------------------------------------------------------------- part 3
def part3(eng: Engine, tuned: dict | None = None):
    import networkx as nx
    queries = load_queries()
    qrels, source = load_qrels()
    sp = splits(queries)
    dev, test = sp["dev"], sp["test"]
    if tuned is None:
        p1 = json.loads((RESULTS / "part1.json").read_text())
        tuned = p1["tuned"]
    eng.params.update(tuned["bm25f"])
    g = eng.graph

    # ---- PageRank correctness and convergence
    pr = eng.pagerank
    deltas = eng.pr_deltas
    G = g.to_networkx()
    nxpr = nx.pagerank(G, alpha=0.85, tol=1e-12, max_iter=1000)
    nxv = np.array([nxpr[i] for i in range(g.N)])
    v_fa = g.teleport_for({"fastapi"})
    ppr_mine = g.pagerank(teleport=v_fa)[0]
    nxppr = nx.pagerank(G, alpha=0.85, personalization={i: v_fa[i] for i in range(g.N)},
                        dangling={i: 1.0 for i in range(g.N)}, tol=1e-12, max_iter=1000)
    nxppr = np.array([nxppr[i] for i in range(g.N)])
    check = {"max_abs_diff_pagerank": float(np.abs(pr - nxv).max()),
             "max_abs_diff_personalised": float(np.abs(ppr_mine - nxppr).max()),
             "iterations": len(deltas), "final_delta": deltas[-1]}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.semilogy(range(1, len(deltas) + 1), deltas, marker="o", ms=3, color="#2a6fdb")
        ax.axhline(1e-8, ls="--", color="#999", lw=1)
        ax.text(len(deltas) * 0.6, 2e-8, "tolerance 1e-8", color="#666", fontsize=8)
        ax.set_xlabel("iteration")
        ax.set_ylabel("L1 change ||r(t) - r(t-1)||")
        ax.set_title(f"PageRank convergence (d=0.85, N={g.N}, {g.edges} content edges)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(RESULTS / "pagerank_convergence.png", dpi=150)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)

    # ---- graph statistics: content links vs navigation links
    from ir.links import LinkGraph
    gnav = LinkGraph(eng.pages, use_nav=True)
    prnav, _ = gnav.pagerank()

    def gini(x):
        x = np.sort(x)
        n = len(x)
        return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))

    stats = {
        "pages": g.N, "content_edges": g.edges, "nav_edges": gnav.edges,
        "dangling_pages_content": int(sum(1 for o in g.out if not o)),
        "gini_pagerank_content": gini(pr), "gini_pagerank_nav": gini(prnav),
        "top_content": [(g.urls[i], float(pr[i])) for i in np.argsort(-pr)[:10]],
        "top_nav": [(gnav.urls[i], float(prnav[i])) for i in np.argsort(-prnav)[:5]],
    }

    # ---- HITS example: hubs and authorities for one query
    ex_q = "dependencies"
    base = eng.run("bm25", ex_q, k=400)
    root = []
    for d, _ in base:
        p = int(eng.page_of[d])
        if p not in root:
            root.append(p)
        if len(root) >= 50:
            break
    auth, hub = g.hits(root)
    hits_ex = {"query": ex_q, "base_set": len(auth),
               "top_authorities": [(g.urls[p], round(v, 4)) for p, v in sorted(auth.items(), key=lambda x: -x[1])[:5]],
               "top_hubs": [(g.urls[p], round(v, 4)) for p, v in sorted(hub.items(), key=lambda x: -x[1])[:5]]}

    # ---- retrieval ablation
    systems = [("BM25F tuned", "bm25f", "beta_pr"), ("+ PageRank", "bm25f+pr", "beta_pr"),
               ("+ personalised PageRank (user stack)", "bm25f+ppr", "beta_ppr"),
               ("+ HITS authority", "bm25f+hits", "beta_hits")]
    if eng.clusters is not None:
        systems.append(("+ personalised PageRank (query cluster)", "bm25f+pprc", "beta_pprc"))
    betas = {}
    for label, s, bkey in systems[1:]:
        best, _ = grid(eng, s, dev, qrels, {bkey: [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]})
        betas[s] = best[bkey]
    rows, pers = [], {}
    for label, s, bkey in systems:
        params = {bkey: betas[s]} if s in betas else {}
        stack_q = [dict(q, query=f"{q['query']}") for q in test]
        run = {}
        for q in stack_q:
            kw = dict(params)
            if s == "bm25f+ppr":
                kw["stack"] = q["framework"]
            run[q["qid"]] = [eng.chunks[d]["id"] for d, _ in eng.run(s, q["query"], k=100, **kw)]
        mean, per = evaluate(run, qrels, [q["qid"] for q in test])
        pers[label] = per
        bt = by_type(per, test)
        rows.append({"system": label, "beta": betas.get(s, 0.0), **{m: mean[m] for m in METRICS},
                     "api": bt["api"], "howto": bt["howto"], "cross": bt["cross"]})
    # forced (untuned) beta, to show the effect when the dev split does not switch it off
    for label, s, bkey in systems[1:]:
        lab = f"{label} (beta=0.1 fixed)"
        run = {}
        for q in test:
            kw = {bkey: 0.1}
            if s == "bm25f+ppr":
                kw["stack"] = q["framework"]
            run[q["qid"]] = [eng.chunks[d]["id"] for d, _ in eng.run(s, q["query"], k=100, **kw)]
        mean, per = evaluate(run, qrels, [q["qid"] for q in test])
        pers[lab] = per
        bt = by_type(per, test)
        rows.append({"system": lab, "beta": 0.1, **{m: mean[m] for m in METRICS},
                     "api": bt["api"], "howto": bt["howto"], "cross": bt["cross"]})
    sig = {f"{l} vs BM25F tuned": compare(pers["BM25F tuned"], pers[l]) for l in pers if l != "BM25F tuned"}
    type_sig = {}
    for t in ("api", "howto", "cross"):
        qs = {q["qid"] for q in test if q["type"] == t}
        for l in [x for x in pers if x != "BM25F tuned"]:
            type_sig[f"[{t}] {l} vs BM25F tuned"] = compare(
                {k: v for k, v in pers["BM25F tuned"].items() if k in qs}, {k: v for k, v in pers[l].items() if k in qs})

    md = ["# Part 3: link analysis (PageRank, personalised PageRank, HITS)\n",
          f"Judgments: **{source}**. Test split ({len(test)} queries); beta tuned on dev. "
          "Score = BM25F/max(BM25F) + beta * g(page), g = log-scaled min-max of the page score, "
          "re-ranking the BM25F top 100.\n",
          fmt_table(rows, ["system", "beta"] + METRICS + ["api", "howto", "cross"]),
          "\n## Significance (test)\n", SIG_HEAD]
    md += [sig_line(*k.split(" vs ")[::-1], v) for k, v in sig.items()]
    md += ["\n### By query type\n", SIG_HEAD] + [sig_line(*k.split(" vs ")[::-1], v) for k, v in type_sig.items()]
    md += ["\n## PageRank implementation check\n",
           f"* Against `networkx.pagerank` on the same graph: max |difference| = "
           f"{check['max_abs_diff_pagerank']:.2e} (global), {check['max_abs_diff_personalised']:.2e} (personalised to FastAPI).",
           f"* Converged in {check['iterations']} iterations (final L1 change {check['final_delta']:.1e}); "
           "plot: `pagerank_convergence.png`.",
           "\n## Graph: content links vs navigation links\n",
           f"* {stats['pages']} pages, **{stats['content_edges']} content edges**, "
           f"{stats['nav_edges']} navigation edges; {stats['dangling_pages_content']} pages have no content out-links.",
           f"* Gini coefficient of PageRank: content graph {stats['gini_pagerank_content']:.3f} vs nav graph "
           f"{stats['gini_pagerank_nav']:.3f} (lower = flatter). Navigation links make every page look alike.",
           "\nTop pages by content-graph PageRank:\n"]
    md += [f"{i + 1}. {u} ({v:.4f})" for i, (u, v) in enumerate(stats["top_content"])]
    md += [f"\n## HITS example: query `{ex_q}` (base set {hits_ex['base_set']} pages)\n", "Authorities:"]
    md += [f"* {u} ({v})" for u, v in hits_ex["top_authorities"]]
    md += ["\nHubs:"] + [f"* {u} ({v})" for u, v in hits_ex["top_hubs"]]
    write("part3", {"judgments": source, "rows": rows, "betas": betas, "significance": sig,
                    "type_significance": type_sig, "check": check, "graph": stats, "hits_example": hits_ex},
          "\n".join(md) + "\n")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    eng = Engine("data")
    tuned = None
    if what in ("part1", "all"):
        tuned = part1(eng)
    if what in ("part3", "all"):
        from ir.cluster import ConceptAligner
        ConceptAligner(eng, space="lsa", method="kmeans").attach(eng)   # for cluster-personalised PageRank
        part3(eng, tuned)
