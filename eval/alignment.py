"""
Part 2 evaluation: cluster quality and cross-framework concept alignment.

    python -m eval.alignment

Gold: eval/concept_pairs.jsonl - 30 Flask->FastAPI concept pairs (10 seeds used
for tuning, 20 test) and 7 Flask concepts without a FastAPI counterpart
(2 seeds, 5 test) for the "no match" threshold.

Reports
  * cluster quality vs weak labels from URL sections (purity, NMI, ARI, Rand)
  * alignment hit@1 / hit@3 / hit@10 / MRR vs plain-retrieval baselines
  * ablations: tf-idf vs LSA (vs MiniLM), k-means vs HAC, restriction on/off,
    nearest vs Hungarian, framework-mean centring on/off
  * no-match threshold precision / recall
  * cluster pruning: recall vs number of clusters searched vs docs scored / time
  * HAC dendrogram over the cluster centroids of both frameworks
"""
from __future__ import annotations

import fnmatch
import itertools
import json
import time
from collections import Counter
from urllib.parse import urlsplit

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score

from eval.common import EVAL_DIR, RESULTS, fmt_table
from ir.cluster import ConceptAligner
from ir.engine import Engine


def load_pairs():
    with open(EVAL_DIR / "concept_pairs.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def section(url: str) -> str:
    path = urlsplit(url).path.replace("/en/stable", "")
    segs = [s for s in path.split("/") if s]
    return "/" + "/".join(segs[:-1]) + "/" if len(segs) > 1 else "/"


def purity(labels, truth):
    tot = 0
    for c in set(labels):
        tot += Counter(t for l, t in zip(labels, truth) if l == c).most_common(1)[0][1]
    return tot / len(labels)


def cluster_quality(eng, al: ConceptAligner):
    out = {}
    for fw in al.FRAMEWORKS:
        idx = al.members_fw[fw]
        lab = al.labels[idx]
        truth = [section(eng.chunks[i]["page_url"]) for i in idx]
        rng = np.random.default_rng(0)
        rnd = rng.integers(0, al.k_chosen[fw], len(idx))
        out[fw] = {"k": al.k_chosen[fw], "silhouette": al.silhouette[fw], "sections": len(set(truth)),
                   "purity": purity(lab, truth), "NMI": normalized_mutual_info_score(truth, lab),
                   "ARI": adjusted_rand_score(truth, lab), "RI": rand_score(truth, lab),
                   "purity_random": purity(rnd, truth), "NMI_random": normalized_mutual_info_score(truth, rnd)}
    return out


def ranks_to_metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    return {"hit@1": sum(1 for r in ranks if r is not None and r <= 1) / n,
            "hit@3": sum(1 for r in ranks if r is not None and r <= 3) / n,
            "hit@10": sum(1 for r in ranks if r is not None and r <= 10) / n,
            "MRR": sum(1 / r for r in ranks if r is not None) / n}


def first_hit(ids: list[str], gold: list[str]) -> int | None:
    for i, cid in enumerate(ids, 1):
        if any(fnmatch.fnmatch(cid, p) for p in gold):
            return i
    return None


def eval_baseline(eng, system, pairs):
    ranks = []
    for p in pairs:
        res = eng.run(system, f"framework:fastapi {p['flask']}", k=10)
        ranks.append(first_hit([eng.chunks[d]["id"] for d, _ in res], p["fastapi_gold"]))
    return ranks


def eval_aligner(eng, al, pairs, **kw):
    ranks, sims = [], []
    for p in pairs:
        r = al.align(p["flask"], **kw)
        ranks.append(first_hit([eng.chunks[d]["id"] for d, _ in r.ranked], p["fastapi_gold"]))
        sims.append(r.similarity)
    return ranks, sims


def tune(eng, al, seeds, space=None):
    space = space or {"m": [1, 2, 3, 5], "w": [0.3, 0.5, 0.7, 1.0], "one_to_one": [False, True],
                      "route": ["centroid", "evidence"]}
    best, best_s = None, -1
    keys = list(space)
    for vals in itertools.product(*(space[k] for k in keys)):
        kw = dict(zip(keys, vals))
        ranks, _ = eval_aligner(eng, al, seeds, **kw)
        m = ranks_to_metrics(ranks)
        s = m["MRR"] + 0.5 * m["hit@3"]
        if s > best_s + 1e-9:
            best, best_s = kw, s
    return best


def tune_tau(eng, al, seeds_pos, seeds_neg, kw):
    _, sp = eval_aligner(eng, al, seeds_pos, **kw)
    _, sn = eval_aligner(eng, al, seeds_neg, **kw)
    cands = sorted(set(sp + sn))
    best, best_acc = 0.0, -1
    for i in range(len(cands)):
        tau = (cands[i] + cands[i + 1]) / 2 if i + 1 < len(cands) else cands[i] + 1e-6
        acc = (sum(s >= tau for s in sp) + sum(s < tau for s in sn)) / (len(sp) + len(sn))
        if acc > best_acc:
            best, best_acc = tau, acc
    return best


class CentredAligner(ConceptAligner):
    def _vectors(self, space):
        V = super()._vectors(space)
        from sklearn.preprocessing import normalize
        fw = np.array([c["framework"] for c in self.eng.chunks])
        V = V.copy()
        for f in self.FRAMEWORKS:
            m = fw == f
            V[m] -= V[m].mean(0)
        return normalize(V).astype(np.float32)


def main():
    eng = Engine("data")
    pairs = load_pairs()
    pos = [p for p in pairs if p["fastapi"]]
    neg = [p for p in pairs if not p["fastapi"]]
    seeds_pos = [p for p in pos if p["seed"]]
    test_pos = [p for p in pos if not p["seed"]]
    seeds_neg = [p for p in neg if p["seed"]]
    test_neg = [p for p in neg if not p["seed"]]
    eng.params.update(json.loads((RESULTS / "part1.json").read_text())["tuned"]["bm25f"]) \
        if (RESULTS / "part1.json").exists() else None

    spaces = ["tfidf", "lsa"] + (["minilm"] if eng.use_minilm else [])
    variants = []
    for space, method, centre in itertools.product(spaces, ["kmeans", "hac"], [False, True]):
        variants.append((space, method, centre))

    rows, quality, configs = [], {}, {}
    base_rows = []
    for label, system in [("BM25F: Flask term -> FastAPI index", "bm25f"),
                          ("Dense LSA: Flask term -> FastAPI index", "lsa")] + \
            ([("Dense MiniLM: Flask term -> FastAPI index", "minilm")] if eng.use_minilm else []):
        m = ranks_to_metrics(eval_baseline(eng, system, test_pos))
        base_rows.append({"system": label, **m})

    aligners = {}
    for space, method, centre in variants:
        t = time.time()
        cls = CentredAligner if centre else ConceptAligner
        al = cls(eng, space=space, method=method)
        build = time.time() - t
        name = f"{space}/{method}{'/centred' if centre else ''}"
        aligners[name] = al
        quality[name] = cluster_quality(eng, al)
        kw = tune(eng, al, seeds_pos)
        tau = tune_tau(eng, al, seeds_pos, seeds_neg, kw)
        configs[name] = {**kw, "tau": tau, "build_s": build, "k": al.k_chosen}
        ranks, sims = eval_aligner(eng, al, test_pos, **kw)
        _, sims_neg = eval_aligner(eng, al, test_neg, **kw)
        ranks_off, _ = eval_aligner(eng, al, test_pos, **{**kw, "restrict": False})
        m_on = ranks_to_metrics(ranks)
        m_off = ranks_to_metrics(ranks_off)
        pred_nm_pos = sum(s < tau for s in sims)
        pred_nm_neg = sum(s < tau for s in sims_neg)
        prec = pred_nm_neg / (pred_nm_pos + pred_nm_neg) if pred_nm_pos + pred_nm_neg else float("nan")
        rec = pred_nm_neg / len(test_neg)
        rows.append({"space": space, "method": method, "centred": "yes" if centre else "no",
                     "route": kw["route"], "m": kw["m"], "w": kw["w"], "1:1": "yes" if kw["one_to_one"] else "no",
                     **m_on, "hit@3 (no restriction)": m_off["hit@3"], "MRR (no restriction)": m_off["MRR"],
                     "no-match P": prec, "no-match R": rec})

    # best configuration (by test MRR is not allowed; pick by seed tuning score of each variant)
    def seed_score(name):
        al = aligners[name]
        kw = {k: configs[name][k] for k in ("m", "w", "one_to_one", "route")}
        m = ranks_to_metrics(eval_aligner(eng, al, seeds_pos, **kw)[0])
        return m["MRR"] + 0.5 * m["hit@3"]
    best_name = max(aligners, key=seed_score)
    best = aligners[best_name]
    kw_best = {k: configs[best_name][k] for k in ("m", "w", "one_to_one", "route")}

    from eval.common import paired_bootstrap, wilcoxon_p
    rr = lambda ranks: [1 / r if r else 0.0 for r in ranks]
    best_rr = rr(eval_aligner(eng, best, test_pos, **kw_best)[0])
    sig_rows = []
    for label, system in [("BM25F baseline", "bm25f"), ("Dense LSA baseline", "lsa")]:
        b_rr = rr(eval_baseline(eng, system, test_pos))
        sig_rows.append({"comparison": f"{best_name} vs {label}", "delta MRR": float(np.mean(best_rr) - np.mean(b_rr)),
                         "p (bootstrap)": paired_bootstrap(b_rr, best_rr), "p (Wilcoxon)": wilcoxon_p(b_rr, best_rr)})

    # cluster pruning trade-off
    prune = []
    n_tgt = len(best.members_fw["fastapi"])
    for m in [1, 2, 3, 5, 10, 20, best.k_chosen["fastapi"]]:
        t = time.time()
        ranks, _ = eval_aligner(eng, best, test_pos, **{**kw_best, "m": m})
        dt = (time.time() - t) / len(test_pos) * 1000
        scored = np.mean([best.align(p["flask"], **{**kw_best, "m": m}).candidates_scored for p in test_pos])
        mm = ranks_to_metrics(ranks)
        prune.append({"clusters searched": m, "docs scored": float(scored), "% of FastAPI": 100 * scored / n_tgt,
                      "hit@3": mm["hit@3"], "hit@10": mm["hit@10"], "MRR": mm["MRR"], "ms/query": dt})

    # per-pair output for the best configuration
    per_pair = []
    for p in test_pos + test_neg:
        r = best.align(p["flask"], **kw_best)
        ids = [eng.chunks[d]["id"] for d, _ in r.ranked]
        per_pair.append({"pid": p["pid"], "flask": p["flask"], "expected": p["fastapi"] or "(no match)",
                         "source chunk": eng.chunks[r.source_chunk]["id"] if r.source_chunk is not None else "",
                         "matched cluster": best.describe(r.target_clusters[0]) if r.target_clusters else "",
                         "sim": round(r.similarity, 3),
                         "predicted": "no match" if r.similarity < configs[best_name]["tau"] else (ids[0] if ids else ""),
                         "gold rank": first_hit(ids, p["fastapi_gold"]) if p["fastapi"] else "-"})

    # dendrogram over centroids (best aligner)
    dendro_info = dendrogram(best, eng)

    q_rows = []
    for name, q in quality.items():
        for fw, v in q.items():
            q_rows.append({"clustering": name, "framework": fw, **v})
    md = ["# Part 2: clustering and cross-framework concept alignment\n",
          f"Gold: {len(pos)} Flask->FastAPI concept pairs ({len(seeds_pos)} seeds for tuning, "
          f"{len(test_pos)} test) and {len(neg)} Flask concepts with no FastAPI counterpart "
          f"({len(seeds_neg)} seed, {len(test_neg)} test). All numbers below are on the test items.\n",
          "## Cluster quality (weak labels = URL section of the page)\n",
          fmt_table(q_rows, ["clustering", "framework", "k", "silhouette", "sections", "purity", "NMI", "ARI", "RI",
                             "purity_random", "NMI_random"]),
          "\n## Alignment: Flask concept -> FastAPI chunks\n", "Baselines (plain retrieval into the FastAPI index):\n",
          fmt_table(base_rows, ["system", "hit@1", "hit@3", "hit@10", "MRR"]),
          "\nCluster-based alignment (m = clusters searched, w = weight of the dense evidence in RRF; "
          "tuned on seeds):\n",
          fmt_table(rows, ["space", "method", "centred", "route", "m", "w", "1:1", "hit@1", "hit@3", "hit@10", "MRR",
                           "hit@3 (no restriction)", "MRR (no restriction)", "no-match P", "no-match R"]),
          "\nSignificance of the seed-selected configuration vs the baselines (per-pair reciprocal rank, "
          f"n={len(test_pos)}):\n", fmt_table(sig_rows, ["comparison", "delta MRR", "p (bootstrap)", "p (Wilcoxon)"]),
          f"\nBest configuration on the seeds: **{best_name}** {json.dumps({**kw_best, 'tau': round(configs[best_name]['tau'], 3)})}\n",
          "## Cluster pruning: speed vs recall (best configuration)\n",
          fmt_table(prune, ["clusters searched", "docs scored", "% of FastAPI", "hit@3", "hit@10", "MRR", "ms/query"]),
          "\n## Per concept (best configuration, test items)\n",
          fmt_table(per_pair, ["pid", "flask", "expected", "sim", "gold rank", "predicted", "matched cluster"]),
          "\n## Hierarchy\n", dendro_info]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "part2.md").write_text("\n".join(md) + "\n")
    (RESULTS / "part2.json").write_text(json.dumps({"quality": quality, "baselines": base_rows, "variants": rows,
                                                    "configs": configs, "best": best_name, "pruning": prune, "significance": sig_rows,
                                                    "per_pair": per_pair}, indent=2, default=float))
    print("\n".join(md))


def dendrogram(al: ConceptAligner, eng) -> str:
    from scipy.cluster.hierarchy import dendrogram as dg, linkage
    Z = linkage(al.C, method="average", metric="cosine")
    labels = [f"{'FL' if al.cluster_fw[c] == 'flask' else 'FA'}: {' '.join(al.cluster_terms(c, 3))}"
              for c in range(len(al.C))]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, max(8, len(labels) * 0.16)))
        d = dg(Z, labels=labels, orientation="left", ax=ax, leaf_font_size=6, color_threshold=0.7)
        for t in ax.get_ymajorticklabels():
            t.set_color("#c0392b" if t.get_text().startswith("FL") else "#1f5fa8")
        ax.set_title("HAC (average linkage, cosine) over k-means centroids: Flask (red) and FastAPI (blue)", fontsize=9)
        ax.set_xlabel("cosine distance")
        fig.tight_layout()
        fig.savefig(RESULTS / "concept_dendrogram.png", dpi=150)
    except Exception as e:
        print("dendrogram plot skipped:", e)
    # earliest merges that join a Flask cluster with a FastAPI cluster = discovered alignments
    n = len(al.C)
    members = {i: [i] for i in range(n)}
    lines = []
    for i, (a, b, dist, _) in enumerate(Z):
        a, b = int(a), int(b)
        ma, mb = members[a], members[b]
        members[n + i] = ma + mb
        fa = {al.cluster_fw[c] for c in ma}
        fb = {al.cluster_fw[c] for c in mb}
        if len(ma) == 1 and len(mb) == 1 and fa != fb:
            fl = ma[0] if al.cluster_fw[ma[0]] == "flask" else mb[0]
            fp = mb[0] if fl == ma[0] else ma[0]
            lines.append(f"* d={dist:.3f}: FL [{', '.join(al.cluster_terms(fl, 4))}]  <->  "
                         f"FA [{', '.join(al.cluster_terms(fp, 4))}]")
    return ("Dendrogram: `concept_dendrogram.png`. First cross-framework merges of two single clusters "
            "(HAC finds these alignments without any supervision):\n\n" + "\n".join(lines[:15]))


if __name__ == "__main__":
    main()
