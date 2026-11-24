"""
Part 4 evaluation: simulated relevance feedback on the residual collection.

    python -m eval.feedback

Simulated user: the judgments decide which of the top 5 are relevant
(grade >= 1) or not. After feedback the documents already shown and judged
are removed from BOTH the baseline ranking and the feedback ranking, and
both are scored against the remaining judgments (residual collection), so
the gain is not inflated by re-ranking documents the user already saw.

  * explicit feedback, 1 round and 2 rounds, for BM25F, LSA and hybrid
  * pseudo-relevance feedback (no user), with and without the confidence gate,
    evaluated on the full collection and on the residual collection
Parameters (number of expansion terms, PRF confidence gate) are tuned on dev.
"""
from __future__ import annotations

import json

import numpy as np

from eval.common import RESULTS, compare, evaluate, fmt_table, load_qrels, load_queries, splits
from ir.engine import Engine
from ir.feedback import confidence, feedback_search, pseudo_relevance_feedback

SHOW = 5


def ids(eng, run):
    return [eng.chunks[d]["id"] for d, _ in run]


def residual_qrels(qrels, qid, removed: set[str]):
    return {d: g for d, g in qrels.get(qid, {}).items() if d not in removed}


def simulate(eng, queries, qrels, system, rounds=1, n_terms=25):
    """Return per-query metrics for baseline and feedback rankings on the residual collection."""
    base_per, fb_per = {}, {}
    for q in queries:
        qid = q["qid"]
        rel_grades = qrels.get(qid, {})
        initial = eng.run(system, q["query"], k=200)
        shown = [d for d, _ in initial[:SHOW]]
        judged = set(shown)
        rel = [d for d in shown if rel_grades.get(eng.chunks[d]["id"], 0) >= 1]
        nonrel = [d for d in shown if d not in rel]
        fb = feedback_search(eng, q["query"], rel, nonrel, system=system, k=200, n_terms=n_terms, exclude=judged)
        for _ in range(rounds - 1):
            nxt = [d for d, _ in fb[:SHOW]]
            judged |= set(nxt)
            rel += [d for d in nxt if rel_grades.get(eng.chunks[d]["id"], 0) >= 1]
            nonrel += [d for d in nxt if rel_grades.get(eng.chunks[d]["id"], 0) < 1]
            fb = feedback_search(eng, q["query"], rel, nonrel, system=system, k=200, n_terms=n_terms, exclude=judged)
        removed = {eng.chunks[d]["id"] for d in judged}
        rq = {qid: residual_qrels(qrels, qid, removed)}
        base_rank = [c for c in ids(eng, initial) if c not in removed]
        mb, pb = evaluate({qid: base_rank}, rq, [qid])
        mf, pf = evaluate({qid: ids(eng, fb)}, rq, [qid])
        base_per[qid], fb_per[qid] = pb[qid], pf[qid]
    return base_per, fb_per


def mean(per, m):
    return float(np.mean([v[m] for v in per.values()])) if per else float("nan")


def main():
    eng = Engine("data")
    p1 = RESULTS / "part1.json"
    if p1.exists():
        t = json.loads(p1.read_text())["tuned"]
        eng.params.update(t["bm25f"])
        if "hybrid_lsa" in t:
            eng.params["rrf_w"] = t["hybrid_lsa"]["rrf_w"]
    queries = load_queries()
    qrels, source = load_qrels()
    sp = splits(queries)
    dev, test = sp["dev"], sp["test"]
    systems = ["bm25f", "lsa", "hybrid_lsa"] + (["minilm", "hybrid_minilm"] if eng.use_minilm else [])

    # tune number of expansion terms on dev (explicit, 1 round, BM25F)
    best_nt, best = 25, -1
    for nt in (20, 25, 30):
        b, f = simulate(eng, dev, qrels, "bm25f", 1, nt)
        s = mean(f, "nDCG@10")
        if s > best:
            best_nt, best = nt, s

    rows, sig = [], {}
    for system in systems:
        for rounds in (1, 2):
            b, f = simulate(eng, test, qrels, system, rounds, best_nt)
            c = compare(b, f)
            rows.append({"system": system, "feedback": f"explicit, {rounds} round{'s' if rounds > 1 else ''}",
                         "judged removed": SHOW * rounds,
                         "nDCG@10 before": mean(b, "nDCG@10"), "nDCG@10 after": mean(f, "nDCG@10"),
                         "MAP before": mean(b, "AP"), "MAP after": mean(f, "AP"),
                         "delta nDCG": c["delta"], "p (bootstrap)": c["p_bootstrap"], "wins/losses": f"{c['wins']}/{c['losses']}"})

    # pseudo-relevance feedback: tune the confidence gate on dev
    confs = sorted(confidence(eng.run("bm25f", q["query"], k=100)) for q in dev)
    gates = [0.0] + [float(np.quantile(confs, x)) for x in (0.25, 0.5, 0.75)]
    best_gate, best = 0.0, -1
    for g in gates:
        run = {q["qid"]: ids(eng, pseudo_relevance_feedback(eng, q["query"], k=100, n_terms=best_nt, min_conf=g)) for q in dev}
        m, _ = evaluate(run, qrels, [q["qid"] for q in dev])
        if m["nDCG@10"] > best:
            best_gate, best = g, m["nDCG@10"]
    base_run = {q["qid"]: ids(eng, eng.run("bm25f", q["query"], k=100)) for q in test}
    mb, pb = evaluate(base_run, qrels, [q["qid"] for q in test])
    prf_rows = []
    for label, gate in (("PRF, always on", 0.0), (f"PRF, confidence gate {best_gate:.2f} (dev)", best_gate)):
        run = {q["qid"]: ids(eng, pseudo_relevance_feedback(eng, q["query"], k=100, n_terms=best_nt, min_conf=gate))
               for q in test}
        applied = sum(1 for q in test if confidence(eng.run("bm25f", q["query"], k=100)) >= gate)
        m, pf = evaluate(run, qrels, [q["qid"] for q in test])
        c = compare(pb, pf)
        # residual view: remove the top 5 of the initial ranking from both
        res_b, res_f = {}, {}
        for q in test:
            removed = set(base_run[q["qid"]][:SHOW])
            rq = {q["qid"]: residual_qrels(qrels, q["qid"], removed)}
            res_b.update(evaluate({q["qid"]: [d for d in base_run[q["qid"]] if d not in removed]}, rq, [q["qid"]])[1])
            res_f.update(evaluate({q["qid"]: [d for d in run[q["qid"]] if d not in removed]}, rq, [q["qid"]])[1])
        cr = compare(res_b, res_f)
        prf_rows.append({"variant": label, "applied to": f"{applied}/{len(test)}",
                         "nDCG@10 base": mb["nDCG@10"], "nDCG@10 PRF": m["nDCG@10"], "delta": c["delta"],
                         "p": c["p_bootstrap"], "residual delta": cr["delta"], "residual p": cr["p_bootstrap"],
                         "MAP base": mb["MAP"], "MAP PRF": m["MAP"]})

    md = ["# Part 4: relevance feedback (Rocchio) and pseudo-relevance feedback\n",
          f"Judgments: **{source}**. Test split ({len(test)} queries). Rocchio alpha=1, beta=0.75, gamma=0.15; "
          f"{best_nt} expansion terms (tuned on dev). The simulated user judges the top {SHOW} with the qrels; "
          "scores are on the **residual collection** (judged documents removed from both rankings).\n",
          "## Explicit feedback\n",
          fmt_table(rows, ["system", "feedback", "judged removed", "nDCG@10 before", "nDCG@10 after", "delta nDCG",
                           "p (bootstrap)", "wins/losses", "MAP before", "MAP after"]),
          "\n## Pseudo-relevance feedback (BM25F, top 5 assumed relevant)\n",
          fmt_table(prf_rows, ["variant", "applied to", "nDCG@10 base", "nDCG@10 PRF", "delta", "p",
                               "residual delta", "residual p", "MAP base", "MAP PRF"])]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "part4.md").write_text("\n".join(md) + "\n")
    (RESULTS / "part4.json").write_text(json.dumps({"judgments": source, "n_terms": best_nt, "prf_gate": best_gate,
                                                    "explicit": rows, "prf": prf_rows}, indent=2, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
