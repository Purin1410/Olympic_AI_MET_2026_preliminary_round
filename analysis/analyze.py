"""Original parsing, scoring and SC calculations, without training dependencies."""


import ast, json, math, os, re, statistics
from collections import Counter, defaultdict

import numpy as np

METRICS = ['cF1', 'cP', 'cR', 'F1_exact', 'TP_cat', 'cTP', 'VA_loss', 'FP', 'FN']


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def lower_quads(qs, keep_case=False):
    """Lower quads."""
    f = (lambda x: x) if keep_case else str.lower
    out = []
    for q in qs:
        v, a = map(float, q["VA"].split("#"))
        out.append((f(q["Aspect"]), f(q.get("Category", "")), f(q.get("Opinion", "")), v, a))
    return out


def quads_by_id(rows, key="Quadruplet", keep_case=False):
    return {r["ID"]: lower_quads(r.get(key, []), keep_case) for r in rows}


def low_key(k):
    return tuple(x.lower() for x in k)


def sent_counts(gq, pq, sc):
    """Apply the official task 3 duplicate and VA matching rules."""
    d_max = math.sqrt(sc["d_max_sq"])
    ctp = 0.0; tp = fn = 0
    for g in gq:
        scores = []
        for p in pq:
            if g[:3] != p[:3]:
                continue
            if not (sc["va_min"] <= p[3] <= sc["va_max"] and sc["va_min"] <= p[4] <= sc["va_max"]):
                scores.append(0.0); continue
            scores.append(max(0.0, 1.0 - math.hypot(p[3] - g[3], p[4] - g[4]) / d_max))
        if len(scores) == 1:
            tp += 1; ctp += scores[0]
        else:
            fn += 1
    return ctp, tp, len(pq) - tp, fn


def metrics_from(ctp, tp, fp, fn):
    cp = ctp / (tp + fp) if tp + fp else 0.0
    cr = ctp / (tp + fn) if tp + fn else 0.0
    p_ = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    return {"cF1": 2 * cp * cr / (cp + cr) if cp + cr else 0.0, "cP": cp, "cR": cr,
            "F1_exact": 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0,
            "TP_cat": tp, "cTP": ctp, "VA_loss": tp - ctp, "FP": fp, "FN": fn}


def per_sentence(gold, pred, sc, ids=None):
    ids = ids if ids is not None else sorted(set(gold) | set(pred))
    return np.array([sent_counts(gold.get(i, []), pred.get(i, []), sc) for i in ids], dtype=float)


def score_task3(gold, pred, sc, ids=None):
    return metrics_from(*per_sentence(gold, pred, sc, ids).sum(0))


def cf1_from_sums(s):
    """Cf1 from sums."""
    ctp, tp, fp, fn = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    cp = np.divide(ctp, tp + fp, out=np.zeros_like(ctp), where=tp + fp > 0)
    cr = np.divide(ctp, tp + fn, out=np.zeros_like(ctp), where=tp + fn > 0)
    return np.divide(2 * cp * cr, cp + cr, out=np.zeros_like(cp), where=cp + cr > 0)


def vote_table(runs_for_id):
    """Vote table."""
    c = defaultdict(list)
    for qs in runs_for_id:
        for q in qs:
            c[q[:3]].append(q[3:])
    return c


def merge(runs_for_id, min_votes):
    """Keep case-sensitive majority keys and round mean VA to two decimals."""
    out = []
    for k, vs in vote_table(runs_for_id).items():
        if len(vs) >= min_votes:
            out.append(low_key(k) + (round(statistics.mean(x[0] for x in vs), 2), round(statistics.mean(x[1] for x in vs), 2)))
    return out


def merged_preds(samples, idx, min_votes):
    ids = samples[0].keys()
    return {i: merge([samples[j][i] for j in idx], min_votes) for i in ids}


def load_nc(cfg):
    s = cfg["scsg"]
    data = {}
    for mname, mdir in s["models"].items():
        for g in s["guidance"]:
            for seed in s["eval_seeds"]:
                base = os.path.join(s["results_dir"], str(seed), mdir, s["file_prefix"])
                greedy = quads_by_id(read_jsonl(f"{base}_{s['greedy_temp_tag']}_{g}.jsonl"))
                samples = [quads_by_id(read_jsonl(f"{base}_{s['sc_temp_tag']}_{g}_run{k}.jsonl"), keep_case=True)
                           for k in range(s["n_samples"])]
                data[(mname, g, seed)] = {"greedy": greedy, "samples": samples}
    return data


def load_author_parser(helper_py, names):
    """Extract only the parser functions; do not import model dependencies."""
    tree = ast.parse(open(helper_py, encoding="utf-8").read())
    mod = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    ns = {"re": re, "json": json}
    exec(compile(mod, helper_py, "exec"), ns)
    return ns


def raw_to_quads(H, raw, ex_id, text, null_drop, keep_case=False):
    """Raw to quads."""
    try:
        tup = H["parse_label_string"](raw, subtask=3)
        out = H["convert_tuples_to_output_format"](tup, ex_id, subtask=3, disable_null_aspect=null_drop, text=text)
        qs = out["Quadruplet"]
        return lower_quads(qs, keep_case), len(tup) > 0, sum(q["Aspect"] == "NULL" or q["Opinion"] == "NULL" for q in qs)
    except Exception:
        return [], False, 0


def mean_std(xs):
    return (statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0)


def a1_rows(data, gold, sc, n, mv):
    groups = defaultdict(list)
    for (mname, g, seed), d in data.items():
        groups[(mname, g)].append(d)
    rows = []
    for (mname, g), ds in groups.items():
        per = {"greedy": [score_task3(gold, d["greedy"], sc) for d in ds],
               f"sc{n}": [score_task3(gold, merged_preds(d["samples"], range(n), mv), sc) for d in ds]}
        for mode, lst in per.items():
            row = {"model": mname, "source": g, "mode": mode, "n_seeds": len(lst)}
            for m in METRICS:
                row[m], row[m + "_std"] = mean_std([x[m] for x in lst])
            row["cF1_per_seed"] = [x["cF1"] for x in lst]
            rows.append(row)
    return rows


def detail(ds, d0, gold, cfg, rng):
    """Calculate SC curves and component errors for the selected seeds."""
    s, sc = cfg["scsg"], cfg["scorer"]
    n, mv = s["n_samples"], s["min_votes"]
    r = {}

    curve = []
    for k in s["sc_k_list"]:
        vals = []
        for d in ds:
            for _ in range(1 if k == n else s["sc_k_draws"]):
                vals.append(score_task3(gold, merged_preds(d["samples"], rng.sample(range(n), k), k // 2 + 1), sc)["cF1"])
        m, sd = mean_std(vals)
        curve.append({"k": k, "cF1": m, "std": sd, "n": len(vals)})
    r["A2_sc_at_k"] = curve
    r["greedy_cF1"] = statistics.mean(score_task3(gold, d["greedy"], sc)["cF1"] for d in ds)

    r["A3_vote_threshold"] = [{"min_votes": v, **{m: statistics.mean(x[m] for x in lst) for m in ["cF1", "cP", "cR"]}}
                              for v in range(1, n + 1)
                              for lst in [[score_task3(gold, merged_preds(d["samples"], range(n), v), sc) for d in ds]]]

    hit, tot = Counter(), Counter()
    for d in ds:
        for i, gq in gold.items():
            gk = {q[:3] for q in gq}
            for k, vs in vote_table([d["samples"][j].get(i, []) for j in range(n)]).items():
                tot[len(vs)] += 1; hit[len(vs)] += low_key(k) in gk
    r["A4_votes_vs_precision"] = [{"votes": v, "n": tot[v], "precision": hit[v] / tot[v]} for v in sorted(tot)]

    sc0 = merged_preds(d0["samples"], range(n), mv)
    r["A6_breakdown"] = {"greedy": score_task3(gold, d0["greedy"], sc), f"sc{n}": score_task3(gold, sc0, sc)}

    r["A7_errors"] = error_analysis(gold, sc0, s)
    r["_sc0"] = sc0
    return r


def error_analysis(gold, pred, s):
    """Error types per wrong prediction. Ties between several near-miss gold tuples are broken by sorted order (deterministic)."""
    kinds = Counter()
    for i, gq in gold.items():
        gk, pk = {q[:3] for q in gq}, {q[:3] for q in pred.get(i, [])}
        for p in pk - gk:
            near = [g for g in sorted(gk - pk) if sum(a == b for a, b in zip(g, p)) == 2]
            if near:
                g = near[0]
                kind = ["aspect", "category", "opinion"][[a != b for a, b in zip(g, p)].index(True)]
            else:
                g, kind = None, "spurious"
            kinds["FP_" + kind] += 1
        for g in gk - pk:
            if not any(sum(a == b for a, b in zip(g, p)) == 2 for p in pk - gk):
                kinds["FN_missed"] += 1
    return {"counts": dict(kinds)}


def by_category(gold, pred, sc, min_support, train_cats):
    cats = Counter(q[1] for qs in gold.values() for q in qs)
    rows = []
    for c, n in cats.most_common():
        if n < min_support:
            continue
        g = {i: [q for q in qs if q[1] == c] for i, qs in gold.items()}
        p = {i: [q for q in qs if q[1] == c] for i, qs in pred.items()}
        rows.append({"category": c, "n_gold": n, "n_train": train_cats.get(c, 0), **{k: v for k, v in score_task3(g, p, sc).items() if k in ("cF1", "cP", "cR", "F1_exact")}})
    return rows


def parser_ceiling(H, gold_rows, gold, sc):
    """Parser ceiling."""
    out = {}
    for null_drop in (True, False):
        pred, lost = {}, []
        for r in gold_rows:
            tup = [(q["Aspect"], q["Category"], q["Opinion"], *q["VA"].split("#")) for q in r["Quadruplet"]]
            qs, _, _ = raw_to_quads(H, str(tup), r["ID"], r["Text"], null_drop)
            pred[r["ID"]] = qs
            got = {q[:3] for q in qs}
            lost += [q for q in lower_quads(r["Quadruplet"]) if q[:3] not in got]
        m = score_task3(gold, pred, sc)
        out["null_drop" if null_drop else "null_keep"] = {"cF1_max": m["cF1"], "n_gold_lost": len(lost),
                                                         "lost_share": len(lost) / sum(len(v) for v in gold.values()),
                                                         "lost_examples": [list(q[:3]) for q in lost[:10]]}
    return out


def sc_decompose(gold, d, sc, n, mv):
    """Cross greedy and SC tuples with their VA estimates."""
    G, S = d["greedy"], merged_preds(d["samples"], range(n), mv)
    gs, sg = {}, {}; cov_gs = cov_sg = tot_g = tot_s = 0
    for i in G:
        vt = defaultdict(list)
        for j in range(n):
            for q in d["samples"][j].get(i, []):
                vt[low_key(q[:3])].append(q[3:])
        out = []
        for q in G[i]:
            tot_g += 1
            if vt.get(q[:3]):
                cov_gs += 1; vs = vt[q[:3]]
                out.append(q[:3] + (round(statistics.mean(x[0] for x in vs), 2), round(statistics.mean(x[1] for x in vs), 2)))
            else:
                out.append(q)
        gs[i] = out
        gm = {q[:3]: q for q in G[i]}; out = []
        for q in S.get(i, []):
            tot_s += 1
            if q[:3] in gm:
                cov_sg += 1; out.append(q[:3] + gm[q[:3]][3:])
            else:
                out.append(q)
        sg[i] = out
    r = {k: score_task3(gold, v, sc) for k, v in [("greedy", G), ("sc", S), ("greedy_tuples+sc_VA", gs), ("sc_tuples+greedy_VA", sg)]}
    base = r["greedy"]["cF1"]
    return {"cF1": {k: v["cF1"] for k, v in r.items()},
            "delta_total": r["sc"]["cF1"] - base,
            "delta_VA_smoothing_only": r["greedy_tuples+sc_VA"]["cF1"] - base,
            "delta_tuple_selection_only": r["sc_tuples+greedy_VA"]["cF1"] - base,
            "VA_loss": {k: v["VA_loss"] for k, v in r.items()}, "TP_cat": {k: v["TP_cat"] for k, v in r.items()},
            "coverage_greedy_tuples_with_sample_VA": cov_gs / max(1, tot_g), "coverage_sc_tuples_in_greedy": cov_sg / max(1, tot_s)}
