from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def main(c, ac, summary, figs):
    p, sc, sty = c['paths'], c['sc'], c['plot']
    from . import analyze as a
    summary = summary['scsg']
    rows = read(ac['data']['gold_task3_test'])
    gold = a.quads_by_id(rows)
    texts = {r['ID']: r['Text'] for r in rows}
    ids = sorted(gold)
    H = a.load_author_parser(ac['ours']['helper_py'], ['parse_label_string', 'convert_tuples_to_output_format'])
    ceiling_pred = {}
    for r in rows:
        tuples = [(q['Aspect'], q['Category'], q['Opinion'], *q['VA'].split('#')) for q in r['Quadruplet']]
        ceiling_pred[r['ID']] = a.raw_to_quads(H, str(tuples), r['ID'], r['Text'], True)[0]
    ceiling_counts = a.per_sentence(gold, ceiling_pred, ac['scorer'], ids)
    ceiling = summary['A6b_parser_ceiling']['null_drop']['cF1_max']
    assert np.isclose(a.cf1_from_sums(ceiling_counts.sum(0)), ceiling, atol=1e-12, rtol=0)
    pred = {}
    for model, spec in c['models'].items():
        if 'directory' in spec:
            base = Path(p['scsg']) / spec['directory']
            pred[model] = {'sc': a.quads_by_id(read(base / f'pred_sc_s{sc["seed"]}_test_nulldrop.jsonl'))}
    assert not any(q[0] == 'null' or q[2] == 'null' for qs in gold.values() for q in qs)
    nc = ac['scsg'].copy()
    nc.update(models={m: ac['scsg']['models'][m] for m in c['models'] if m not in pred},
              eval_seeds=[sc['seed']], guidance=[sc['guidance']], n_samples=sc['n_samples'])
    for (model, _, _), data in a.load_nc({'scsg': nc}).items():
        pred[model] = {'sc': a.merged_preds(data['samples'], range(sc['n_samples']), sc['min_votes'])}
    rng = np.random.default_rng(c['bootstrap']['seed'])
    draws = rng.integers(len(ids), size=(c['bootstrap']['n'], len(ids)))
    plt.rcParams.update(c['theme'])
    plt.rcParams['font.family'] = sty['font']
    plt.rcParams.update({'font.size': sty['fontsize'], 'axes.labelsize': sty['fontsize'],
                         'xtick.labelsize': sty['fontsize'], 'ytick.labelsize': sty['fontsize'],
                         'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, len(c['models']), figsize=sty['size'], sharey=True)
    notes = ['# NOTES', '', '## fig_scsg_cf1_waterfall.png / .svg',
             'Extraction làm mất cF1 nhiều hơn VA ở cả ba model; đây là phân rã mô tả, không phải causal attribution.',
             f'Nguồn: summary.json A6 (SC, seed {sc["seed"]}, no_guidance/nullDrop), A6b; A1 chỉ đối chiếu, 27B A1 là trung bình 5 seed.',
             f'n={len(ids)} câu, {sum(map(len, gold.values()))} gold tuples. Parser ceiling={ceiling:.12f} (0.989 khi làm tròn 3 chữ số).']
    for ax, (model, spec) in zip(axes, c['models'].items()):
        counts = a.per_sentence(gold, pred[model]['sc'], ac['scorer'], ids)
        measured = a.metrics_from(*counts.sum(0))
        m = summary['detail'][model]['A6_breakdown']['sc15']
        assert all(np.isclose(measured[k], v, atol=1e-10, rtol=0) for k, v in m.items()), model
        a1 = next(r for r in summary['A1_greedy_vs_sc'] if r['model'] == model and r['mode'] == 'sc15' and r['source'] in ['no_guidance', 'ours_hf_noguid'])
        assert np.isclose(a1['cF1_per_seed'][sc['seed']], m['cF1'], atol=1e-12, rtol=0)
        sums = counts[draws].sum(1)
        den = 2 * sums[:, 1] + sums[:, 2] + sums[:, 3]
        boot = np.column_stack([2 * sums[:, 1] / den, a.cf1_from_sums(sums)])
        ci = np.quantile(boot, c['bootstrap']['quantiles'], axis=0)
        values = np.array([1., ceiling, m['F1_exact'], m['cF1']])
        ax.step(range(4), values, where='post', color=spec['color'], lw=2)
        for j, color in enumerate([sty['neutral'], spec['color'], sty['highlight']], 1):
            ax.fill_between([j - .18, j + .18], values[j], values[j - 1], color=color, alpha=.75, zorder=3)
        ax.vlines(range(2, 4), ci[0], ci[1], color='black', lw=1.2, zorder=4)
        ax.scatter(range(4), values, color=spec['color'], s=24, zorder=5)
        for j, val in enumerate(values):
            offset = (-14, -10) if j == 2 else (0, -22 if j == 3 else 10)
            ax.annotate(f'{val:.3f}', (j, val), xytext=offset, textcoords='offset points', ha='right' if j == 2 else 'center', va='top' if j >= 2 else 'bottom')
        ax.text(1.65, (values[1]+values[2])/2, f'Extraction\n−{ceiling-m["F1_exact"]:.3f}', ha='right', va='center', fontsize=10)
        ax.text(3.32, (values[2]+values[3])/2, f'VA\n−{m["F1_exact"]-m["cF1"]:.3f}', ha='left', va='center', fontsize=10)
        ax.set(title=model, ylim=sty['ylim'], xlim=(-.35, 4.15))
        ax.set_xticks(range(4), ['Lý tưởng', 'Gold\nround-trip', 'F1_exact', 'cF1'])
        ax.grid(axis='x', visible=False)
        ax.grid(axis='y', color='#E8E8E8', linewidth=.7)
        notes.append(f'- {model}: TP_cat={m["TP_cat"]:.0f}, FP={m["FP"]:.0f}, FN={m["FN"]:.0f}; F1_exact={values[2]:.12f}, cF1={values[3]:.12f}; extraction={values[1]-values[2]:.12f}, VA={values[2]-values[3]:.12f}.')
        notes.append(f'  CI95% [F1_exact; cF1]: ' + '; '.join(f'[{lo:.6f}, {hi:.6f}]' for lo, hi in ci.T) + '.')
    axes[0].set_ylabel('Điểm F1 / cF1')
    fig.suptitle('SCSG mất điểm do extraction nhiều hơn VA', fontsize=sty['title_fontsize'], y=.98)
    fig.text(.5, .90, f'Test tiếng Anh · SC · seed {sc["seed"]} · nullDrop · n = {len(ids):,} câu', ha='center', fontsize=10)
    fig.text(.5, .025, f'Nguồn: A6 / A6b · CI 95% chỉ cho F1_exact / cF1: bootstrap {c["bootstrap"]["n"]:,} lần theo câu (seed 0)', ha='center', fontsize=10)
    fig.subplots_adjust(left=.075, right=.985, bottom=.20, top=.81, wspace=.13)
    for extension in ['png']:
        fig.savefig(figs / f'fig_scsg_cf1_waterfall.{extension}', dpi=sty['dpi'])
    plt.close(fig)
    return notes[2:]
