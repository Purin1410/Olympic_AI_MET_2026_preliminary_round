from pathlib import Path

import copy
from collections import defaultdict, deque

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch


def main(cfg, author, out):
    p, boot = cfg['plot'], cfg['bootstrap']
    from .analyze import read_jsonl, quads_by_id, load_nc, merged_preds
    plt.rcParams.update({'font.family': p['font'], 'font.size': p['font_size'],
                         'axes.titlesize': 12, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none',
                         'figure.facecolor': 'white', 'axes.axisbelow': True})

    def save(fig, name):
        for ext in ('png',):
            fig.savefig(out / f'{name}.{ext}', dpi=p['dpi'], bbox_inches='tight')
        plt.close(fig)

    test_rows = read_jsonl(cfg['gold_test'])
    gold = quads_by_id(test_rows)
    assert len(gold) == len(test_rows), 'Duplicate gold sentence IDs'
    gold = {i: [q for q in qs if q[0] != cfg['null_aspect']] for i, qs in gold.items()}
    ids = list(gold)
    log_rows = read_jsonl(cfg['logsigma'])
    log = {(r['ID'], r['aspect'].lower()): np.array([r['pred_V'], r['pred_A']])
           for r in log_rows}
    assert len(log) == len(log_rows), 'Duplicate LogSigma (ID, aspect)'
    assert all((i, q[0]) in log for i, qs in gold.items() for q in qs)
    author = copy.deepcopy(author)
    author['scsg'].update(models={m: author['scsg']['models'][m] for m, v in cfg['models'].items()
                                  if v.get('author')}, eval_seeds=[cfg['author_seed']],
                          guidance=[cfg['author_guidance']], n_samples=cfg['n_samples'])
    nc = load_nc(author)
    rng = np.random.default_rng(boot['seed'])
    draws = rng.integers(len(ids), size=(boot['n'], len(ids)))
    groups = []
    for model, spec in cfg['models'].items():
        for phase in cfg['phases']:
            if spec.get('author'):
                entry = nc[(model, cfg['author_guidance'], cfg['author_seed'])]
                pred = entry['greedy'] if phase == 'greedy' else merged_preds(
                    entry['samples'], list(range(cfg['n_samples'])), cfg['min_votes'])
            else:
                path = Path(cfg['our_predictions']) / spec['directory']
                pred = quads_by_id(read_jsonl(path / f"pred_{phase}_s{cfg['our_seed']}_test_nulldrop.jsonl"))
            assert set(pred) == set(gold), f'Incomplete predictions: {model}/{phase}'
            llm, reg, sentence = [], [], []
            for j, i in enumerate(ids):
                pool = defaultdict(deque)
                for q in pred[i]:
                    pool[q[:3]].append(q[3:])
                for q in gold[i]:
                    if pool[q[:3]]:
                        llm.append(np.array(pool[q[:3]].popleft()) - np.array(q[3:]))
                        reg.append(log[(i, q[0])] - np.array(q[3:]))
                        sentence.append(j)
            llm, reg = np.array(llm), np.array(reg)
            assert len(llm) and np.isfinite(llm).all() and np.isfinite(reg).all()
            distances = np.stack([np.linalg.norm(llm, axis=1), np.linalg.norm(reg, axis=1)])
            counts = np.bincount(sentence, minlength=len(ids))
            sums = np.bincount(sentence, weights=distances[0] - distances[1], minlength=len(ids))
            deltas = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
            groups.append(dict(model=model, phase=phase, color=spec['color'], n=len(llm),
                               ns=int((counts > 0).sum()), dist=distances,
                               means=distances.mean(axis=1), rmse=np.sqrt(np.stack([
                                   (llm ** 2).mean(axis=0), (reg ** 2).mean(axis=0)])),
                               delta=float(np.mean(distances[0] - distances[1])),
                               ci=np.percentile(deltas, boot['percentiles'])))

    fig, ax = plt.subplots(figsize=p['single_size'])
    fig.subplots_adjust(left=0.16, right=0.99, top=0.84, bottom=0.46)
    strip_rng = np.random.default_rng(boot['seed'])
    for x, g in enumerate(groups):
        for method, offset in enumerate((-p['box_offset'], p['box_offset'])):
            color = g['color'] if method == 0 else p['logsigma_color']
            if method == 0 and g['phase'] == 'greedy':
                color = tuple(0.45 + 0.55 * c for c in to_rgb(color))
            d = g['dist'][method]
            ax.scatter(x + offset + strip_rng.uniform(-p['jitter'], p['jitter'], len(d)),
                       d, s=p['strip_size'], color=color, alpha=p['strip_alpha'],
                       linewidths=0, rasterized=True)
            box = ax.boxplot([d], positions=[x + offset], widths=p['box_width'],
                             whis=p['whiskers'], showfliers=False, patch_artist=True,
                             manage_ticks=False, medianprops={'color': '#222222'})
            box['boxes'][0].set(facecolor=color, edgecolor=color, alpha=0.8)
            if method == 0 and g['phase'] == 'greedy':
                box['boxes'][0].set_hatch('///')
            ax.scatter(x + offset, d.mean(), marker='D', s=20, color='white',
                       edgecolor='#222222', linewidth=0.7, zorder=5)
    ax.set(xlim=(-0.6, len(groups) - 0.4), ylim=(0, None),
           ylabel='Khoảng cách VA tới gold (thấp hơn tốt hơn)')
    ax.set_xticks(range(len(groups)), [f"{g['model']}\n{'SC' if g['phase'] == 'sc' else 'greedy'}\nn = {g['n']:,}"
                                      for g in groups])
    ax.tick_params(axis='x', length=0, pad=9)
    ax.grid(axis='y', alpha=0.18)
    better = all(g['ci'][1] < 0 for g in groups)
    fig.suptitle('SCSG dự đoán VA tốt hơn trên các tuple khớp' if better else
                 'SCSG mạnh ở V; chưa thắng VA nhất quán', y=0.985, fontsize=12)
    fig.text(0.5, 0.94, f'Gold task3 test · {len(ids):,} câu · {sum(map(len, gold.values())):,} tuple · mỗi nhóm có tập khớp riêng',
             ha='center', fontsize=10)
    ax.legend(handles=[Patch(facecolor=p['neutral_color'], label='SCSG (màu model)'),
                       Patch(facecolor=p['logsigma_color'], label='LogSigma (seed42)'),
                       plt.Line2D([], [], marker='D', color='none', markeredgecolor='#222222',
                                  markerfacecolor='white', label='Trung bình')],
              loc='lower center', bbox_to_anchor=(0.5, 1.015), ncol=3, frameon=False,
              handlelength=1.2, columnspacing=1.2)
    rows = [[f"{g['means'][0]:.2f} / {g['means'][1]:.2f}" for g in groups],
            [f"{g['rmse'][0, 0]:.2f} / {g['rmse'][1, 0]:.2f}" for g in groups],
            [f"{g['rmse'][0, 1]:.2f} / {g['rmse'][1, 1]:.2f}" for g in groups],
            [f"{g['delta']:+.3f}" for g in groups],
            [f"[{g['ci'][0]:+.3f};\n{g['ci'][1]:+.3f}]" for g in groups]]
    table = ax.table(cellText=rows, rowLabels=['TB S / L', 'RMSE_V S / L', 'RMSE_A S / L',
                                              'Δ TB (S − L)', 'CI 95% của Δ'],
                     cellLoc='center', rowLoc='right', bbox=[0, -0.91, 1, 0.59])
    table.auto_set_font_size(False)
    table.set_fontsize(p['font_size'])
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#EEEEEE')
        cell.set_linewidth(0.5)
        if r == 4:
            cell.set_height(cell.get_height() * 1.5)
    fig.text(0.5, 0.025, f"S = SCSG; L = LogSigma. CI: {boot['n']:,} paired bootstrap theo câu (seed {boot['seed']}).\n"
             'Hộp: Q1–Q3; râu: P5–P95; chấm: từng tuple. Δ < 0: SCSG tốt hơn.',
             ha='center', va='bottom', fontsize=10)
    save(fig, 'fig_va_llm_vs_regressor')

    notes = ['## fig_va_llm_vs_regressor']
    for g in groups:
        notes.append(f"- {g['model']} {g['phase']}: n={g['n']} tuples/{g['ns']} sentences; "
                     f"mean distance S/L={g['means'][0]:.4f}/{g['means'][1]:.4f}; "
                     f"RMSE_V S/L={g['rmse'][0,0]:.4f}/{g['rmse'][1,0]:.4f}; "
                     f"RMSE_A S/L={g['rmse'][0,1]:.4f}/{g['rmse'][1,1]:.4f}; "
                     f"delta={g['delta']:+.4f}, CI95%={g['ci'].round(4).tolist()}.")
    return notes
