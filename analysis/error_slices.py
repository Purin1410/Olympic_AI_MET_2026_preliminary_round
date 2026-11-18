from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import linear_sum_assignment

import copy


def main(C, ac, OUT):
    P, B, T, S = C['paths'], C['bootstrap'], C['thresholds'], C['scorer']
    from . import analyze as A
    plt.rcParams.update(C['theme'])
    plt.rcParams['font.family'] = C['plot']['font']
    plt.rcParams.update({'font.size': C['plot']['font_size'], 'axes.titlesize': C['plot']['title_size'],
                         'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    notes, small = [], []
    colors = C['plot']['colors']
    models = list(C['models'])
    r3 = A.read_jsonl(P['task3_test'])
    ids3 = [r['ID'] for r in r3]
    gold = A.quads_by_id(r3)
    assert len(set(ids3)) == len(ids3)
    preds = {}
    for m in models:
        if m != 'gemma27b':
            path = Path(P['ours']) / C['models'][m] / f"pred_{C['sc']['phase']}_s{C['sc']['eval_seed']}_test_nulldrop.jsonl"
            preds[m] = A.quads_by_id(A.read_jsonl(path))
        else:
            cfg = copy.deepcopy(ac)
            cfg['scsg'].update(models={m: C['models'][m]}, eval_seeds=[C['sc']['eval_seed']],
                               guidance=[C['sc']['guidance']], n_samples=C['sc']['samples'])
            entry = A.load_nc(cfg)[(m, C['sc']['guidance'], C['sc']['eval_seed'])]
            preds[m] = A.merged_preds(entry['samples'], range(C['sc']['samples']), C['sc']['min_votes'])
        assert set(preds[m]) == set(ids3)
        if C['sc']['null_drop']:
            preds[m] = {i: [q for q in qs if q[0] != 'null' and q[2] != 'null'] for i, qs in preds[m].items()}
    counts = {m: A.per_sentence(gold, preds[m], S, ids3) for m in models}
    for m in models:
        assert np.isclose(A.cf1_from_sums(counts[m].sum(0)), A.score_task3(gold, preds[m], S)['cF1'])
    def estimate(arr, fn):
        arr = np.asarray(arr, float)
        rng = np.random.default_rng(B['seed'])
        draws = [fn(arr[rng.integers(len(arr), size=(min(B['batch'], B['n'] - b), len(arr)))].sum(1))
                 for b in range(0, B['n'], B['batch'])]
        return fn(arr.sum(0)), np.percentile(np.concatenate(draws), B['percentiles'], axis=0)

    def scalar_bar(ax, x, arr, fn, color, width, label=None, support=None):
        value, ci = estimate(arr, fn)
        n = len(arr) if support is None else support
        if min(n, len(arr)) < T['small_n']:
            small.append(f'{label or x}: n={n}, {len(arr)} câu')
            color = C['plot']['grey']
        ax.bar(x, value, width, color=color, label=label, zorder=2)
        ax.errorbar(x, value, yerr=np.abs(ci - value).reshape(2, 1), fmt='none', ecolor='#333333', capsize=2, lw=.8, zorder=3)
        return value, ci

    def save(fig, name, caption, title, top=.96):
        fig.text(.02, .015, caption, fontsize=9, va='bottom')
        fig.tight_layout(rect=(0, .025 + .15 * len(caption.splitlines()) / fig.get_figheight(), 1, top))
        fig.suptitle(title, y=.99, fontsize=C['plot']['title_size'])
        for ext in ['png']:
            fig.savefig(OUT / f'{name}.{ext}', dpi=C['plot']['dpi'], facecolor='white')
        plt.close(fig)

    def relation(p, g):
        if p == g:
            return 0
        pt, gt = [tuple(re.findall(C['spans']['token_regex'], s)) for s in [p, g]]
        if pt and gt and pt != gt:
            if len(pt) > len(gt) and any(pt[j:j + len(gt)] == gt for j in range(len(pt) - len(gt) + 1)):
                return 1
            if len(gt) > len(pt) and any(gt[j:j + len(pt)] == pt for j in range(len(gt) - len(pt) + 1)):
                return 2
        return 3 if set(pt) & set(gt) else 4

    def relaxed(gq, pq):
        base = np.array(A.sent_counts(gq, pq, S))
        gg = [g for g in gq if not any(g[:3] == p[:3] for p in pq)]
        pp = [p for p in pq if not any(g[:3] == p[:3] for g in gq)]
        w = np.zeros((len(gg), len(pp)))
        for j, g in enumerate(gg):
            for k, p in enumerate(pp):
                if g[1] == p[1] and relation(p[0], g[0]) <= 2 and relation(p[2], g[2]) <= 2:
                    valid = S['va_min'] <= p[3] <= S['va_max'] and S['va_min'] <= p[4] <= S['va_max']
                    w[j, k] = (max(0, 1 - np.linalg.norm(np.array(p[3:]) - g[3:]) / np.sqrt(S['d_max_sq'])) if valid else 0) + 1e-12
        if w.size:
            rr, cc = linear_sum_assignment(w, maximize=True)
            keep = w[rr, cc] > 0
            n = keep.sum()
            base += [w[rr[keep], cc[keep]].sum() - n * 1e-12, n, -n, -n]
        return base

    fig = plt.figure(figsize=(C['plot']['width'], 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    span_summary = []
    for col, (field, anchor, title) in enumerate([(2, (0, 1), 'Opinion: cố định Aspect + Category'), (0, (1, 2), 'Aspect: cố định Category + Opinion')]):
        ax = fig.add_subplot(gs[0, col])
        for j, m in enumerate(models):
            arr = np.zeros((len(ids3), 5))
            excluded = 0
            for row, i in enumerate(ids3):
                for p in preds[m][i]:
                    pool = [g for g in gold[i] if all(p[k] == g[k] for k in anchor)]
                    if pool:
                        arr[row, min(relation(p[field], g[field]) for g in pool)] += 1
                    else:
                        excluded += 1
            exact = int(arr[:, 0].sum())
            totals = arr.sum(0).astype(int)[1:]
            n, left = int(totals.sum()), 0
            for k, count in enumerate(totals, 1):
                color = colors[m] if count >= T['small_n'] else C['plot']['grey']
                if count < T['small_n']:
                    small.append(f'{title}/{m}/{C["spans"]["labels"][k]}: n={count}')
                ax.barh(j, count, left=left, color=color, edgecolor='white', hatch=C['spans']['hatches'][k], height=.55)
                if count >= T['span_label_n']:
                    ax.text(left + count / 2, j, str(count), ha='center', va='center', fontsize=9, bbox={'fc': 'white', 'ec': 'none', 'alpha': .85, 'pad': 1})
                left += count
            ax.annotate(f'n={n}', (n, j), xytext=(5, 0), textcoords='offset points', va='center', fontsize=9)
            span_summary.append(f'{title.split(":")[0]}/{m}: non-exact n={n}; 4 nhóm={totals.tolist()}; exact đã bỏ={exact}; ngoài pool={excluded}.')
        ax.set(yticks=range(3), yticklabels=models, xlim=(0, ax.get_xlim()[1] * 1.15), xlabel='Số prediction không khớp span', title=title)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=5))
        ax.invert_yaxis()
    fig.legend(handles=[Patch(facecolor='white', edgecolor='#555555', hatch=h, label=l) for h, l in zip(C['spans']['hatches'][1:], C['spans']['labels'][1:])], loc='upper center', bbox_to_anchor=(.5, .945), ncol=4, fontsize=9)
    ax = fig.add_subplot(gs[1, :])
    for j, m in enumerate(models):
        relaxed_counts = np.array([relaxed(gold[i], preds[m][i]) for i in ids3])
        vals = []
        for k, arr in enumerate([counts[m], relaxed_counts]):
            x = j + (k - .5) * .3
            v, ci = scalar_bar(ax, x, arr, A.cf1_from_sums, colors[m], .28)
            if k == 0:
                ax.patches[-1].set(hatch='///', alpha=.5)
            ax.text(x, ci[1] + .012, f'{v:.3f}', ha='center', fontsize=9)
            vals.append(f'{v:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]')
        notes.append(f'{m}: official / lenient cF1 = {" / ".join(vals)}; n=1000 câu; gold={sum(map(len, gold.values()))}, pred={sum(map(len, preds[m].values()))}.')
    ax.set(xticks=range(3), xticklabels=models, ylabel='cF1', ylim=(0, .99), title='Nới biên span tăng cF1 khoảng 0.07–0.10')
    ax.yaxis.grid(True, alpha=.18)
    ax.legend(handles=[Patch(facecolor='#aaaaaa', hatch='///', label='Chính thức'), Patch(facecolor='#777777', label='Nới biên span')], loc='upper center', ncol=2, frameon=False)
    save(fig, 'fig_span_near_miss', 'SCSG · SC15, nullDrop · seed 0; 4B/12B của nhóm, 27B tác giả no_guidance.\nTrên: n = prediction non-exact có anchor; không vẽ CI. Dưới: n=1000 câu, CI 95% bootstrap theo câu.',
         'Opinion thường ngắn hơn gold; nới biên tăng cF1 0.07–0.10', top=.90)

    return ['## fig_span_near_miss', *span_summary, *notes]
