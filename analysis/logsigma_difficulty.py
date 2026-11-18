from pathlib import Path

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(c, out):
    rows = [json.loads(s) for s in Path(c['gold']).read_text().splitlines() if s.strip()]
    gold, sid, size, mixed, same, minority, other = [], [], [], [], [], [], []
    tied_sentences = 0
    for i, row in enumerate(rows):
        va = np.array([[float(z) for z in a['VA'].split('#')] for a in row['Aspect_VA']])
        assert va.ndim == 2 and va.shape[1] == 2, 'Empty or invalid Aspect_VA'
        v, n = va[:, 0], len(va)
        hi, lo = (v >= c['sentiment_high']).sum(), (v <= c['sentiment_low']).sum()
        is_mixed = bool(hi and lo)
        tied_sentences += int(is_mixed and hi == lo)
        for pair in va:
            gold.append(pair); sid.append(i); size.append(n); mixed.append(is_mixed)
            same.append(n > 1 and (hi == n or lo == n))
            minority.append(is_mixed and ((hi > lo and pair[0] <= c['sentiment_low']) or
                                          (lo > hi and pair[0] >= c['sentiment_high'])))
            other.append((v.sum() - pair[0]) / (n - 1) if n > 1 else np.nan)
    gold, sid, size = np.array(gold), np.array(sid), np.array(size)
    mixed, same, minority, other = map(np.array, (mixed, same, minority, other))
    preds = [np.load(p) for p in c['predictions']]
    assert len(preds) == 3 and all(p.shape == gold.shape for p in preds), 'Pair count/shape mismatch'
    pred = np.mean(preds, axis=0)
    assert np.isfinite(pred).all() and np.isfinite(gold).all(), 'Nonfinite data'
    error = pred - gold
    sq = error ** 2
    rmse = np.sqrt(sq.sum(axis=1).mean())
    print(f'RMSE_VA sanity check: {rmse:.10f} (reference {c["reference_rmse_va"]}, '
          f'tolerance {c["sanity_tolerance"]})', flush=True)
    assert abs(rmse - c['reference_rmse_va']) <= c['sanity_tolerance'], 'STOP: RMSE_VA mismatch'
    rng = np.random.default_rng(c['seed'])
    weights = rng.multinomial(len(rows), np.full(len(rows), 1 / len(rows)), size=c['n_boot'])

    def estimate(mask, values, root=False):
        counts = np.bincount(sid[mask], minlength=len(rows))
        sums = np.bincount(sid[mask], weights=values[mask], minlength=len(rows))
        den = weights @ counts
        draws = (weights @ sums)[den > 0] / den[den > 0]
        point = values[mask].mean()
        if root:
            point, draws = np.sqrt(point), np.sqrt(draws)
        return point, np.percentile(draws, c['ci_percentiles'])

    def counts(mask):
        return f'n={mask.sum()} cặp; {np.unique(sid[mask]).size} câu'

    def numbers(mask):
        val, ci = estimate(mask, sq.sum(axis=1), True)
        bias, bci = estimate(mask, error[:, 0])
        return (f'{counts(mask)}; RMSE_VA={val:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]; '
                f'ΔV={bias:+.3f} [{bci[0]:+.3f}, {bci[1]:+.3f}]')

    plt.rcParams.update(c['theme'])
    plt.rcParams.update({'font.family': c['font'], 'font.size': 10, 'axes.labelsize': 10,
                         'axes.titlesize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'svg.fonttype': 'none'})
    def save(fig, name, title, subtitle):
        for ax in fig.axes:
            ax.grid(False, axis='x'); ax.grid(True, axis='y', alpha=.25)
            ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout(rect=(0, .02, 1, .88))
        fig.suptitle(title, fontsize=12, y=.98)
        fig.text(.5, .915, subtitle, ha='center', fontsize=9, color='#555555')
        for ext in ('png',):
            fig.savefig(out / f'{name}.{ext}', dpi=c['dpi'], facecolor='white')
        plt.close(fig)

    def points(ax, masks, values, root=False, offset=0, label=None, color=None, marker='o'):
        stats = [estimate(m, values, root) for m in masks]
        y = np.array([s[0] for s in stats]); ci = np.array([s[1] for s in stats]).T
        ax.errorbar(np.arange(len(masks)) + offset, y, yerr=np.abs(ci - y), fmt=marker,
                    color=color or c['color'], capsize=3, label=label, markersize=5)
        return y

    if c['same_side'] == 'nonmixed':
        same = (size > 1) & ~mixed
    else:
        assert c['same_side'] == 'strict'
    groups = [size == 1, same, mixed, minority]
    labels = ['Một\naspect', 'Nhiều aspect\ncùng phía' if c['same_side'] == 'strict' else
              'Nhiều aspect\nkhông mixed', 'Mixed\ntất cả', 'Mixed\nminority¹']
    labels = [f'{s}\nn={m.sum()}' for s, m in zip(labels, groups)]
    x, y = other[mixed], error[mixed, 0]
    r = np.corrcoef(x, y)[0, 1]
    w = weights[:, sid[mixed]].astype(float)
    total = w.sum(axis=1)
    sx, sy = w @ x, w @ y
    with np.errstate(divide='ignore', invalid='ignore'):
        rb = (w @ (x*y) - sx*sy/total) / np.sqrt(
            (w @ (x*x) - sx*sx/total) * (w @ (y*y) - sy*sy/total))
    rci = np.percentile(rb[np.isfinite(rb)], c['ci_percentiles'])
    scatter = abs(r) >= c['scatter_min_abs_r']
    fig, axes = plt.subplots(1, 3 if scatter else 2, figsize=(10, 4.9))
    points(axes[0], groups, sq.sum(axis=1), True)
    points(axes[1], groups, error[:, 0])
    axes[0].set(ylabel='RMSE_VA (điểm VA)', ylim=(0, None))
    axes[1].set(ylabel='Sai số V trung bình (pred − gold)')
    axes[1].axhline(0, color=c['neutral'], linewidth=1, linestyle='--')
    for ax in axes[:2]:
        ax.set_xticks(range(len(groups)), labels)
        ax.margins(x=.12)
    if scatter:
        axes[2].scatter(x, y, color=c['color'], alpha=.5, s=15, edgecolors='none')
        axes[2].axhline(0, color=c['neutral'], linewidth=1, linestyle='--')
        axes[2].set(xlabel='V gold trung bình của\ncác aspect còn lại', ylabel='Sai số V (pred − gold)',
                    title=f'Trong câu mixed: Pearson r={r:.2f}')
    fig.text(.02, .015, '¹ Minority là tập con của mixed; câu hòa hai phía không có minority.', fontsize=8)
    save(fig, 'fig_logsigma_mixed_sentiment', 'Trong câu mixed, sai số V đi cùng sentiment của aspect khác',
         f'Task1 test · ensemble3 · {mixed.sum()} cặp mixed / {np.unique(sid[mixed]).size} câu · n = số cặp · CI 95% theo câu')

    edges = np.arange(c['gold_range'][0], c['gold_range'][1] + c['bin_width']/2, c['bin_width'])
    bin_stats = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for d, (ax, dim) in enumerate(zip(axes, ('V', 'A'))):
        bins = np.clip(np.searchsorted(edges, gold[:, d], side='right') - 1, 0, len(edges)-2)
        valid = [k for k in range(len(edges)-1) if (bins == k).any()]
        masks = [bins == k for k in valid]
        stats = [estimate(m, error[:, d]) for m in masks]
        centers = (edges[:-1] + edges[1:]) / 2
        means = np.array([s[0] for s in stats]); ci = np.array([s[1] for s in stats]).T
        ax.errorbar(centers[valid], means, yerr=np.abs(ci-means), fmt='o-', color=c['color'],
                    linewidth=1.3, markersize=4, capsize=2)
        ax.axhline(0, color=c['neutral'], linestyle='--', linewidth=1)
        ax.set(xlabel=f'Giá trị gold {dim}', title=f'{dim}: trung bình ± CI 95%',
               xlim=c['gold_range'], xticks=np.arange(c['gold_range'][0], c['gold_range'][1]+1))
        bin_stats.append([(edges[k], int(m.sum()), s[0], *s[1]) for k, m, s in zip(valid, masks, stats)])
    axes[0].set_ylabel('Sai số có dấu (pred − gold)')
    save(fig, 'fig_logsigma_error_by_gold', 'LogSigma kéo giá trị cực đoan về giữa thang đo',
         f'Task1 test · ensemble3 · {len(gold):,} cặp · bin {c["bin_width"]} điểm · CI 95% theo câu')

    slopes = np.polyfit(gold[:, 0], pred[:, 0], 1)[0], np.polyfit(gold[:, 1], pred[:, 1], 1)[0]
    return [f"## fig_logsigma_error_by_gold", f"- Slopes V / A: {slopes[0]:.4f} / {slopes[1]:.4f}.",
            "## fig_logsigma_mixed_sentiment", f"- Mixed: {numbers(mixed)}; minority: {numbers(minority)}; Pearson r={r:.4f}."]
