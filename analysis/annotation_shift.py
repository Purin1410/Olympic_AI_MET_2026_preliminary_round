
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


def main(cfg, output):
    from .analyze import read_jsonl, load_nc, merged_preds, quads_by_id
    plt.rcParams.update(cfg['theme'])
    plt.rcParams.update({'font.family': cfg['font'], 'axes.titlesize': 12, 'font.size': 10,
                         'axes.labelsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
                         'svg.fonttype': 'none'})
    rows = {s: read_jsonl(p) for s, p in cfg['gold'].items()}
    for s, rs in rows.items():
        assert len({r['ID'] for r in rs}) == len(rs), s
    counts = {s: defaultdict(Counter) for s in rows}
    for s, rs in rows.items():
        for r in rs:
            for q in r['Quadruplet']:
                counts[s][q['Aspect'].lower()][q['Category'].upper()] += 1
    combined = {a: counts['train'][a] + counts['dev'][a]
                for a in counts['train'].keys() | counts['dev'].keys()}
    cats = list(cfg['categories'])
    assert {q['Category'].upper() for rs in rows.values() for r in rs
            for q in r['Quadruplet']} <= set(cats)
    vector = lambda c: np.array([c[k] for k in cats], dtype=float)
    chosen = []
    for a in cfg['fixed_aspects']:
        test, td = counts['test'][a], combined[a]
        assert sum(test.values()) >= cfg['min_test'] and sum(td.values()) >= cfg['min_train_dev'], a
        tv = np.abs(vector(td) / sum(td.values()) - vector(test) / sum(test.values())).sum() / 2
        chosen.append((a, tv))
    selected = {x[0] for x in chosen}
    rng = np.random.default_rng(cfg['seed'])

    def boot(matrix):
        draws = rng.integers(len(matrix), size=(cfg['n_boot'], len(matrix)))
        return matrix[draws].sum(axis=1)

    def bounds(values):
        return np.percentile(values, cfg['ci'], axis=0)

    def save(fig, name):
        for ax in fig.axes:
            ax.grid(False, axis='x')
            ax.grid(True, axis='y', color='#eeeeee', linewidth=0.6)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
        for ext in ('png',):
            fig.savefig(output / f'{name}.{ext}', dpi=cfg['dpi'], facecolor='white')
        plt.close(fig)

    def aspect_matrix(rs, aspect):
        return np.array([vector(Counter(q['Category'].upper() for q in r['Quadruplet']
                                       if q['Aspect'].lower() == aspect)) for r in rs
                         if any(q['Aspect'].lower() == aspect for q in r['Quadruplet'])])

    fig, (ax, tvax) = plt.subplots(1, 2, figsize=cfg['sizes'][0],
                                  gridspec_kw={'width_ratios': [4, 1]})
    fig.subplots_adjust(left=.22, right=.97, top=.88, bottom=.29, wspace=.22)
    used = [k for k in cats if any(combined[a][k] + counts['test'][a][k] for a in selected)]
    tv_intervals = {}
    for i, (a, tv) in enumerate(chosen):
        for j, c in enumerate([combined[a], counts['test'][a]]):
            total, left, y = sum(c.values()), 0., 2.5 * i + j
            for k in used:
                share = c[k] / total
                color, hatch = cfg['categories'][k]
                ax.barh(y, share, left=left, height=.75, color=color, hatch=hatch,
                        edgecolor='#444444', linewidth=.35)
                if share >= cfg['segment_label_min']:
                    ax.text(left + share / 2, y, str(c[k]), ha='center', va='center', fontsize=10,
                            color='white' if color in ['#0072B2', '#009E73'] else '#111111')
                left += share
            ax.text(1.02, y, f'n={total}', va='center', fontsize=10)
        b0 = boot(aspect_matrix(rows['train'] + rows['dev'], a))
        b1 = boot(aspect_matrix(rows['test'], a))
        lo, hi = bounds(np.abs(b0 / b0.sum(1)[:, None] - b1 / b1.sum(1)[:, None]).sum(1) / 2)
        tv_intervals[a] = (lo, hi)
        tvax.plot([lo, hi], [2.5*i+.5]*2, color='#7F7F7F', linewidth=2)
        tvax.scatter(tv, 2.5*i+.5, color='#111111', s=20, zorder=3)
    ticks = [2.5*i+j for i in range(len(chosen)) for j in range(2)]
    ax.set(yticks=ticks, yticklabels=[f'{a} · {s}' for a, _ in chosen for s in ['train+dev', 'test']],
           xlim=(0, 1), xlabel='Tỷ lệ category (số ghi trong thanh = tuples)')
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(ticks[-1]+.8, -.8)
    tvax.set(xlim=(-.05, 1.05), ylim=ax.get_ylim(), yticks=[], xticks=[0, .5, 1], xlabel='TV [0–1]')
    tvax.set_title('TV và CI 95%', fontsize=10)
    handles = [Patch(facecolor=cfg['categories'][k][0], hatch=cfg['categories'][k][1],
                     edgecolor='#444444', label=k) for k in used]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, .045), ncol=2, fontsize=10)
    fig.suptitle('Cùng aspect, phân bố category đổi giữa các split', y=.975, fontsize=12)
    fig.text(.22, .925, f'{len(chosen)} aspect cố định; test ≥ {cfg["min_test"]}, train+dev ≥ {cfg["min_train_dev"]}; n = số tuples.', fontsize=10)
    fig.text(.03, .018, 'Nguồn: gold DimABSA eng-restaurant; CI: bootstrap theo câu, 2.000 lần, seed 0.', fontsize=10)
    save(fig, 'fig_category_shift_aspects')

    fig, axes = plt.subplots(1, 2, figsize=cfg['sizes'][1])
    fig.subplots_adjust(left=.08, right=.98, bottom=.24, top=.76, wspace=.28)
    split_totals = {}
    for ax, pair in zip(axes, cfg['count_pairs']):
        for j, k in enumerate(pair):
            vals = []
            for s, rs in rows.items():
                v = np.array([sum(q['Category'].upper() == k for q in r['Quadruplet']) for r in rs])
                vals.append(int(v.sum()))
            split_totals[k] = vals
            x = np.arange(3) + (j-.5)*.36
            color, hatch = cfg['categories'][k]
            bars = ax.bar(x, vals, .34, color=color, hatch=hatch, edgecolor='#444444',
                          linewidth=.5, label=k.split('#')[1])
            for b, n in zip(bars, vals):
                ax.text(b.get_x()+b.get_width()/2, n+max(vals)*.025, str(n), ha='center', fontsize=10)
        ax.set(xticks=np.arange(3), xticklabels=list(rows), ylabel='Số tuples')
        ax.set_ylim(0, ax.get_ylim()[1]*1.12)
        ax.set_title(pair[0].split('#')[0], pad=38)
        ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False, fontsize=10)
    fig.suptitle('AMBIENCE#STYLE_OPTIONS vắng hoàn toàn ở train', y=.98, fontsize=12)
    fig.text(.08, .09, 'Nguồn: toàn bộ gold mỗi split. Thanh và nhãn = số tuples quan sát, không có CI.', fontsize=10)
    fig.text(.08, .04, 'Split: train 2.284 câu · dev 200 câu · test 1.000 câu.', fontsize=10)
    save(fig, 'fig_style_options_counts')

    native = next(iter(load_nc(cfg).values()))
    predictions = {'gemma12b': quads_by_id(read_jsonl(cfg['prediction12'])),
                   'gemma27b': merged_preds(native['samples'], range(cfg['scsg']['n_samples']), cfg['scsg']['min_votes'])}
    majority = {}
    for a, c in counts['train'].items():
        if sum(c.values()) >= cfg['min_train_majority']:
            modes = [k for k, n in c.items() if n == max(c.values())]
            if len(modes) == 1:
                majority[a] = modes[0]
    shifted = {r['ID']: [q for q in r['Quadruplet'] if q['Aspect'].lower() in majority
                        and q['Category'].upper() != majority[q['Aspect'].lower()]] for r in rows['test']}
    summaries, train_rates = [], []
    fig, ax = plt.subplots(figsize=cfg['sizes'][2])
    fig.subplots_adjust(left=.12, right=.98, bottom=.29, top=.76)
    eligible = sum(map(len, shifted.values()))
    shifted_aspects = Counter(q['Aspect'].lower() for qs in shifted.values() for q in qs)
    match_draws = rng.integers(len(rows['test']), size=(cfg['n_boot'], len(rows['test'])))
    for j, model in enumerate(cfg['models']):
        matrix, ambiguous = [], 0
        assert set(predictions[model]) == {r['ID'] for r in rows['test']}
        for r in rows['test']:
            index, cells = defaultdict(set), np.zeros(3)
            for a, c, o, *_ in predictions[model][r['ID']]:
                index[a, o].add(c.upper())
            for q in shifted[r['ID']]:
                a = q['Aspect'].lower()
                if not index[a, q['Opinion'].lower()]:
                    continue
                cs, gold = index[a, q['Opinion'].lower()], q['Category'].upper()
                ambiguous += len(cs) > 1
                bucket = 0 if cs == {majority[a]} else 1 if cs == {gold} else 2
                cells[bucket] += 1
            matrix.append(cells)
        matrix = np.array(matrix); nums = matrix.sum(0); n = int(nums.sum()); shares = nums/n
        train_rates.append(shares[0])
        sampled = matrix[match_draws].sum(1); valid = sampled.sum(1) > 0
        lo, hi = bounds(sampled[valid]/sampled[valid].sum(1)[:, None])
        x = np.arange(3) + (j-.5)*.35
        ax.bar(x, shares, .33, color=cfg['model_colors'][model], label=f'{model}: n={n}/{eligible}')
        ax.errorbar(x, (lo+hi)/2, yerr=(hi-lo)/2, fmt='none', ecolor='#333333', capsize=4)
        for pos, v, u, num in zip(x, shares, hi, nums):
            ax.text(pos, u+.025, f'{v:.1%}\n({int(num)})', ha='center', fontsize=10)
        summaries.append(f'- {model}: khớp A/O {n}/{eligible}; theo train (sai) {int(nums[0])}/{n} ({shares[0]:.1%}); đúng gold {int(nums[1])}/{n} ({shares[1]:.1%}); khác/mơ hồ {int(nums[2])}/{n} ({shares[2]:.1%}); nhiều category {ambiguous}. CI 95% (%), thứ tự train/gold/khác: {np.round(np.array([lo, hi]).T*100, 1).tolist()}.')
    ax.set(xticks=np.arange(3), xticklabels=['Theo train\n(sai gold)', 'Đúng gold', 'Khác / mơ hồ'],
           ylabel='Tỷ lệ trên ca lệch nhãn khớp Aspect + Opinion', ylim=(0, 1.16))
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.set_yticks(np.arange(0, 1.01, .2))
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), frameon=False, ncol=2, fontsize=10)
    result = 'Theo train cũ: ' + ', '.join(f'{m} {v:.1%}' for m, v in zip(cfg['models'], train_rates))
    fig.suptitle(result, y=.98, fontsize=12)
    fig.text(.06, .14, f'Chỉ gold ≠ train-majority · train ≥ {cfg["min_train_majority"]} tuples · {eligible} ca lệch nhãn.', fontsize=10)
    fig.text(.06, .09, 'SC seed 0 · Khớp A/O lowercase trong cùng ID; không khớp: loại khỏi mẫu số.', fontsize=10)
    fig.text(.06, .04, 'Nguồn: gold + SCSG 12B/27B; CI 95% bootstrap theo 1.000 câu, 2.000 lần.', fontsize=10)
    save(fig, 'fig_models_follow_train')

    notes = ['## fig_category_shift_aspects']
    for aspect, tv in chosen:
        lo, hi = tv_intervals[aspect]
        notes.append(f'- {aspect}: TV={tv:.3f}, CI95%=[{lo:.3f}, {hi:.3f}].')
    notes += ['## fig_style_options_counts']
    notes += [f'- {category}: train/dev/test={values}.' for category, values in split_totals.items()]
    return notes + ['## fig_models_follow_train', *summaries]
