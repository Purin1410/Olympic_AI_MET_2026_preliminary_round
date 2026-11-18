"""SC curves and epoch histories, ported from the article figures."""
from pathlib import Path
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter
from . import analyze as old


def main(C, cfg, summary, FIGS, log):
    P = C['style']['colors']
    plt.rcParams.update({
        'font.family': C['style']['font'], 'font.size': C['style']['font_size'],
        'axes.titlesize': 12, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.axisbelow': True, 'axes.grid': True, 'axes.grid.axis': 'y',
        'grid.alpha': .18, 'figure.facecolor': 'white', 'svg.fonttype': 'none',
        'legend.frameon': False,
    })
    notes = []
    def vn(x, decimals=0):
        return f'{x:,.{decimals}f}'.replace(',', '_').replace('.', ',').replace('_', '.')

    def save(fig, name, title, caption):
        fig.suptitle(title, fontsize=12, y=.985)
        fig.text(.5, .025, caption, ha='center', va='bottom', fontsize=8.5)
        fig.tight_layout(rect=(0, .14 if caption.count('\n') > 1 else .105, 1, .93), h_pad=2, w_pad=2)
        for ext in ['png']:
            fig.savefig(FIGS / f'{name}.{ext}', dpi=C['style']['dpi'], facecolor='white')
        plt.close(fig)

    cfg_nc = copy.deepcopy(cfg)
    cfg_nc['scsg'].update(models={C['author_model']: cfg['scsg']['models'][C['author_model']]},
                          guidance=[C['guidance']], eval_seeds=C['author_seeds'])
    data = old.load_nc(cfg_nc)
    author = [data[(C['author_model'], C['guidance'], seed)] for seed in C['author_seeds']]
    gold_rows = old.read_jsonl(cfg['data']['gold_task3_test']); gold = old.quads_by_id(gold_rows)
    ids = sorted(gold); n_sent = len(ids); scfg = cfg['scorer']; n = C['n_samples']; mv = C['min_votes']
    predictions = {model: {'greedy': old.read_jsonl(Path(C['our_predictions']) / directory /
                       f'pred_greedy_s{C["main_seed"]}_test_nulldrop.jsonl')}
                   for model, directory in C['models'].items()}
    det = summary['scsg']['detail']
    recomputed = [old.score_task3(gold, old.merged_preds(d['samples'], range(n), mv), scfg)['cF1'] for d in author]
    np.testing.assert_allclose(np.mean(recomputed), det[C['author_model']]['A2_sc_at_k'][-1]['cF1'], atol=1e-12)
    np.testing.assert_allclose(np.mean(recomputed), det[C['author_model']]['A3_vote_threshold'][mv - 1]['cF1'], atol=1e-12)

    rng = np.random.default_rng(C['bootstrap_seed'])
    draws = rng.integers(0, n_sent, size=(C['n_boot'], n_sent))
    weights = np.array([np.bincount(x, minlength=n_sent) for x in draws], dtype=float)
    per_model = {C['author_model']: author}
    H = old.load_author_parser(cfg['ours']['helper_py'], ['parse_label_string', 'convert_tuples_to_output_format'])
    texts = {r['ID']: r['Text'] for r in gold_rows}
    for model, directory in C['models'].items():
        raw = old.read_jsonl(Path(C['our_predictions']) / directory / f'raw_sc_s{C["main_seed"]}.jsonl')
        samples = [{r['ID']: old.raw_to_quads(H, r['outputs'][j], r['ID'], texts[r['ID']], True, keep_case=True)[0]
                    for r in raw} for j in range(n)]
        greedy = old.quads_by_id(predictions[model]['greedy'])
        per_model[model] = [{'greedy': greedy, 'samples': samples}]
    ci_a3 = {}; ci_a4 = {}; delta_ci = {}
    for model, ds in per_model.items():
        all_counts = np.stack([[old.per_sentence(gold, old.merged_preds(d['samples'], range(n), threshold), scfg, ids)
                                for d in ds] for threshold in range(1, n + 1)])
        boot = (weights @ all_counts.transpose(2, 0, 1, 3).reshape(n_sent, -1)).reshape(C['n_boot'], n, len(ds), 4)
        bf = old.cf1_from_sums(boot).mean(axis=2)
        ci_a3[model] = np.percentile(bf, C['ci_percentiles'], axis=0)
        means = old.cf1_from_sums(all_counts.sum(axis=2)).mean(axis=1)
        np.testing.assert_allclose(means, [r['cF1'] for r in det[model]['A3_vote_threshold']], atol=1e-12)
        counts = np.zeros((n_sent, n, 2))
        for j, i in enumerate(ids):
            gk = {q[:3] for q in gold[i]}
            for d in ds:
                for key, vs in old.vote_table([sample.get(i, []) for sample in d['samples']]).items():
                    counts[j, len(vs) - 1] += [old.low_key(key) in gk, 1]
        for row in det[model]['A4_votes_vs_precision']:
            t = counts[:, row['votes'] - 1].sum(0)
            assert t[1] == row['n']; np.testing.assert_allclose(t[0] / t[1], row['precision'])
        bt = (weights @ counts.reshape(n_sent, -1)).reshape(C['n_boot'], n, 2)
        ratios = np.divide(bt[:, :, 0], bt[:, :, 1], out=np.full(bt.shape[:2], np.nan), where=bt[:, :, 1] > 0)
        ci_a4[model] = np.nanpercentile(ratios, C['ci_percentiles'], axis=0)
        d = ds[0]
        base = old.per_sentence(gold, d['greedy'], scfg, ids)
        kept = all_counts[mv - 1, 0]
        bd = old.cf1_from_sums(weights @ kept) - old.cf1_from_sums(weights @ base)
        delta_ci[model] = np.percentile(bd, C['ci_percentiles'])
    fig, ax = plt.subplots(figsize=(C['style']['single_width'], 5))
    for model in C['models'] | {C['author_model']: ''}:
        d = det[model]; rows = d['A2_sc_at_k']
        ax.errorbar([r['k'] for r in rows], [r['cF1'] for r in rows], yerr=[r['std'] for r in rows],
                    color=P[model], marker='o', capsize=3, label=f'{model} SC')
        ax.axhline(d['greedy_cF1'], color=P[model], ls='--', alpha=.6, lw=1)
        notes.append(f"`fig_A2_sc_at_k` {model}: S.scsg.detail.{model}, n_TEST={n_sent} câu; k/cF1=" + str([(r['k'], r['cF1']) for r in rows]) + f"; greedy={d['greedy_cF1']:.10f}; CI95% Δ(SC15−greedy) seed0=[{delta_ci[model][0]:.10f},{delta_ci[model][1]:.10f}].")
    ax.set(xlabel='Số lần sinh k (giữ tuple đạt đa số phiếu)', ylabel='cF1 trên TEST')
    ax.set_xticks([r['k'] for r in rows]); ax.legend(loc='lower right', fontsize=9)
    save(fig, 'fig_A2_sc_at_k', 'SC cải thiện cF1, lợi ích giảm dần khi tăng k',
         'TEST: 1.000 câu · out/v3 · nét đứt: greedy · thanh sai số: SD gốc\n'
         'CI95% Δ(SC15 − greedy), bootstrap theo câu, seed0:\n' +
         ' · '.join(f'{m.replace("gemma", "")}: [{vn(delta_ci[m][0],3)}; {vn(delta_ci[m][1],3)}]'
                    for m in C['models'] | {C['author_model']: ''}))
    fig, axes = plt.subplots(1, 3, figsize=(C['style']['multi_width'], 4.7), sharey=True)
    for ax, model in zip(axes, C['models'] | {C['author_model']: ''}):
        rows = det[model]['A3_vote_threshold']; x = [r['min_votes'] for r in rows]
        for metric, ls, color in [('cF1', '-', P[model]), ('cP', '--', P['neutral']), ('cR', ':', P['highlight'])]:
            ax.plot(x, [r[metric] for r in rows], ls=ls, color=color, label=metric, lw=1.7)
        ax.fill_between(x, *ci_a3[model], color=P[model], alpha=.18)
        ax.axvline(mv, color=P['neutral'], lw=.8); ax.set(title=model, xlabel=f'Phiếu tối thiểu / {n}')
        ax.set_xticks([1, 4, 8, 12, 15])
        r = rows[mv - 1]
        notes.append(f"`fig_A3_vote_threshold` {model}: S.scsg.detail + analyze.merge/score_task3; n={n_sent} câu × {len(per_model[model])} eval seed; tại {mv}/{n}: cP={r['cP']:.10f}, cR={r['cR']:.10f}, cF1={r['cF1']:.10f}; CI cF1=[{ci_a3[model][0,mv-1]:.10f},{ci_a3[model][1,mv-1]:.10f}].")
    axes[0].set_ylabel('cP / cR / cF1 trên TEST'); axes[0].legend(fontsize=9, loc='lower right')
    save(fig, 'fig_A3_vote_threshold', 'Tăng ngưỡng phiếu đổi recall lấy precision',
         'TEST: 1.000 câu · 4B/12B: 1 seed; 27B: trung bình 5 seed · đường gốc out/v3\nDải màu: CI95% cF1 bootstrap theo câu (giữ cố định eval seed) · vạch dọc: 8/15')
    fig, ax = plt.subplots(figsize=(C['style']['single_width'], 4.7))
    for model in C['models'] | {C['author_model']: ''}:
        rows = det[model]['A4_votes_vs_precision']; x = [r['votes'] for r in rows]
        ax.plot(x, [r['precision'] for r in rows], marker='o', color=P[model], label=model)
        ax.fill_between(x, ci_a4[model][0, np.array(x)-1], ci_a4[model][1, np.array(x)-1], color=P[model], alpha=.13)
        notes.append(f"`fig_A4_votes_vs_precision` {model}: S.scsg.detail + analyze.vote_table; n={sum(r['n'] for r in rows)} ứng viên/{n_sent} câu × {len(per_model[model])} eval seed; (phiếu,n,precision) đầu/cuối="+str([(r['votes'],r['n'],r['precision']) for r in [rows[0],rows[-1]]])+'.')
    ax.axvline(mv - .5, color=P['neutral'], ls='--'); ax.set(xlabel=f'Số phiếu đồng thuận / {n}', ylabel='Tỷ lệ ứng viên khớp gold (A, C, O)')
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.legend(); ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
    save(fig, 'fig_A4_votes_vs_precision', 'Ứng viên nhiều phiếu thường khớp gold hơn',
         'TEST: 1.000 câu · 4B/12B: 1 seed; 27B: gộp 5 seed · out/v3 + raw samples\nDải màu: CI95% theo câu · precision ở đây chỉ xét A,C,O, không phải cP')
    fig, axes = plt.subplots(1, 2, figsize=(C['style']['multi_width'], 5))
    for j, key in enumerate(cfg['logsigma']['hist_keys']):
        h = log[key]['hist']; epochs = [r['ep'] for r in h]
        color = P['LogSigma'] if key in C['sigma_paper_runs'] else P['neutral']
        marker = ['o', 's', '^'][j % 3]
        alpha = [1, .7, .45][j % 3]
        axes[1].plot(epochs, [r['dev_rmse'] for r in h], color=color, marker=marker, markersize=3,
                     ls='-' if key in C['sigma_paper_runs'] else '--', alpha=alpha,
                     label=key.replace('seed','').replace('_',' / '))
        if key in C['sigma_paper_runs']:
            axes[0].plot(epochs, [r['var_v'] for r in h], color=color, marker=marker, markersize=3,
                         alpha=alpha, label=key.replace('seed','').replace('_paper',''))
            axes[0].plot(epochs, [r['var_a'] for r in h], color=color, marker=marker, markersize=3, alpha=alpha, ls='--')
        best = min(h, key=lambda x: x['dev_rmse']); ref = summary['logsigma']['B2_final'][key]
        np.testing.assert_allclose([h[0]['var_v'], h[-1]['var_a'], best['dev_rmse']],
                                   [ref['var_v_ep1'], ref['var_a_last'], ref['best_dev']], atol=1e-12)
        notes.append(f"`fig_B2_sigma_dev` {key}: n={len(h)} epoch, best ep={best['ep']}, best DEV RMSE={best['dev_rmse']:.10f}; σ²V đầu/cuối={h[0]['var_v']:.10f}/{h[-1]['var_v']:.10f}, σ²A={h[0]['var_a']:.10f}/{h[-1]['var_a']:.10f}; nguồn log hist, so với S.B2_final.")
    axes[0].set(yscale='log', title='a · σ² giảm trong ba run paper-config', xlabel='Epoch', ylabel='σ² (thang log)')
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [Line2D([],[],color=P['neutral'],label='σ²_V'), Line2D([],[],color=P['neutral'],ls='--',label='σ²_A')]
    axes[0].legend(handles=handles, fontsize=9, ncol=2)
    axes[1].set(title='b · DEV RMSE tùy seed và cấu hình', xlabel='Epoch', ylabel='RMSE_VA trên DEV')
    axes[1].set_ylim(top=cfg['logsigma']['dev_plot_ymax']); axes[1].legend(fontsize=8, ncol=2, loc='upper right')
    for ax in axes: ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    save(fig, 'fig_B2_sigma_dev', 'LogSigma: σ² được học; DEV quyết định checkpoint',
         'LogSigma · TRAIN/DEV task1 · DEV: 200 câu, 340 pair · lịch sử gốc logsigma_result.json\nTrái: 3 run paper-config; phải: đủ 7 run gốc · đường mô tả theo epoch, không phải CI')
    return notes
