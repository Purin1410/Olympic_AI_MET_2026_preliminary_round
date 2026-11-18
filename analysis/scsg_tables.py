

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

def main(cfg, data, out):
    s = data['scsg']
    models, colors = cfg['models'], cfg['colors']
    n_sent = s['n_sent']
    n_gold = s['n_gold']
    plt.rcParams.update(cfg['theme'])
    plt.rcParams.update({'font.family': cfg['font'], 'font.size': 10, 'axes.labelsize': 11,
                         'axes.titlesize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid.axis': 'y', 'grid.color': '#EEEEEE',
                         'grid.linewidth': .6, 'svg.fonttype': 'none'})
    notes = []
    def save(fig, name, title, caption, rect=(0, .11, 1, .88)):
        fig.tight_layout(rect=rect)
        fig.suptitle(title, fontsize=12, fontweight='bold', y=.98)
        fig.text(.02, .025, caption, fontsize=10, va='bottom', linespacing=1.5)
        for ext in ('png',):
            fig.savefig(out / f'{name}.{ext}', dpi=cfg['dpi'], facecolor='white')
        plt.close(fig)


    fig, ax = plt.subplots(figsize=(7, 6.6))
    keys = list(cfg['error_labels'])
    y = np.arange(len(keys))
    matched = []
    for j, m in enumerate(models):
        counts = s['detail'][m]['A7_errors']['counts']
        assert set(counts) == set(keys), f'Unexpected A7 categories: {m}'
        vals = [counts[k] for k in keys]
        bars = ax.barh(y + (j - 1) * .24, vals, height=.22,
                       color=colors[m], edgecolor='white', label=m)
        ax.bar_label(bars, padding=4, fontsize=10)
        n = int(s['detail'][m]['A6_breakdown']['sc15']['TP_cat'])
        matched.append(f'{m}: {n:,}')
    ax.set_yticks(y, [f'{cfg["error_labels"][k]}\n({k})' for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(0, max(s['detail'][m]['A7_errors']['counts'][k]
                       for m in models for k in keys) * 1.16)
    ax.set_xlabel('Số tuple theo loại lỗi A7')
    ax.legend(loc='lower right', frameon=False, fontsize=10)
    caption = (f"Test: {n_sent:,} câu, {n_gold:,} tuple gold · SC {cfg['sc_samples']} mẫu · seed {cfg['seed']}\n"
               'Nguồn: summary.json, A7/A6 · khớp (Aspect, Category, Opinion), chưa trừ VA:\n'
               + '  ·  '.join(matched))
    save(fig, 'fig_scsg_error_types', 'Bỏ sót gold là loại lỗi phổ biến nhất', caption,
         rect=(0, .16, 1, .92))
    notes += ['## fig_scsg_error_types (figs/fig_scsg_error_types.png + .svg)',
              'Bỏ sót gold là loại lỗi A7 lớn nhất ở cả ba model.',
              'Thứ tự số: FN_missed / FP_opinion / FP_category / FP_aspect / FP_spurious; khớp ACO:']
    for m in models:
        d = s['detail'][m]
        notes.append(f"- {m}: " + ' / '.join(str(d['A7_errors']['counts'][k]) for k in keys)
                     + f"; TP_cat = {d['A6_breakdown']['sc15']['TP_cat']:.0f}.")
    notes += ['A7 dùng tập khóa ACO và ghép gần đúng 2/3 thành phần; FN_missed không phải toàn bộ FN của scorer; các nhóm không tạo một phân hoạch gold+prediction.', '']


    rows = {m: {r['category']: r for r in s['by_category_sc'][m]} for m in models}
    focus = cfg['highlight_category']
    cats = [c for c, r in rows[models[0]].items()
            if r['n_gold'] >= cfg['category_min_support'] or c == focus]
    assert all(c in rows[m] for c in cats for m in models)
    assert focus in cats, 'Highlighted category missing from precomputed table'
    cats.sort(key=lambda c: np.mean([rows[m][c]['cF1'] for m in models]), reverse=True)
    y = np.arange(len(cats))
    fig, (ax, counts_ax) = plt.subplots(1, 2, figsize=(10, 6.8), sharey=True,
                                      gridspec_kw={'width_ratios': [3.6, 1.3]})
    for j, (m, marker) in enumerate(zip(models, ['o', 's', '^'])):
        ax.scatter([rows[m][c]['cF1'] for c in cats], y + (j - 1) * .20,
                   color=colors[m], marker=marker, s=45, label=m, zorder=3)
    for a in (ax, counts_ax):
        a.axhspan(cats.index(focus) - .44, cats.index(focus) + .44,
                  color=colors['highlight'], alpha=.17, zorder=0)
    ax.set_yticks(y, cats)
    ax.invert_yaxis()
    ax.set_xlim(-.025, 1)
    ax.set_xticks(np.arange(0, 1.01, .2))
    ax.set_xlabel('cF1 (cao hơn tốt hơn)')
    ax.set_title('SC · xếp theo cF1 trung bình ba model', fontsize=11, pad=14)
    ax.legend(loc='lower right', frameon=False, fontsize=10)
    for tick in ax.get_yticklabels():
        if tick.get_text() == focus:
            tick.set_fontweight('bold')
    counts_ax.set_xlim(0, 2)
    counts_ax.set_title('Số tuple', fontsize=11, pad=14)
    counts_ax.text(.45, 1.015, 'Train', transform=counts_ax.transAxes, ha='center')
    counts_ax.text(.88, 1.015, 'Test', transform=counts_ax.transAxes, ha='center')
    for i, c in enumerate(cats):
        r = rows[models[0]][c]
        counts_ax.text(.9, i, f"{r['n_train']:,}", ha='center', va='center',
                       fontweight='bold' if c == focus else 'normal')
        counts_ax.text(1.76, i, str(r['n_gold']), ha='center', va='center')
    counts_ax.tick_params(axis='y', left=False, labelleft=False)
    counts_ax.set_xticks([])
    counts_ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    caption = (f"Test · SC · seed {cfg['seed']} · support ≥ {cfg['category_min_support']} · nguồn: summary.json, by_category_sc\n"
               'Nền vàng: ambience#style_options — 0 tuple train, 37 tuple test, cF1 = 0 ở cả ba model.\n'
               'Bảng nguồn không có CI theo category; các điểm là ước lượng đơn trên test.')
    save(fig, 'fig_scsg_by_category', 'Category chưa có trong train đạt cF1 = 0', caption,
         rect=(0, .16, 1, .92))
    notes += ['## fig_scsg_by_category (figs/fig_scsg_by_category.png + .svg)',
              'ambience#style_options có 0 tuple train và cF1 = 0 ở cả ba model; support test 37 ≥ 30, không cần ngoại lệ.',
              'Xếp theo trung bình cF1 ba model (chỉ để sắp hàng); số sau theo gemma4b / gemma12b / gemma27b:']
    for c in cats:
        r = rows[models[0]][c]
        notes.append(f"- {c}: train/test {r['n_train']}/{r['n_gold']}; cF1 "
                     + ' / '.join(f"{rows[m][c]['cF1']:.4f}" for m in models) + '.')
    notes += ['Không có CI category trong summary.json; không suy luận ý nghĩa thống kê từ thứ hạng.', '']


    fig, ax = plt.subplots(figsize=(7, 5.4))
    notes += ['## fig_sc_decompose (figs/fig_sc_decompose.png + .svg)',
              'Tuple selection có hiệu ứng riêng lớn hơn VA smoothing ở gemma12b và gemma27b; gemma4b theo chiều ngược lại.',
              'Giữ nguyên delta_*_only: đây là hai phép ghép chéo cùng so với greedy, không phải hai phần cộng chính xác. Kim cương là delta_total; không chuẩn hóa hay gán phần dư cho VA.']
    for i, m in enumerate(models):
        r = s['SC_decompose'][m]
        selection = r['delta_tuple_selection_only']
        smoothing = r['delta_VA_smoothing_only']
        total = r['delta_total']
        ax.barh(i, selection, color=colors[m], height=.48, edgecolor='white')
        ax.barh(i, smoothing, left=selection, color=colors[m], alpha=.38,
                hatch='///', height=.48, edgecolor=colors[m])
        ax.text(selection / 2, i, f'{selection:.5f}', ha='center', va='center',
                color='white' if m != 'gemma4b' else '#222222', fontsize=10)
        ax.text(selection + smoothing / 2, i, f'{smoothing:.5f}',
                ha='center', va='center', fontsize=10)
        ax.plot(total, i + .34, 'D', color=colors['highlight'], markersize=6)
        ax.annotate(f'Tổng {total:+.5f}', (total, i + .34), xytext=(8, 0),
                    textcoords='offset points', ha='left', va='center', fontsize=10)
        notes.append(f"- {m} (n_seed={r['n_seeds']}): selection {selection:+.8f}; smoothing {smoothing:+.8f}; total {total:+.8f}.")
    ax.set_yticks(range(len(models)),
                  [f"{m}\n{ s['SC_decompose'][m]['n_seeds']} eval seed" for m in models])
    ax.set_ylim(2.85, -.6)
    ax.set_xlim(0, max(s['SC_decompose'][m]['delta_total'] for m in models) * 1.25)
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    ax.set_xlabel('ΔcF1 so với greedy (đơn vị cF1)')
    handles = [Patch(color=colors['neutral'], label='Tuple selection riêng'),
               Patch(facecolor='#DDDDDD', edgecolor=colors['neutral'], hatch='///', label='VA smoothing riêng'),
               Line2D([], [], color=colors['highlight'], marker='D', linestyle='', label='Tổng gain thực tế')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .90),
               ncol=2, frameon=False, fontsize=10)
    caption = ('Nguồn: summary.json, SC_decompose · trung bình eval seed như nhãn\n'
               'Hai hiệu ứng riêng không cộng chính xác thành tổng; xem kim cương.\n'
               'Không có CI cho phép tách trong bảng nguồn; không diễn giải thành tỷ lệ đóng góp.')
    save(fig, 'fig_sc_decompose', 'Tuple selection nổi trội ở gemma12b và gemma27b', caption,
         rect=(0, .19, 1, .78))
    return notes
