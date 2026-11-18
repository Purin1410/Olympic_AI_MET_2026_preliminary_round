"""Re-run the article analysis on CPU from the supplied predictions."""
import copy
import json
import os
from pathlib import Path
import random
import re
import statistics
from collections import Counter

import yaml

REPO = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(REPO / '.cache' / 'matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

from common.fetch import fetch_repo
from . import analyze as a
from . import error_core, logsigma_difficulty, scsg_tables, error_slices
from . import annotation_shift, fixes_v5, va_cross


def summary(cfg):
    """Calculate only the original summary fields consumed by these figures."""
    sc, scorer = cfg['scsg'], cfg['scorer']
    n, mv = sc['n_samples'], sc['min_votes']
    rows = a.read_jsonl(cfg['data']['gold_task3_test'])
    gold = a.quads_by_id(rows)
    parser = a.load_author_parser(cfg['ours']['helper_py'],
                                  ['parse_label_string', 'convert_tuples_to_output_format'])
    data = a.load_nc(cfg)
    groups = {sc['main_model']: [data[(sc['main_model'], sc['main_guidance'], seed)]
                                for seed in sc['eval_seeds']]}
    for model, directory in cfg['ours']['models'].items():
        raw = a.read_jsonl(Path(directory) / f'raw_sc_s{sc["main_seed"]}.jsonl')
        texts = {r['ID']: r['Text'] for r in rows}
        samples = [{r['ID']: a.raw_to_quads(parser, r['outputs'][j], r['ID'], texts[r['ID']],
                                           cfg['ours']['null_drop'], keep_case=True)[0]
                    for r in raw} for j in range(n)]
        greedy = a.quads_by_id(a.read_jsonl(Path(directory) /
                              f'pred_greedy_s{sc["main_seed"]}_test_nulldrop.jsonl'))
        entry = {'greedy': greedy, 'samples': samples}
        data[(model, cfg['ours']['label'], sc['main_seed'])] = entry
        groups[model] = [entry]
    rng = random.Random(cfg['seed'])
    details = {}
    for model, seeds in groups.items():
        print(f'Calculating SC curves: {model}', flush=True)
        details[model] = a.detail(seeds, seeds[sc['main_seed']], gold, cfg, rng)
    train_cats = Counter(q['Category'].lower()
                         for r in a.read_jsonl(cfg['data']['train_task1'])
                         for q in r.get('Quadruplet', []))
    result = {'detail': details, 'n_sent': len(rows), 'n_gold': sum(map(len, gold.values())),
              'A1_greedy_vs_sc': a.a1_rows(data, gold, scorer, n, mv),
              'A6b_parser_ceiling': a.parser_ceiling(parser, rows, gold, scorer),
              'by_category_sc': {m: a.by_category(gold, d['_sc0'], scorer,
                                sc['category_min_support'], train_cats) for m, d in details.items()},
              'SC_decompose': {}}
    for model, seeds in groups.items():
        per = [a.sc_decompose(gold, d, scorer, n, mv) for d in seeds]
        result['SC_decompose'][model] = {'n_seeds': len(seeds), **{
            k: statistics.mean(r[k] for r in per) for k in
            ['delta_total', 'delta_VA_smoothing_only', 'delta_tuple_selection_only']}}
    histories, finals = {}, {}
    for key in cfg['logsigma']['hist_keys']:
        base = Path(cfg['logsigma']['runs']) / key
        history = json.loads((base / 'history.json').read_text())
        report = json.loads((base / 'summary.json').read_text())
        histories[key] = {'hist': [dict(r, ep=r['epoch']) for r in history]}
        finals[key] = {'var_v_ep1': history[0]['var_v'], 'var_a_last': history[-1]['var_a'],
                       'best_dev': report['best_dev_rmse']}
    return {'scsg': result, 'logsigma': {'B2_final': finals}}, histories


def main():
    os.chdir(REPO)
    cfg = yaml.safe_load((Path(__file__).with_name('config.yaml')).read_text())
    for repo in cfg['repositories'].values():
        fetch_repo(repo['url'], repo['commit'], repo['directory'])
    out = Path(cfg['output'])
    figs = out / 'figs'
    figs.mkdir(parents=True, exist_ok=True)
    for section in ['error_core', 'logsigma_difficulty', 'scsg_tables', 'error_slices',
                    'annotation_shift', 'fixes_v5', 'va_cross']:
        spec = cfg[section]
        font = spec.get('plot', spec.get('style', spec))
        try:
            font_manager.findfont(font['font'], fallback_to_default=False)
        except ValueError:
            font['font'] = matplotlib.rcParamsDefault['font.family'][0]
    data, histories = summary(cfg)
    numbers = {}
    for module, args in [
        (error_core, (cfg['error_core'], cfg, data, figs)),
        (logsigma_difficulty, (cfg['logsigma_difficulty'], figs)),
        (scsg_tables, (cfg['scsg_tables'], data, figs)),
        (error_slices, (cfg['error_slices'], cfg, figs)),
        (annotation_shift, (cfg['annotation_shift'], figs)),
        (fixes_v5, (cfg['fixes_v5'], cfg, data, figs, histories)),
        (va_cross, (cfg['va_cross'], cfg, figs)),
    ]:
        print(f'Plotting {module.__name__}', flush=True)
        plt.rcdefaults()
        figure = None
        for line in module.main(*copy.deepcopy(args)):
            match = re.match(r"(?:## |`)(fig_\w+)", line)
            if match:
                figure = match[1]
                numbers.setdefault(figure, [])
                if line.startswith('`'):
                    numbers[figure].append(line.split('`', 2)[2].strip())
            elif figure and (line.startswith(('- ', '  CI95%', 'n=')) or module is error_slices):
                numbers[figure].append(line.removeprefix('- ').strip())

    notes = ['# Analysis numbers', '',
             'Fixed predictions; 2,000 sentence bootstrap resamples, seed 0. '
             '4B/12B: one eval seed; 27B: five for SC curves, seed 0 for error slices.', '',
             '| Figure | Key numbers from the original notes |', '|---|---|']
    notes += [f"| {figure} | {'; '.join(values)} |" for figure, values in numbers.items()]
    (out / 'numbers.md').write_text('\n'.join(notes) + '\n', encoding='utf-8')
    print(f'Wrote 15 PNGs to {figs} and numbers.md to {out}', flush=True)


if __name__ == '__main__':
    main()
