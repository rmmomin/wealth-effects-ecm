"""Research outputs for a frozen-coefficient S&P AI/other ECM split.

Capitalization observations are supplied locally and excluded from public Git.
"""
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ecm_ai_split import (
    COMPONENTS, bootstrap_ai_components, capitalization_bridge,
    decompose_ai, load_frozen_fits,
)
from .ecm_gdp import bridge_channels, converters
from .ecm_publication import ROOT, LABELS, _hash, _json, _read_quarterly, _save
from .note454_ecm import DEFAULT_SPEC, decompose_ecm
from .note454_ratio_ecm import decompose_ratio_ecm

CHANNEL_LABELS = {
    'ai_sp': 'AI-linked S&P', 'non_ai_sp': 'Other S&P',
    'index_cap_basis': 'Index/cap basis', 'unallocated_wealth': 'Remaining wealth / inherited effects',
    'original_wealth': 'Original total wealth',
}
COLORS = {'ai_sp': '#246c9d', 'non_ai_sp': '#dd9839',
          'index_cap_basis': '#9a65a4', 'unallocated_wealth': '#adb4bd'}
KEYS = ['model', 'variant', 'split_start', 'quarter', 'channel']
PERIOD_KEYS = ['model', 'variant', 'split_start', 'period', 'channel']


def _intervals(frame, keys, value):
    result = frame.groupby(keys)[value].quantile([.025, .5, .975]).unstack().reset_index()
    result.columns = keys + ['ci_low_pp', 'bootstrap_median_pp', 'ci_high_pp']
    return result


def _period(frame, draw=False):
    result = frame.copy()
    quarters = pd.PeriodIndex(result.quarter, freq='Q')
    result['period'] = [str(q.year) if q.year < 2026 else '2026 H1' for q in quarters]
    keys = PERIOD_KEYS + (['draw'] if draw else [])
    return result.groupby(keys, as_index=False).agg(
        mean_quarterly_annualized_gdp_pp=('gross_gdp_growth_pp', 'mean'),
        quarters=('quarter', 'nunique'))


def build_ai_split(output_dir=ROOT / 'results/ai_split', config_path=ROOT / 'config/ai_split.json'):
    """Produce a subdivision, not a new fit, with locally supplied capitalization.

    All original equations, sample, coefficients and saved joint bootstrap
    vectors remain frozen. Original wealth and every other channel are checked
    over the complete 1990Q1 onward recursion before post-2022 outputs are saved.
    """
    output, config_path = Path(output_dir).resolve(), Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    ecm_path = ROOT / config['ecm_config']
    ecm_config = json.loads(ecm_path.read_text())
    fixed_keys = ('start', 'end', 'short_lags', 'consumption_lags', 'hac_lags')
    if any(ecm_config['model'].get(k, DEFAULT_SPEC[k]) != DEFAULT_SPEC[k] for k in fixed_keys):
        raise ValueError('Saved point coefficients and draws belong to the original 1990–2019 specification')
    if config.get('fit_new_parameters') is not False or config.get('confidence_level') != .95:
        raise ValueError('This runner supports frozen coefficients and 95% intervals only')
    if config.get('bootstrap_draws') != 399:
        raise ValueError('The published split retains all 399 saved joint draws per model')
    if config.get('ownership_timing') != 'lagged one quarter' or config.get('market_cap_total_divisor') != 1000:
        raise ValueError('Changed ownership/units require an explicitly revised bridge')
    protected = [ROOT / folder for folder in ('data', 'config', 'wealth_effects', 'scripts', 'tests', 'docs')]
    if output in (ROOT, ROOT / 'results') or any(output == p or output.is_relative_to(p) for p in protected):
        raise ValueError('Use a separate output folder; preserve original results and source inputs')
    paths = {name: ROOT / value for name, value in config['data'].items()}
    if not paths['capitalization'].is_file():
        raise FileNotFoundError(
            'AI/non-AI market-cap inputs are excluded from the public repository. '
            'Supply '+config['data']['capitalization']+' locally using the schema in docs/AI_ECM_SPLIT.md.')
    original_hashes = {name: _hash(path) for name, path in paths.items()}
    panel, accounting, inputs = (_read_quarterly(paths[name]) for name in ('panel', 'gdp', 'capitalization'))
    fits = load_frozen_fits(pd.read_csv(paths['coefficients']), ecm_config['model'])
    parameters = pd.read_csv(paths['bootstrap'])
    if set(parameters.model.unique()) != set(fits):
        raise ValueError('Saved parameters must contain exactly the two frozen specifications')
    start, end = config['decomposition_start'], config['data_end']
    if start != ecm_config['decomposition_start'] or end != ecm_config['data_end']:
        raise ValueError('Preserve the original anchored decomposition horizon')
    dates = pd.period_range(start, end, freq='Q')
    conversion = converters(panel, dates, accounting)
    gdp_factor = pd.Series(conversion.primary_gdp_pp_per_log_growth.to_numpy(), index=dates.astype(str))
    first = config['split_start']
    scenarios = [first, config['sensitivity_split_start']]
    if len(set(scenarios)) != 2:
        raise ValueError('Use distinct primary and sensitivity split dates')
    ledgers, tables, channels, periods, checks = [], [], [], [], []
    for split in scenarios:
        ledger = capitalization_bridge(panel, inputs, split, end)
        ledgers.append(ledger.reset_index().assign(split_start=split))
    output.mkdir(parents=True, exist_ok=True)
    for model in ('ratio_ecm', 'log_ecm'):
        fit = fits[model]
        original = (decompose_ratio_ecm if model == 'ratio_ecm' else decompose_ecm)(fit, panel, start, end)
        original_gdp = bridge_channels(panel, original, accounting)
        wealth = original_gdp.loc[original_gdp.channel == 'wealth'].assign(channel='original_wealth')
        pdraws = parameters.loc[parameters.model == model].drop(columns='model')
        if pdraws.draw.nunique() != 399:
            raise ValueError('Missing saved joint draw; no filtering or refitting')
        for variant in config['variants']:
            for split in scenarios:
                decomposition = decompose_ai(fit, panel, inputs, model, split, variant, start, end)
                bridged = bridge_channels(panel, decomposition, accounting)
                split_wealth = bridged.loc[bridged.channel.isin(COMPONENTS)]
                error = float((split_wealth.groupby('quarter').gross_gdp_growth_pp.sum()
                    - wealth.set_index('quarter').gross_gdp_growth_pp).abs().max())
                other = bridged.loc[~bridged.channel.isin(COMPONENTS)].set_index(['quarter', 'channel'])
                original_other = original_gdp.loc[original_gdp.channel != 'wealth'].set_index(['quarter', 'channel'])
                other_error = float((other.gross_gdp_growth_pp - original_other.gross_gdp_growth_pp).abs().max())
                if error > 1e-9 or other_error > 1e-12:
                    raise ValueError('Subdivision changed the published accounting')
                full = pd.concat([split_wealth, wealth], ignore_index=True)
                full = full.assign(model=model, variant=variant, split_start=split)
                full = full.loc[full.quarter >= first].copy()
                full['display_label'] = full.channel.map(CHANNEL_LABELS)
                draws = bootstrap_ai_components(fit, panel, inputs, pdraws, model, split, variant, start, end)
                draw_error = float((draws[list(COMPONENTS)].sum(axis=1) - draws.original_wealth).abs().max())
                draws = draws.loc[draws.quarter >= first].copy()
                draws = draws.melt(id_vars=['draw', 'quarter'], value_vars=[*COMPONENTS, 'original_wealth'],
                                   var_name='channel', value_name='log_growth_contribution')
                draws['gross_gdp_growth_pp'] = draws.log_growth_contribution * draws.quarter.map(gdp_factor)
                draws = draws.assign(model=model, variant=variant, split_start=split)
                full = full.merge(_intervals(draws, KEYS, 'gross_gdp_growth_pp'), on=KEYS, validate='one_to_one')
                period = _period(full)
                path_period = _period(draws, draw=True)
                period = period.merge(_intervals(path_period, PERIOD_KEYS, 'mean_quarterly_annualized_gdp_pp'),
                                      on=PERIOD_KEYS, validate='one_to_one')
                period['measure'] = 'Mean quarterly annualized GDP contributions; 2026 H1 is two quarters, not full year'
                tables.append(full); periods.append(period)
                channels.append(bridged.loc[bridged.quarter >= first].assign(model=model, variant=variant, split_start=split))
                checks.append({'model': model, 'variant': variant, 'split_start': split,
                               'point_wealth_reconciliation_max_gdp_pp': error,
                               'other_channels_max_change_gdp_pp': other_error,
                               'draw_wealth_reconciliation_max_log_growth': draw_error,
                               'joint_draws_retained': int(pdraws.draw.nunique())})
    quarterly, period_table = pd.concat(tables, ignore_index=True), pd.concat(periods, ignore_index=True)
    exports = {'capitalization_ownership_ledger.csv': pd.concat(ledgers, ignore_index=True),
               'quarterly_wealth_split.csv': quarterly,
               'quarterly_all_channels.csv': pd.concat(channels, ignore_index=True),
               'period_summary.csv': period_table}
    for name, table in exports.items():
        table.to_csv(output / name, index=False, float_format='%.15g')
    for name, path in paths.items():
        if _hash(path) != original_hashes[name]:
            raise RuntimeError('Original input changed during run: ' + name)
    diagnostics = {'checks': checks, 'new_estimation_calls': 0, 'api_calls': 0,
        'original_base': str(dates[0] - 1), 'primary_split_base': str(pd.Period(first, freq='Q') - 1),
        'unexplained_or_residual_wealth_is_not_housing_only': True,
        'uncertainty': '399 saved joint two-stage coefficient vectors per model; exploratory conditional percentile intervals; ownership, basket, baseline and vintage are held fixed within each scenario',
        'observational_status': 'Actual aggregate quarterly observations; user-supplied market-cap classification and household-incidence mapping are not separately estimated',
        'replication_status': 'Extension of the ECM adaptation, not a new Banque de France numerical replication or identified AI causal effect',
        'source_status': 'Original baseline sources public; new market-cap workbook is user-supplied with unverified public vendor/license provenance. AI/non-AI source observations and the detailed capitalization/ownership ledger are excluded from the public repository.',
        'gdp_interpretation': 'Gross consumption-channel allocation of published PCE GDP contribution; no imports or general-equilibrium offsets',
        'gdp_vintage': 'August 26, 2026 frozen baseline; existing AI inputs from later September snapshot',
        'ecm_diagnostics': 'Log cointegration null not rejected (p about .883); residual serial correlation remains in both forms; ratio cointegration not formally validated'}
    _json(output / 'diagnostics.json', diagnostics)
    _plots(output, quarterly, period_table, first, config['sensitivity_split_start'],
           config.get('display_start', '2023Q1'), config.get('display_model', 'ratio_ecm'))
    _results(output, quarterly, period_table, first, config.get('display_start', '2023Q1'),
             config.get('display_model', 'ratio_ecm'))
    sources = [ROOT / 'scripts/run_ai_ecm_split.py', Path(__file__).resolve(), ROOT / 'wealth_effects/ecm_ai_split.py',
               ROOT / 'wealth_effects/ecm_gdp.py', ROOT / 'wealth_effects/ecm_publication.py',
               ROOT / 'wealth_effects/note454_ratio.py', ROOT / 'wealth_effects/note454_ecm.py',
               ROOT / 'wealth_effects/note454_ratio_ecm.py', ROOT / 'wealth_effects/note454_gdp_draws.py']
    config_key = str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else 'external_configuration'
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'configuration': config,
        'input_sha256': {str(paths[name].relative_to(ROOT)): value for name, value in original_hashes.items()},
        'configuration_sha256': {config_key: _hash(config_path), str(ecm_path.relative_to(ROOT)): _hash(ecm_path)},
        'source_sha256': {str(path.relative_to(ROOT)): _hash(path) for path in sources},
        'dependencies': {name: version(name) for name in ('numpy','pandas','scipy','statsmodels','matplotlib')},
        'new_estimation_calls': 0,
        'excluded_from_public_git': [config['data']['capitalization'], 'results/ai_split/capitalization_ownership_ledger.csv'],
        'output_sha256': {p.name: _hash(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'run_manifest.json'}}
    _json(output / 'run_manifest.json', manifest)
    return {'output': str(output), 'quarterly_rows': len(quarterly), 'period_rows': len(period_table),
            'checks': len(checks), 'new_estimation_calls': 0, 'max_wealth_reconciliation_gdp_pp': max(c['point_wealth_reconciliation_max_gdp_pp'] for c in checks)}


def _plots(output, quarterly, periods, first, alternate,
           display_start='2023Q1', display_model='ratio_ecm'):
    """Present one specification from the chosen date without reanchoring it."""
    import os
    import tempfile
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'wealth-ecm-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    display_quarter = pd.Period(display_start, freq='Q')
    if display_model not in LABELS:
        raise ValueError('Unknown display model')
    if display_quarter < pd.Period(first, freq='Q') or display_quarter > pd.Period(quarterly.quarter.max(), freq='Q'):
        raise ValueError('Display start must lie within the calculated split horizon')
    primary = quarterly.loc[(quarterly.variant == 'ownership_bridge')
        & (quarterly.split_start == first) & (quarterly.model == display_model)
        & (pd.PeriodIndex(quarterly.quarter, freq='Q') >= display_quarter)]
    table = primary.pivot(index='quarter', columns='channel', values='gross_gdp_growth_pp').sort_index()
    x = np.arange(len(table))
    model_label = LABELS[display_model]
    anchor = str(pd.Period(first, freq='Q') - 1)
    fig, ax = plt.subplots(figsize=(11, 6.2))
    positive, negative = np.zeros(len(table)), np.zeros(len(table))
    for channel in COMPONENTS:
        values = table[channel].to_numpy()
        ax.bar(x, np.maximum(values, 0), bottom=positive, color=COLORS[channel], label=CHANNEL_LABELS[channel], width=.72)
        ax.bar(x, np.minimum(values, 0), bottom=negative, color=COLORS[channel], width=.72)
        positive += np.maximum(values, 0); negative += np.minimum(values, 0)
    ax.plot(x, table.original_wealth, color='#20242b', lw=1.8, marker='o', ms=4, label='Original wealth total')
    ax.axhline(0, color='#444', lw=.7); ax.grid(axis='y', alpha=.18)
    ax.set_ylabel('Contribution to annualized real GDP growth, pp')
    ax.set_xticks(x, table.index, rotation=45, ha='right')
    ax.legend(frameon=False, ncol=2, fontsize=9, loc='upper left', bbox_to_anchor=(0, -.18))
    ax.set_title(model_label + ' | AI-linked and other S&P wealth contributions', loc='left', fontsize=15, pad=15)
    fig.text(.5, .015, 'Display starts '+display_start+'; the ownership bridge still uses a '+anchor+' baseline. Original ECM totals are unchanged.\n'
        'Gross consumption-channel allocation, not a causal GDP estimate. Remaining wealth includes inherited effects and mapping residuals.',
        ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .11, 1, 1)); _save(fig, output, 'wealth_split', plt)

    fig, ax = plt.subplots(figsize=(11, 5.4))
    for channel in ('ai_sp', 'non_ai_sp'):
        values = primary.loc[primary.channel == channel].sort_values('quarter')
        color = COLORS[channel]
        ax.fill_between(x, values.ci_low_pp.to_numpy(), values.ci_high_pp.to_numpy(), color=color, alpha=.16)
        ax.plot(x, values.gross_gdp_growth_pp, color=color, lw=2, marker='o', ms=3, label=CHANNEL_LABELS[channel])
    ax.axhline(0, color='#444', lw=.7); ax.grid(axis='y', alpha=.18)
    ax.set_ylabel('Contribution to annualized real GDP growth, pp')
    ax.set_xticks(x, table.index, rotation=45, ha='right'); ax.legend(frameon=False, ncol=2, loc='upper left')
    ax.set_title(model_label + ' | AI-linked and other S&P: estimates and uncertainty', loc='left', fontsize=15, pad=15)
    fig.text(.5, .015, 'Shading: exploratory pointwise 95% intervals from 399 saved joint coefficient draws. Ownership-bridge baseline: '+anchor+'.\n'
        'Observed paths and ownership assumptions held fixed; unresolved ECM diagnostics remain. No new equation is fitted.',
        ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .1, 1, 1)); _save(fig, output, 'ai_other_intervals', plt)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    settings = [('ownership_bridge',first), ('price_control_only',first),
                ('ownership_bridge',alternate), ('price_control_only',alternate)]
    base = str(pd.Period(first, freq='Q') - 1)
    alternate_base = str(pd.Period(alternate, freq='Q') - 1)
    labels = ['Ownership bridge\n'+base+' baseline', 'Price control only\n'+base+' baseline',
              'Ownership bridge\n'+alternate_base+' baseline', 'Price control only\n'+alternate_base+' baseline']
    sensitivity = periods.loc[(periods.model == display_model) & (periods.period == '2026 H1')
        & (periods.channel == 'ai_sp')].set_index(['variant','split_start']).loc[settings]
    y = sensitivity.mean_quarterly_annualized_gdp_pp.to_numpy(); positions = np.arange(len(sensitivity))
    ax.bar(positions, y, color=['#246c9d','#9bbdd3','#dd9839','#efd0a6'], width=.65)
    for j, row in enumerate(sensitivity.itertuples()):
        ax.plot([j,j], [row.ci_low_pp,row.ci_high_pp], color='#222', lw=1.2)
        ax.scatter([j,j], [row.ci_low_pp,row.ci_high_pp], color='#222', marker='_', s=60)
    ax.axhline(0, color='#444', lw=.7); ax.set_xticks(positions, labels, fontsize=9); ax.grid(axis='y', alpha=.18)
    ax.set_ylabel('2026 H1 mean annualized GDP contribution, pp')
    ax.set_title(model_label + ' | AI-linked contribution: mapping sensitivity', loc='left', fontsize=14, pad=15)
    fig.text(.5, .015, 'All bars describe 2026 H1. Baseline changes alter the attribution, while original total wealth contributions stay fixed.\n'
        'Whiskers: exploratory conditional 95% intervals; assumptions are scenarios, not estimated model probabilities.',
        ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .1, 1, 1)); _save(fig, output, 'mapping_sensitivity', plt)


def _results(output, quarterly, periods, first, display_start='2023Q1', display_model='ratio_ecm'):
    primary=periods.loc[(periods.variant=='ownership_bridge')&(periods.split_start==first)&(periods.period=='2026 H1')]
    rows=[]
    for model in ('ratio_ecm','log_ecm'):
        t=primary.loc[primary.model==model].set_index('channel')
        vals=[t.loc[c,'mean_quarterly_annualized_gdp_pp'] for c in (*COMPONENTS,'original_wealth')]
        rows.append('| '+LABELS[model]+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
    text='\n'.join(rows)
    (output/'RESULTS.md').write_text(f'''# AI-linked S&P subdivision of the quarterly ECM

New split: **2022Q1–2026Q2**, with a **2021Q4** opening portfolio. The full ECM recursion still starts from 1989Q4. Existing 1990–2019 coefficients and all 399 saved joint draws per model are reused; no equation is refitted.

2026 H1 **mean quarterly annualized** real GDP contributions, percentage points, using the ownership bridge:

| Model | AI-linked S&P | Other S&P | Index/cap basis | Remaining wealth / inherited | Original total wealth |
| --- | ---: | ---: | ---: | ---: | ---: |
{text}

Charts focus on the **{LABELS[display_model]} from {display_start} onward**. This display cutoff
does not change the primary 2021Q4 split baseline or any estimated contribution.

![Quarterly split](wealth_split.png)

The four parts sum to the original wealth total in every quarter and every joint draw. Income, credit, unemployment, residual and initial-condition channels are preserved. The remaining-wealth category includes housing, other assets/liabilities, inherited effects and the discrepancy between mapped capitalization changes and observed net worth; it is not an estimate of housing alone.

The ownership bridge allocates capitalization changes using lagged aggregate household beneficial-equity ownership and lagged DFA income-group equity shares, with a common AI/other composition within each income group. These are explicit incidence assumptions. A second `price_control_only` version subdivides only the existing S&P price control. A 2023Q1 start/2022Q4 baseline is included as sensitivity, because the 2022 equity decline affects the attribution.

[Quarterly estimates and pointwise intervals](quarterly_wealth_split.csv), [period means and intervals](period_summary.csv), [all consumption-to-GDP channels](quarterly_all_channels.csv), [mapping sensitivity chart](mapping_sensitivity.png), [diagnostics](diagnostics.json), [hash manifest](run_manifest.json), [methodology](../../docs/AI_ECM_SPLIT.md).

These are conditional gross consumption allocations of BEA's published PCE contribution, not an identified causal contribution of AI to GDP. The original ECM diagnostics remain unresolved. Intervals condition on fixed incidence assumptions, data, basket and model form; component interval endpoints need not sum. S&P-specific AI capitalization is narrower than the broader AI basket in the original AI wealth calculation, and this ECM bridge does not apply that study's import/overlap offsets.

The new market-cap input is user-supplied; its public vendor/constituent provenance is not verified. AI/non-AI market-cap observations and the detailed capitalization/ownership ledger are excluded from public Git; [provenance and units](../../data/spx_ai_split_sources.json) are retained. Reproduction requires supplying the input locally using the [documented schema](../../docs/AI_ECM_SPLIT.md). The S&P total is divided by 1,000, as confirmed by the user, to match AI caps in USD billions. Capitalization changes may include issuance, buybacks and membership effects. Original GDP/PCE retain the August 26, 2026 vintage; AI data come from the existing later September snapshot.

```bash
python scripts/run_ai_ecm_split.py
python -m unittest discover -s tests -p 'test_ecm_ai_split.py' -v
```
''')
