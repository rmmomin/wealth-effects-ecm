"""Column charts built from public ECM outputs, without source capitalizations."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'ratio_ecm': 'Ratio/DOLS ECM', 'log_ecm': 'Log ECM'}


def prepare_column_data(quarterly, accounting, model='ratio_ecm', split_start='2022Q1',
                        display_start='2023Q1', end='2026Q2'):
    """Reconcile three wealth categories and an arithmetic GDP subtraction.

    GDP and the adjusted GDP series are annualized quarterly percentage growth
    rates. Wealth contributions are percentage points. ``other`` combines the
    index/capitalization basis and unallocated wealth; it is not housing alone.
    No coefficients are estimated and no private capitalization data are read.
    """
    required = {'model', 'variant', 'split_start', 'quarter', 'channel',
                'gross_gdp_growth_pp', 'observed_gdp_growth_saar_pct'}
    if not required.issubset(quarterly):
        raise ValueError('Missing saved quarterly ECM output columns')
    if model not in LABELS:
        raise ValueError('Unknown ECM display model')
    if not isinstance(accounting.index, pd.PeriodIndex):
        accounting = accounting.set_index('quarter').copy()
        accounting.index = pd.PeriodIndex(accounting.index, freq='Q')
    if accounting.index.duplicated().any():
        raise ValueError('Duplicate GDP accounting quarter')
    dates = pd.period_range(display_start, end, freq='Q')
    if dates.empty or dates[0] < pd.Period(split_start, freq='Q'):
        raise ValueError('Display dates must lie within the calculated split horizon')
    selected = quarterly.loc[(quarterly.model == model)
        & (quarterly.variant == 'ownership_bridge') & (quarterly.split_start == split_start)].copy()
    selected['quarter'] = pd.PeriodIndex(selected.quarter, freq='Q')
    selected = selected.loc[selected.quarter.isin(dates)]
    expected = {'ai_sp', 'non_ai_sp', 'index_cap_basis', 'unallocated_wealth', 'original_wealth'}
    if selected.duplicated(['quarter', 'channel']).any():
        raise ValueError('Duplicate saved quarter/channel')
    if set(selected.channel) != expected or len(selected) != len(dates) * len(expected):
        raise ValueError('Incomplete wealth subdivision; no interpolation or substitution')
    amounts = selected.pivot(index='quarter', columns='channel', values='gross_gdp_growth_pp').reindex(dates)
    if not np.isfinite(amounts.to_numpy()).all():
        raise ValueError('Nonfinite wealth contribution')
    np.testing.assert_allclose(amounts[list(expected - {'original_wealth'})].sum(axis=1),
                               amounts.original_wealth, atol=1e-11, rtol=1e-10)
    padded = pd.period_range(dates[0] - 1, dates[-1], freq='Q')
    if 'real_gdp_saar_bn' not in accounting or not padded.isin(accounting.index).all():
        raise ValueError('Missing GDP level or predecessor; no filling')
    real = accounting.real_gdp_saar_bn.reindex(padded).astype(float)
    if not np.isfinite(real.to_numpy()).all() or (real <= 0).any():
        raise ValueError('GDP levels must be positive and finite')
    observed = (100 * np.expm1(4 * np.log(real / real.shift(1)))).reindex(dates)
    saved = selected.groupby('quarter').observed_gdp_growth_saar_pct
    if (saved.max() - saved.min()).abs().max() > 1e-10:
        raise ValueError('Inconsistent saved observed GDP growth')
    np.testing.assert_allclose(observed, saved.first().reindex(dates), atol=1e-10, rtol=1e-10)
    data = pd.DataFrame(index=dates)
    data.index.name = 'quarter'
    data['observed_real_gdp_growth_saar_pct'] = observed
    data['wealth_gdp_contribution_pp'] = amounts.original_wealth
    data['gdp_growth_less_wealth_saar_pct'] = observed - amounts.original_wealth
    data['ai_sp_contribution_pp'] = amounts.ai_sp
    data['non_ai_sp_contribution_pp'] = amounts.non_ai_sp
    data['other_wealth_contribution_pp'] = amounts.index_cap_basis + amounts.unallocated_wealth
    np.testing.assert_allclose(data[['ai_sp_contribution_pp', 'non_ai_sp_contribution_pp',
        'other_wealth_contribution_pp']].sum(axis=1), data.wealth_gdp_contribution_pp, atol=1e-11, rtol=1e-10)
    data['model'] = model
    data['variant'] = 'ownership_bridge'
    data['split_start'] = split_start
    return data.reset_index().assign(quarter=lambda f: f.quarter.astype(str))


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_column_charts(output_dir=ROOT / 'results/ai_split/columns',
                        config_path=ROOT / 'config/ai_split.json'):
    """Regenerate both column charts using only public saved model outputs."""
    config_path, output = Path(config_path).resolve(), Path(output_dir).resolve()
    config = json.loads(config_path.read_text())
    source = ROOT / 'results/ai_split/quarterly_wealth_split.csv'
    gdp_source = ROOT / config['data']['gdp']
    if output == ROOT or any(output == ROOT / p or output.is_relative_to(ROOT / p)
        for p in ('data', 'config', 'wealth_effects', 'scripts', 'tests', 'docs')):
        raise ValueError('Choose a separate chart output directory')
    source_hashes = {str(p.relative_to(ROOT)): _hash(p) for p in (source, gdp_source)}
    data = prepare_column_data(pd.read_csv(source), pd.read_csv(gdp_source),
        model=config.get('display_model', 'ratio_ecm'), split_start=config['split_start'],
        display_start=config.get('display_start', '2023Q1'), end=config['data_end'])
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / 'column_chart_data.csv', index=False, float_format='%.15g')
    _render(data, output)
    (output / 'README.md').write_text(f'''# Quarterly column charts

Both figures use the {LABELS[data.model.iloc[0]]} ownership bridge and display {data.quarter.iloc[0]}–{data.quarter.iloc[-1]}.
The calculation keeps its {pd.Period(config['split_start'], freq='Q')-1} split baseline and original 1989Q4 ECM anchor.
No model is refitted. The national-accounts data retain the original August26,2026 vintage.

1. **Real GDP growth, observed and less the wealth contribution.** The first series is observed quarter-over-quarter annualized growth; the second subtracts the ECM wealth contribution in percentage points. This is an arithmetic accounting adjustment, not an identified causal or general-equilibrium counterfactual.
2. **Wealth contribution: AI, non-AI and other.** AI-linked S&P, the remaining S&P and other wealth sum to the original total wealth contribution. Other combines the index/capitalization basis with unallocated wealth, including non-S&P assets/liabilities, inherited effects and mapping residuals. It is not a housing estimate. Signed stacks retain negative components; diamonds show the net total.

The CSV contains derived model results and public GDP growth only. The excluded AI/non-AI capitalization observations and detailed gain ledger are not read or exported.

```bash
python scripts/plot_ecm_columns.py
```

[GDP comparison](gdp_with_without_wealth.png), [three-part wealth decomposition](wealth_ai_non_ai_other.png), [chart values](column_chart_data.csv). PDF versions are provided for presentations.
''')
    for name, digest in source_hashes.items():
        if _hash(ROOT / name) != digest:
            raise RuntimeError('Chart generation changed an input')
    cfg_key = str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else 'external_configuration'
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'model': config.get('display_model', 'ratio_ecm'), 'display_start': data.quarter.iloc[0],
        'display_end': data.quarter.iloc[-1], 'allocation_baseline': str(pd.Period(config['split_start'], freq='Q')-1),
        'input_sha256': source_hashes, 'configuration_sha256': {cfg_key: _hash(config_path)},
        'source_sha256': {str(Path(__file__).resolve().relative_to(ROOT)): _hash(__file__),
            'scripts/plot_ecm_columns.py': _hash(ROOT / 'scripts/plot_ecm_columns.py')},
        'private_capitalization_inputs_read': False, 'new_estimation_calls': 0, 'api_calls': 0,
        'output_sha256': {p.name: _hash(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'run_manifest.json'}}
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return {'output': str(output), 'quarters': len(data), 'private_inputs_read': False,
            'wealth_categories_reconcile': True, 'latest': data.iloc[-1].to_dict()}


def _render(data, output):
    import os
    import tempfile
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'wealth-ecm-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
        'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42, 'ps.fonttype':42})
    x = np.arange(len(data))
    model = LABELS[data.model.iloc[0]]
    baseline = str(pd.Period(data.split_start.iloc[0], freq='Q') - 1)
    def style(ax, ylabel):
        ax.axhline(0, color='#354150', lw=.8)
        ax.grid(axis='y', color='#d8dde3', lw=.7, alpha=.6)
        ax.set_axisbelow(True)
        ax.set_xticks(x, data.quarter, rotation=45, ha='right')
        ax.set_ylabel(ylabel)
        ax.margins(x=.025)
    def save(fig, name):
        fig.savefig(output / (name + '.png'), dpi=220, bbox_inches='tight')
        fig.savefig(output / (name + '.pdf'), bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    observed = ax.bar(x-.19, data.observed_real_gdp_growth_saar_pct, width=.36,
        color='#234f75', label='Observed real GDP growth')
    adjusted = ax.bar(x+.19, data.gdp_growth_less_wealth_saar_pct, width=.36,
        color='#b3c4d3', label='Observed growth less wealth contribution')
    ax.bar_label(observed, fmt='%.2f', padding=3, fontsize=7)
    ax.bar_label(adjusted, fmt='%.2f', padding=3, fontsize=7)
    style(ax, 'Real GDP growth, q/q annualized (%)')
    lo = min(0, data[['observed_real_gdp_growth_saar_pct', 'gdp_growth_less_wealth_saar_pct']].min().min())
    hi = max(0, data[['observed_real_gdp_growth_saar_pct', 'gdp_growth_less_wealth_saar_pct']].max().max())
    spread = max(hi-lo, .1)
    ax.set_ylim(lo-.1*spread, hi+.24*spread)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc='upper left')
    ax.set_title('Real GDP growth: with and without the estimated wealth contribution\n'+model+' | '+data.quarter.iloc[0]+'–'+data.quarter.iloc[-1],
        loc='left', fontsize=14, pad=15)
    fig.text(.5, .014, '“Without” subtracts the estimated wealth allocation from observed growth; it is not a causal counterfactual.\n'
        'Quarterly annualized rates. Original ECM coefficients and GDP/PCE vintage are unchanged.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0,.1,1,1)); save(fig, 'gdp_with_without_wealth')

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    positive, negative = np.zeros(len(data)), np.zeros(len(data))
    for name, label, color in (
        ('ai_sp_contribution_pp', 'AI-linked S&P', '#246c9d'),
        ('non_ai_sp_contribution_pp', 'Non-AI-linked S&P', '#dd9839'),
        ('other_wealth_contribution_pp', 'Other wealth / residual', '#adb4bd')):
        values = data[name].to_numpy()
        ax.bar(x, np.maximum(values,0), bottom=positive, width=.68, color=color, label=label)
        ax.bar(x, np.minimum(values,0), bottom=negative, width=.68, color=color)
        positive += np.maximum(values,0); negative += np.minimum(values,0)
    ax.scatter(x, data.wealth_gdp_contribution_pp, marker='D', s=19, color='#202b36', label='Net total wealth', zorder=4)
    style(ax, 'Contribution to annualized real GDP growth (pp)')
    spread = max(positive.max()-negative.min(), .1)
    ax.set_ylim(min(0,negative.min())-.08*spread, positive.max()+.24*spread)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc='upper left')
    ax.set_title('Wealth contribution to real GDP growth: AI, non-AI and other\n'+model+' | '+data.quarter.iloc[0]+'–'+data.quarter.iloc[-1],
        loc='left', fontsize=14, pad=15)
    fig.text(.5, .014, 'Other includes the index/capitalization basis, non-S&P wealth, inherited effects and mapping residuals.\n'
        'Ownership-bridge baseline: '+baseline+'. Conditional gross consumption-channel allocations; not causal GDP effects.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0,.1,1,1)); save(fig, 'wealth_ai_non_ai_other')
