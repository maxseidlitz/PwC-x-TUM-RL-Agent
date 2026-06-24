"""Interactive Plotly dashboard matching the static matplotlib visualization."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inventory_ppo import BS_COLORS, BS_VARIANT_LABELS, BS_VARIANTS  # noqa: E402

if TYPE_CHECKING:
    from run_loader import LoadedRun

from tum_theme import (
    BG,
    C_DEM,
    C_HLD,
    C_HOR,
    C_INV,
    C_LST,
    C_ORD,
    C_ORC,
    C_UNM,
    FONT_FAMILY_PLOTLY,
    FUT_SHADE,
    GRID,
    MUTED,
    PANEL,
    TEXT,
    TUM_BLUE_DARK,
)


def _rgba(hex_color, alpha):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def _hatch_marker(color, alpha=0.35):
    """Projected bars: same hue as historical, hatched like matplotlib."""
    return dict(
        color=_rgba(color, alpha),
        line=dict(width=0),
        pattern=dict(
            shape='/',
            fgcolor=_rgba(color, min(alpha + 0.35, 1.0)),
            bgcolor=PANEL,
            size=6,
            solidity=0.35,
        ),
    )

ALL_SERIES = [
    'Actual demand',
    'Unmet demand',
    'Forecast demand',
    'Inventory (after arrival)',
    'Order qty',
    'Projected order qty',
    'Holding',
    'Ordering',
    'Lost sales',
    'Cumulative cost',
    'Projected cumulative cost',
    'Forecast horizon',
]

DEFAULT_VISIBLE = list(ALL_SERIES)

RUN_COLORS = ['#0065BD', '#F57C00', '#2E7D32', '#7B1FA2', '#D32F2F']

COMPARE_SERIES = [
    'Actual demand',
    'Unmet demand',
    'Forecast demand',
    'Forecast horizon',
    'Inventory (after arrival)',
    'Order qty',
    'Weekly total cost',
    'Cumulative cost',
    'Projected cumulative cost',
]

DEFAULT_COMPARE_VISIBLE = list(COMPARE_SERIES)

BASELINE_POLICY_OPTIONS = [
    ('conservative', BS_VARIANT_LABELS['conservative']),
    ('middle', BS_VARIANT_LABELS['middle']),
    ('aggressive', BS_VARIANT_LABELS['aggressive']),
]


def _bs_visible(bs_item, visible_baselines):
    if visible_baselines is None:
        return True
    return bs_item.get('variant', 'middle') in visible_baselines


def _bs_chart_arrays(bs_records):
    inv = np.array([r['inventory_after_arrival'] for r in bs_records], dtype=float)
    orders = np.array([r['order_qty'] for r in bs_records], dtype=float)
    hold = np.array([r['holding_cost_total'] for r in bs_records], dtype=float)
    ord_c = np.array([r['ordering_cost_total'] for r in bs_records], dtype=float)
    lost = np.array([r['lost_sales_cost_total'] for r in bs_records], dtype=float)
    cum = np.cumsum([-r['reward'] for r in bs_records])
    return inv, orders, hold, ord_c, lost, cum


def _add_baseline_traces(fig, base_stock_results, plot_x, visible_baselines=None):
    if not base_stock_results or not plot_x:
        return
    n_weeks = len(plot_x)
    for i, bs in enumerate(base_stock_results):
        if not _bs_visible(bs, visible_baselines):
            continue
        recs = bs.get('records', [])
        if len(recs) != n_weeks:
            continue
        variant = bs.get('variant', 'middle')
        S = bs['S']
        label = BS_VARIANT_LABELS.get(variant, variant)
        color = BS_COLORS[i % len(BS_COLORS)]
        inv, orders, hold, ord_c, lost, cum = _bs_chart_arrays(recs)
        name_prefix = f'BS {label} S={S}'
        lg = f'bs_{variant}'

        fig.add_trace(go.Scatter(
            x=plot_x, y=inv, name=f'{name_prefix} · Inventory',
            mode='lines', line=dict(color=color, width=1.5, dash='dash'),
            legendgroup=lg,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=plot_x, y=orders, name=f'{name_prefix} · Orders',
            mode='lines', line=dict(color=color, width=1.5, dash='dash'),
            legendgroup=lg,
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=plot_x, y=hold + ord_c + lost, name=f'{name_prefix} · Weekly cost',
            mode='lines', line=dict(color=color, width=1.5, dash='dash'),
            legendgroup=lg,
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=plot_x, y=cum, name=f'{name_prefix} · Cumulative',
            mode='lines', line=dict(color=color, width=1.5, dash='dash'),
            legendgroup=lg,
        ), row=4, col=1)


def ppo_kpis_for_table(total_cost, service_level, total_ordered, avg_inventory):
    return {
        'Total Cost (€)': round(float(total_cost), 2),
        'Service Level (%)': round(float(service_level), 2),
        'Total Ordered': int(total_ordered),
        'Avg Inventory': round(float(avg_inventory), 2),
    }


def build_policy_comparison_df(ppo_kpis, base_stock_results):
    rows = [{'Policy': 'PPO Agent', **ppo_kpis}]
    for bs in base_stock_results or []:
        variant = bs.get('variant', 'middle')
        label = BS_VARIANT_LABELS.get(variant, variant)
        rows.append({
            'Policy': f'Base Stock {label} (S={bs["S"]})',
            **bs.get('kpis', {}),
        })
    return pd.DataFrame(rows)


def baseline_by_variant(base_stock_results, variant):
    for bs in base_stock_results or []:
        if bs.get('variant') == variant:
            return bs
    return None


def format_metric_delta(ppo_val, baseline_val, lower_is_better=True):
    if baseline_val == 0:
        return None, 'off'
    diff = ppo_val - baseline_val
    pct = diff / baseline_val * 100.0
    if lower_is_better:
        color = 'inverse' if diff < 0 else 'normal'
    else:
        color = 'normal' if diff > 0 else 'inverse'
    return f'{pct:+.1f}%', color


def _visible(name, visible_series):
    return name in visible_series


def prepare_chart_data(records, future_records=None, hist_demand=None, hist_week_labels=None):
    future_records = future_records or []
    hist_demand = list(hist_demand) if hist_demand else []
    hist_week_labels = list(hist_week_labels) if hist_week_labels else []
    has_hist = bool(hist_demand)

    if has_hist:
        # New mode: historical demand bars (left) + agent planning (right)
        all_agent = records + future_records
        n_planning = len(records)

        def col(key):
            return [r[key] for r in all_agent]

        weeks = hist_week_labels + [r['week'] for r in all_agent]
        n_hist = len(hist_demand)
        x = list(range(len(weeks)))
        x_hist = list(range(n_hist))
        x_fut = list(range(n_hist, n_hist + len(all_agent)))

        demand_plan = np.array(col('actual_demand'), dtype=float)
        orders = np.array(col('order_qty'), dtype=float)
        inventory_after_arrival = np.array(col('inventory_after_arrival'), dtype=float)
        unmet = np.array(col('unmet_demand'), dtype=float)
        hold_c = np.array(col('holding_cost_total'), dtype=float)
        ord_c = np.array(col('ordering_cost_total'), dtype=float)
        lost_c = np.array(col('lost_sales_cost_total'), dtype=float)
        rewards = np.array(col('reward'), dtype=float)
        cum_cost = np.cumsum(-rewards)

        kpis = {
            'total_cost': float(-rewards[:n_planning].sum()),
            'service_level': 100.0 * (1 - unmet[:n_planning].sum() / max(demand_plan[:n_planning].sum(), 1)),
            'total_ordered': int(orders[:n_planning].sum()),
            'avg_inventory': float(np.mean([r['inventory'] for r in records])) if n_planning else 0.0,
            'forecast_weeks': n_planning,
        }

        return {
            'weeks': weeks,
            'demand': demand_plan,
            'hist_demand_values': np.array(hist_demand, dtype=float),
            'orders': orders,
            'inventory_after_arrival': inventory_after_arrival,
            'unmet': unmet,
            'hold_c': hold_c,
            'ord_c': ord_c,
            'lost_c': lost_c,
            'cum_cost': cum_cost,
            'x': x,
            'x_hist': x_hist,
            'x_fut': x_fut,
            'n_hist': n_hist,
            'has_hist': True,
            'kpis': kpis,
        }

    else:
        # Legacy mode: records shown as historical, future_records as projected
        n_hist = len(records)
        all_records = records + future_records

        def col(key):
            return [r[key] for r in all_records]

        weeks = col('week')
        demand = np.array(col('actual_demand'), dtype=float)
        orders = np.array(col('order_qty'), dtype=float)
        inventory_after_arrival = np.array(col('inventory_after_arrival'), dtype=float)
        unmet = np.array(col('unmet_demand'), dtype=float)
        hold_c = np.array(col('holding_cost_total'), dtype=float)
        ord_c = np.array(col('ordering_cost_total'), dtype=float)
        lost_c = np.array(col('lost_sales_cost_total'), dtype=float)
        rewards = np.array(col('reward'), dtype=float)
        cum_cost = np.cumsum(-rewards)

        x = list(range(len(weeks)))
        x_hist = list(range(n_hist))
        x_fut = list(range(n_hist, len(all_records)))

        kpis = {
            'total_cost': float(-rewards[:n_hist].sum()) if n_hist else 0.0,
            'service_level': 100.0 * (1 - unmet[:n_hist].sum() / max(demand[:n_hist].sum(), 1)) if n_hist else 0.0,
            'total_ordered': int(orders[:n_hist].sum()) if n_hist else 0,
            'avg_inventory': float(np.mean([r['inventory'] for r in records])) if n_hist else 0.0,
            'forecast_weeks': n_hist,
        }

        return {
            'weeks': weeks,
            'demand': demand,
            'hist_demand_values': None,
            'orders': orders,
            'inventory_after_arrival': inventory_after_arrival,
            'unmet': unmet,
            'hold_c': hold_c,
            'ord_c': ord_c,
            'lost_c': lost_c,
            'cum_cost': cum_cost,
            'x': x,
            'x_hist': x_hist,
            'x_fut': x_fut,
            'n_hist': n_hist,
            'has_hist': False,
            'kpis': kpis,
        }


def build_dashboard_figure(records, product, location, future_records=None,
                           visible_series=None, hist_demand=None, hist_week_labels=None,
                           base_stock_results=None, visible_baselines=None):
    visible_series = visible_series if visible_series is not None else DEFAULT_VISIBLE
    data = prepare_chart_data(records, future_records, hist_demand, hist_week_labels)
    weeks = data['weeks']
    n_hist = data['n_hist']
    x = data['x']
    x_hist = data['x_hist']
    x_fut = data['x_fut']
    has_hist = data['has_hist']

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.32, 0.22, 0.22, 0.24],
        subplot_titles=(
            'Inventory Level vs Demand',
            'Weekly Order Quantities',
            'Weekly Cost Breakdown',
            'Cumulative Cost',
        ),
    )

    def _show(name):
        return 'legendonly' if not _visible(name, visible_series) else True

    # Forecast horizon shading (all rows)
    if x_fut and _visible('Forecast horizon', visible_series):
        x0 = n_hist - 0.5
        x1 = x_fut[-1] + 0.5
        for row in range(1, 5):
            fig.add_vrect(
                x0=x0, x1=x1, fillcolor=FUT_SHADE, opacity=0.07,
                layer='below', line_width=0, row=row, col=1,
            )
            if row == 1:
                fig.add_vline(
                    x=n_hist - 0.5, line_dash='dash', line_color=C_HOR,
                    line_width=1.2, opacity=0.85, row=row, col=1,
                    annotation_text='Forecast horizon',
                    annotation_position='top right',
                )
            else:
                fig.add_vline(
                    x=n_hist - 0.5, line_dash='dash', line_color=C_HOR,
                    line_width=1.2, opacity=0.85, row=row, col=1,
                )

    if has_hist:
        # ── New mode: historical demand bars (left) + agent planning (right) ──

        # Panel 1
        if x_hist:
            fig.add_trace(go.Bar(
                x=x_hist, y=data['hist_demand_values'], name='Actual demand',
                marker=dict(color=_rgba(C_DEM, 0.28), line=dict(width=0)),
                legendgroup='p1', visible=_show('Actual demand'),
            ), row=1, col=1)
        if x_fut:
            fig.add_trace(go.Scatter(
                x=x_fut, y=data['inventory_after_arrival'], name='Inventory (after arrival)',
                mode='lines', line=dict(color=C_INV, width=2.2),
                fill='tozeroy', fillcolor=_rgba(C_INV, 0.15),
                legendgroup='p1', visible=_show('Inventory (after arrival)'),
            ), row=1, col=1)
            fig.add_trace(go.Bar(
                x=x_fut, y=data['demand'], name='Forecast demand',
                marker=_hatch_marker(C_DEM, alpha=0.18),
                legendgroup='p1', visible=_show('Forecast demand'),
            ), row=1, col=1)
            fig.add_trace(go.Bar(
                x=x_fut, y=data['unmet'], name='Unmet demand',
                marker=dict(color=_rgba(C_UNM, 0.85), line=dict(width=0)),
                legendgroup='p1', visible=_show('Unmet demand'),
            ), row=1, col=1)

        # Panel 2 — planned orders in future period only
        if x_fut:
            fig.add_trace(go.Bar(
                x=x_fut, y=data['orders'], name='Order qty',
                marker=dict(color=_rgba(C_ORD, 0.85), line=dict(width=0)),
                legendgroup='p2', visible=_show('Order qty'),
            ), row=2, col=1)

        # Panel 3 — costs in future period only
        if x_fut:
            bh = data['hold_c']
            bo = bh + data['ord_c']
            fig.add_trace(go.Bar(
                x=x_fut, y=bh, name='Holding',
                marker=dict(color=_rgba(C_HLD, 0.90), line=dict(width=0)),
                legendgroup='p3', visible=_show('Holding'),
            ), row=3, col=1)
            fig.add_trace(go.Bar(
                x=x_fut, y=data['ord_c'], name='Ordering',
                marker=dict(color=_rgba(C_ORC, 0.90), line=dict(width=0)),
                base=bh, legendgroup='p3', visible=_show('Ordering'),
            ), row=3, col=1)
            fig.add_trace(go.Bar(
                x=x_fut, y=data['lost_c'], name='Lost sales',
                marker=dict(color=_rgba(C_LST, 0.90), line=dict(width=0)),
                base=bo, legendgroup='p3', visible=_show('Lost sales'),
            ), row=3, col=1)

        # Panel 4 — cumulative cost in future period only
        if x_fut:
            fig.add_trace(go.Scatter(
                x=x_fut, y=data['cum_cost'], name='Cumulative cost',
                mode='lines', line=dict(color=C_LST, width=2.2),
                fill='tozeroy', fillcolor=_rgba(C_LST, 0.12),
                legendgroup='p4', visible=_show('Cumulative cost'),
            ), row=4, col=1)

        plot_x = x_fut
        _add_baseline_traces(fig, base_stock_results, plot_x, visible_baselines)

    else:
        # ── Legacy mode: records as historical, future_records as projected ──

        # Panel 1 — inventory fill first, demand bars on top
        fig.add_trace(go.Scatter(
            x=x, y=data['inventory_after_arrival'], name='Inventory (after arrival)',
            mode='lines', line=dict(color=C_INV, width=2.2),
            fill='tozeroy', fillcolor=_rgba(C_INV, 0.15),
            legendgroup='p1', visible=_show('Inventory (after arrival)'),
        ), row=1, col=1)
        if x_hist:
            fig.add_trace(go.Bar(
                x=x_hist, y=data['demand'][:n_hist], name='Actual demand',
                marker=dict(color=_rgba(C_DEM, 0.28), line=dict(width=0)),
                legendgroup='p1', visible=_show('Actual demand'),
            ), row=1, col=1)
            fig.add_trace(go.Bar(
                x=x_hist, y=data['unmet'][:n_hist], name='Unmet demand',
                marker=dict(color=_rgba(C_UNM, 0.85), line=dict(width=0)),
                legendgroup='p1', visible=_show('Unmet demand'),
            ), row=1, col=1)
        if x_fut:
            fig.add_trace(go.Bar(
                x=x_fut, y=data['demand'][n_hist:], name='Forecast demand',
                marker=_hatch_marker(C_DEM, alpha=0.18),
                legendgroup='p1', visible=_show('Forecast demand'),
            ), row=1, col=1)

        # Panel 2
        if x_hist:
            fig.add_trace(go.Bar(
                x=x_hist, y=data['orders'][:n_hist], name='Order qty',
                marker=dict(color=_rgba(C_ORD, 0.85), line=dict(width=0)),
                legendgroup='p2', visible=_show('Order qty'),
            ), row=2, col=1)
        if x_fut:
            fig.add_trace(go.Bar(
                x=x_fut, y=data['orders'][n_hist:], name='Projected order qty',
                marker=_hatch_marker(C_ORD, alpha=0.35),
                legendgroup='p2', visible=_show('Projected order qty'),
            ), row=2, col=1)

        # Panel 3 — manually stacked costs (overlay mode)
        if x_hist:
            bh = data['hold_c'][:n_hist]
            bo = bh + data['ord_c'][:n_hist]
            fig.add_trace(go.Bar(
                x=x_hist, y=bh, name='Holding',
                marker=dict(color=_rgba(C_HLD, 0.90), line=dict(width=0)),
                legendgroup='p3', visible=_show('Holding'),
            ), row=3, col=1)
            fig.add_trace(go.Bar(
                x=x_hist, y=data['ord_c'][:n_hist], name='Ordering',
                marker=dict(color=_rgba(C_ORC, 0.90), line=dict(width=0)),
                base=bh, legendgroup='p3', visible=_show('Ordering'),
            ), row=3, col=1)
            fig.add_trace(go.Bar(
                x=x_hist, y=data['lost_c'][:n_hist], name='Lost sales',
                marker=dict(color=_rgba(C_LST, 0.90), line=dict(width=0)),
                base=bo, legendgroup='p3', visible=_show('Lost sales'),
            ), row=3, col=1)
        if x_fut:
            bf = data['hold_c'][n_hist:]
            bof = bf + data['ord_c'][n_hist:]
            fig.add_trace(go.Bar(
                x=x_fut, y=bf, name='Holding (proj.)',
                marker=_hatch_marker(C_HLD, alpha=0.35),
                showlegend=False, legendgroup='p3', visible=_show('Holding'),
            ), row=3, col=1)
            fig.add_trace(go.Bar(
                x=x_fut, y=data['ord_c'][n_hist:], name='Ordering (proj.)',
                marker=_hatch_marker(C_ORC, alpha=0.35), base=bf,
                showlegend=False, legendgroup='p3', visible=_show('Ordering'),
            ), row=3, col=1)
            fig.add_trace(go.Bar(
                x=x_fut, y=data['lost_c'][n_hist:], name='Lost sales (proj.)',
                marker=_hatch_marker(C_LST, alpha=0.35), base=bof,
                showlegend=False, legendgroup='p3', visible=_show('Lost sales'),
            ), row=3, col=1)

        # Panel 4
        if n_hist:
            fig.add_trace(go.Scatter(
                x=x[:n_hist], y=data['cum_cost'][:n_hist], name='Cumulative cost',
                mode='lines', line=dict(color=C_LST, width=2.2),
                fill='tozeroy', fillcolor=_rgba(C_LST, 0.12),
                legendgroup='p4', visible=_show('Cumulative cost'),
            ), row=4, col=1)
        if x_fut:
            jx = x[n_hist - 1:]
            jy = data['cum_cost'][n_hist - 1:]
            fig.add_trace(go.Scatter(
                x=jx, y=jy, name='Projected cumulative cost',
                mode='lines',
                line=dict(color=_rgba(C_LST, 0.55), width=2.2, dash='dash'),
                legendgroup='p4', visible=_show('Projected cumulative cost'),
            ), row=4, col=1)

        plot_x = x_hist if x_hist else x
        _add_baseline_traces(fig, base_stock_results, plot_x, visible_baselines)

    title_suffix = ' · vs Base Stock' if base_stock_results else ''
    tick_step = max(1, len(weeks) // 24)
    tick_idx = list(range(0, len(weeks), tick_step))
    week_labels = [str(w) for w in weeks]
    tick_vals = [x[i] for i in tick_idx]
    tick_text = [week_labels[i] for i in tick_idx]

    fig.update_layout(
        title=dict(
            text=f'PPO Inventory Policy{title_suffix} · {product}<br><sup style="color:{MUTED}">{location}</sup>',
            x=0.5, xanchor='center',
            font=dict(size=16, color=TUM_BLUE_DARK, family=FONT_FAMILY_PLOTLY),
        ),
        height=900,
        barmode='overlay',
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(family=FONT_FAMILY_PLOTLY, size=10, color=TEXT),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, x=0,
            bgcolor='rgba(255,255,255,0.92)', bordercolor=GRID, borderwidth=1,
        ),
        margin=dict(l=60, r=30, t=100, b=60),
        hovermode='x unified',
    )

    fig.update_xaxes(
        tickvals=tick_vals, ticktext=tick_text, tickangle=40,
        gridcolor=GRID, linecolor=GRID, row=4, col=1,
    )
    for row in range(1, 4):
        fig.update_xaxes(showticklabels=False, gridcolor=GRID, linecolor=GRID, row=row, col=1)
    fig.update_xaxes(title_text='Week', title_font_color=MUTED, row=4, col=1)

    for ann in fig.layout.annotations:
        ann.font = dict(family=FONT_FAMILY_PLOTLY, size=11, color=TUM_BLUE_DARK)

    fig.update_yaxes(title_text='Units', gridcolor=GRID, linecolor=GRID, row=1, col=1)
    fig.update_yaxes(title_text='Units ordered', gridcolor=GRID, linecolor=GRID, row=2, col=1)
    fig.update_yaxes(title_text='Cost (€)', gridcolor=GRID, linecolor=GRID, row=3, col=1)
    fig.update_yaxes(title_text='Cumulative cost (€)', gridcolor=GRID, linecolor=GRID, row=4, col=1)

    fig.update_xaxes(range=[-0.5, len(weeks) - 0.5])

    return fig, data['kpis']


def _run_column_name(run: LoadedRun, index: int) -> str:
    steps = run.summary.timesteps
    if steps >= 1000 and steps % 1000 == 0:
        step_label = f'{steps // 1000}k'
    else:
        step_label = f'{steps:,}'
    return f'Run {index + 1} ({step_label} steps)'


def _trim_run_data(run: LoadedRun, min_hist: int, min_fut: int) -> tuple[list, list]:
    records = run.records[:min_hist]
    future_records = run.future_records[:min_fut]
    return records, future_records


def compute_comparison_kpis(runs: list[LoadedRun]) -> pd.DataFrame:
    rows = [
        ('Total Cost (€)', lambda r: f"€{r.config.get('total_cost', 0):,.0f}"),
        ('Service Level (%)', lambda r: f"{r.config.get('service_level', 0):.1f}%"),
        ('Timesteps', lambda r: f"{r.config.get('timesteps', 0):,}"),
        ('Duration (s)', lambda r: f"{r.config.get('duration_seconds', 0):.1f}"),
        ('Started at', lambda r: r.config.get('started_at', '—')),
        ('Product', lambda r: r.config.get('product', '—')),
        ('Location', lambda r: r.config.get('location', '—')),
    ]
    data: dict[str, list] = {'Metric': [name for name, _ in rows]}
    for i, run in enumerate(runs):
        col = _run_column_name(run, i)
        data[col] = [fmt(run) for _, fmt in rows]
    return pd.DataFrame(data)


def build_comparison_figure(
    runs: list[LoadedRun],
    visible_series: list[str] | None = None,
    base_stock_results: list | None = None,
    visible_baselines: set[str] | None = None,
) -> go.Figure:
    visible_series = visible_series if visible_series is not None else DEFAULT_COMPARE_VISIBLE

    def _show(name):
        return 'legendonly' if not _visible(name, visible_series) else True

    min_hist = min(len(r.records) for r in runs)
    min_fut = min(len(r.future_records) for r in runs)

    ref_run = runs[0]
    ref_records, ref_future = _trim_run_data(ref_run, min_hist, min_fut)
    ref_hist_demand = getattr(ref_run, 'hist_demand', []) or []
    ref_hist_weeks = getattr(ref_run, 'hist_week_labels', []) or []
    ref_data = prepare_chart_data(ref_records, ref_future, ref_hist_demand, ref_hist_weeks)
    weeks = ref_data['weeks']
    n_hist = ref_data['n_hist']
    x = ref_data['x']
    x_hist = ref_data['x_hist']
    x_fut = ref_data['x_fut']
    has_hist = ref_data['has_hist']

    product = runs[0].config.get('product', '')
    location = runs[0].config.get('location', '')
    title_suffix = ''
    if len({r.config.get('product') for r in runs}) > 1 or len({r.config.get('location') for r in runs}) > 1:
        title_suffix = '<br><sup>Mixed product/location selection</sup>'

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.32, 0.22, 0.22, 0.24],
        subplot_titles=(
            'Inventory Level vs Demand',
            'Weekly Order Quantities',
            'Weekly Cost Breakdown',
            'Cumulative Cost',
        ),
    )

    if x_fut and _visible('Forecast horizon', visible_series):
        x0 = n_hist - 0.5
        x1 = x_fut[-1] + 0.5
        for row in range(1, 5):
            fig.add_vrect(
                x0=x0, x1=x1, fillcolor=FUT_SHADE, opacity=0.07,
                layer='below', line_width=0, row=row, col=1,
            )
            if row == 1:
                fig.add_vline(
                    x=n_hist - 0.5, line_dash='dash', line_color=C_HOR,
                    line_width=1.2, opacity=0.85, row=row, col=1,
                    annotation_text='Forecast horizon',
                    annotation_position='top right',
                )
            else:
                fig.add_vline(
                    x=n_hist - 0.5, line_dash='dash', line_color=C_HOR,
                    line_width=1.2, opacity=0.85, row=row, col=1,
                )

    # Shared demand (reference run)
    if has_hist:
        if x_hist:
            fig.add_trace(go.Bar(
                x=x_hist, y=ref_data['hist_demand_values'], name='Actual demand',
                marker=dict(color=_rgba(C_DEM, 0.28), line=dict(width=0)),
                legendgroup='shared', visible=_show('Actual demand'),
            ), row=1, col=1)
        if x_fut:
            fig.add_trace(go.Bar(
                x=x_fut, y=ref_data['demand'], name='Forecast demand',
                marker=_hatch_marker(C_DEM, alpha=0.18),
                legendgroup='shared', visible=_show('Forecast demand'),
            ), row=1, col=1)
    else:
        if x_hist:
            fig.add_trace(go.Bar(
                x=x_hist, y=ref_data['demand'][:n_hist], name='Actual demand',
                marker=dict(color=_rgba(C_DEM, 0.28), line=dict(width=0)),
                legendgroup='shared', visible=_show('Actual demand'),
            ), row=1, col=1)
            fig.add_trace(go.Bar(
                x=x_hist, y=ref_data['unmet'][:n_hist], name='Unmet demand',
                marker=dict(color=_rgba(C_UNM, 0.85), line=dict(width=0)),
                legendgroup='shared', visible=_show('Unmet demand'),
            ), row=1, col=1)
        if x_fut:
            fig.add_trace(go.Bar(
                x=x_fut, y=ref_data['demand'][n_hist:], name='Forecast demand',
                marker=_hatch_marker(C_DEM, alpha=0.18),
                legendgroup='shared', visible=_show('Forecast demand'),
            ), row=1, col=1)

    for i, run in enumerate(runs):
        color = RUN_COLORS[i % len(RUN_COLORS)]
        records, future_records = _trim_run_data(run, min_hist, min_fut)
        run_hist_demand = getattr(run, 'hist_demand', []) or []
        run_hist_weeks = getattr(run, 'hist_week_labels', []) or []
        data = prepare_chart_data(records, future_records, run_hist_demand, run_hist_weeks)
        run_label = _run_column_name(run, i)
        lg = f'run{i}'

        # Inventory and orders placed in planning period (x_fut) for new mode, or all x for legacy
        inv_x = x_fut if has_hist else x
        ord_x = x_fut if has_hist else x
        cost_x = x_fut if has_hist else x

        fig.add_trace(go.Scatter(
            x=inv_x, y=data['inventory_after_arrival'],
            name=f'{run_label} · Inventory',
            mode='lines', line=dict(color=color, width=2.2),
            legendgroup=lg, visible=_show('Inventory (after arrival)'),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=ord_x, y=data['orders'],
            name=f'{run_label} · Orders',
            mode='lines', line=dict(color=color, width=2.0, dash='dot'),
            legendgroup=lg, visible=_show('Order qty'),
        ), row=2, col=1)

        weekly_total = data['hold_c'] + data['ord_c'] + data['lost_c']
        fig.add_trace(go.Scatter(
            x=cost_x, y=weekly_total,
            name=f'{run_label} · Weekly total cost',
            mode='lines', line=dict(color=color, width=2.0),
            legendgroup=lg, visible=_show('Weekly total cost'),
        ), row=3, col=1)

        if has_hist:
            if x_fut:
                fig.add_trace(go.Scatter(
                    x=x_fut, y=data['cum_cost'],
                    name=f'{run_label} · Cumulative cost',
                    mode='lines', line=dict(color=color, width=2.2),
                    legendgroup=lg, visible=_show('Cumulative cost'),
                ), row=4, col=1)
        else:
            if n_hist:
                fig.add_trace(go.Scatter(
                    x=x[:n_hist], y=data['cum_cost'][:n_hist],
                    name=f'{run_label} · Cumulative cost',
                    mode='lines', line=dict(color=color, width=2.2),
                    legendgroup=lg, visible=_show('Cumulative cost'),
                ), row=4, col=1)
            if x_fut:
                jx = x[n_hist - 1:]
                jy = data['cum_cost'][n_hist - 1:]
                fig.add_trace(go.Scatter(
                    x=jx, y=jy,
                    name=f'{run_label} · Projected cumulative',
                    mode='lines',
                    line=dict(color=color, width=2.0, dash='dash'),
                    legendgroup=lg, visible=_show('Projected cumulative cost'),
                ), row=4, col=1)

    if base_stock_results is None and runs:
        base_stock_results = getattr(runs[0], 'base_stock_results', None) or []

    plot_x = x_fut if has_hist else (x_hist if x_hist else x)
    _add_baseline_traces(fig, base_stock_results, plot_x, visible_baselines)

    week_labels = [str(w) for w in weeks]
    tick_step = max(1, len(weeks) // 24)
    tick_idx = list(range(0, len(weeks), tick_step))
    tick_vals = [x[i] for i in tick_idx]
    tick_text = [week_labels[i] for i in tick_idx]

    fig.update_layout(
        title=dict(
            text=f'Run Comparison · {product}<br><sup style="color:{MUTED}">{location}</sup>{title_suffix}',
            x=0.5, xanchor='center',
            font=dict(size=16, color=TUM_BLUE_DARK, family=FONT_FAMILY_PLOTLY),
        ),
        height=900,
        barmode='overlay',
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(family=FONT_FAMILY_PLOTLY, size=10, color=TEXT),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, x=0,
            bgcolor='rgba(255,255,255,0.92)', bordercolor=GRID, borderwidth=1,
        ),
        margin=dict(l=60, r=30, t=110, b=60),
        hovermode='x unified',
    )

    fig.update_xaxes(
        tickvals=tick_vals, ticktext=tick_text, tickangle=40,
        gridcolor=GRID, linecolor=GRID, row=4, col=1,
    )
    for row in range(1, 4):
        fig.update_xaxes(showticklabels=False, gridcolor=GRID, linecolor=GRID, row=row, col=1)
    fig.update_xaxes(title_text='Week', title_font_color=MUTED, row=4, col=1)

    for ann in fig.layout.annotations:
        ann.font = dict(family=FONT_FAMILY_PLOTLY, size=11, color=TUM_BLUE_DARK)

    fig.update_yaxes(title_text='Units', gridcolor=GRID, linecolor=GRID, row=1, col=1)
    fig.update_yaxes(title_text='Units ordered', gridcolor=GRID, linecolor=GRID, row=2, col=1)
    fig.update_yaxes(title_text='Cost (€)', gridcolor=GRID, linecolor=GRID, row=3, col=1)
    fig.update_yaxes(title_text='Cumulative cost (€)', gridcolor=GRID, linecolor=GRID, row=4, col=1)
    fig.update_xaxes(range=[-0.5, len(weeks) - 0.5])

    return fig
