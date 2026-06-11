"""Interactive Plotly dashboard matching the static matplotlib visualization."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Palette aligned with inventory_ppo.visualize_results (matplotlib reference)
BG = '#F0F4FA'
PANEL = '#FFFFFF'
GRID = '#DDE3EE'
TEXT = '#1E293B'
MUTED = '#64748B'
C_INV = '#3B82F6'   # inventory line
C_DEM = '#F87171'   # actual / forecast demand bars
C_UNM = '#DC2626'   # unmet demand bars
C_ORD = '#FB923C'   # order qty bars
C_HLD = '#60A5FA'   # holding cost (lighter blue than inventory line)
C_ORC = '#FBBF24'   # ordering cost (yellow)
C_LST = '#F43F5E'   # lost sales + cumulative cost (rose red)
C_HOR = '#D97706'   # forecast horizon divider
FUT_SHADE = '#FEF9C3'


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


def _visible(name, visible_series):
    return name in visible_series


def prepare_chart_data(records, future_records=None):
    future_records = future_records or []
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
        'historical_weeks': n_hist,
        'projected_weeks': len(future_records),
    }

    return {
        'weeks': weeks,
        'demand': demand,
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
        'kpis': kpis,
    }


def build_dashboard_figure(records, product, location, future_records=None,
                           visible_series=None):
    visible_series = visible_series if visible_series is not None else DEFAULT_VISIBLE
    data = prepare_chart_data(records, future_records)
    weeks = data['weeks']
    n_hist = data['n_hist']
    x = data['x']
    x_hist = data['x_hist']
    x_fut = data['x_fut']

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

    week_labels = [str(w) for w in weeks]

    # Panel 1 — inventory fill first, demand bars on top (matches matplotlib z-order)
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

    tick_step = max(1, len(weeks) // 24)
    tick_idx = list(range(0, len(weeks), tick_step))
    tick_vals = [x[i] for i in tick_idx]
    tick_text = [week_labels[i] for i in tick_idx]

    fig.update_layout(
        title=dict(
            text=f'PPO Inventory Policy · {product}<br><sup style="color:{MUTED}">{location}</sup>',
            x=0.5, xanchor='center', font=dict(size=16, color=TEXT),
        ),
        height=900,
        barmode='overlay',
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(family='sans-serif', size=10, color=TEXT),
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

    fig.update_yaxes(title_text='Units', gridcolor=GRID, linecolor=GRID, row=1, col=1)
    fig.update_yaxes(title_text='Units ordered', gridcolor=GRID, linecolor=GRID, row=2, col=1)
    fig.update_yaxes(title_text='Cost (€)', gridcolor=GRID, linecolor=GRID, row=3, col=1)
    fig.update_yaxes(title_text='Cumulative cost (€)', gridcolor=GRID, linecolor=GRID, row=4, col=1)

    fig.update_xaxes(range=[-0.5, len(weeks) - 0.5])

    return fig, data['kpis']
