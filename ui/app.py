"""Streamlit UI for PPO inventory optimization training and interactive dashboard."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
UI_DIR = Path(__file__).resolve().parent
for path in (ROOT, UI_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inventory_ppo import (  # noqa: E402
    DEFAULT_FILE_PATH,
    TrainingConfig,
    list_locations_for_product,
    list_products,
    load_data,
    run_training_pipeline,
    suggest_max_order_qty,
)
from dashboard import (  # noqa: E402
    ALL_SERIES,
    COMPARE_SERIES,
    DEFAULT_COMPARE_VISIBLE,
    DEFAULT_VISIBLE,
    build_comparison_figure,
    build_dashboard_figure,
    compute_comparison_kpis,
)
from run_loader import (  # noqa: E402
    ALL_FILTER,
    distinct_locations,
    distinct_products,
    filter_runs,
    list_runs,
    load_run,
)
from tum_theme import inject_tum_styles, render_tum_header  # noqa: E402

st.set_page_config(
    page_title='PPO Inventory Optimizer | TUM',
    page_icon='🔷',
    layout='wide',
    initial_sidebar_state='expanded',
)

DATA_FILE = str(ROOT / DEFAULT_FILE_PATH)
LARGE_TIMESTEPS_THRESHOLD = 500_000
MAX_COMPARE_RUNS = 5


@st.cache_data(ttl=60)
def cached_list_runs():
    return list_runs()


@st.cache_data(ttl=60)
def cached_load_run(run_id: str):
    runs = cached_list_runs()
    match = next((r for r in runs if r.run_id == run_id), None)
    if match is None:
        raise FileNotFoundError(f'Run not found: {run_id}')
    return load_run(match.path)


@st.cache_data
def cached_products(file_path):
    return list_products(file_path)


@st.cache_data
def cached_locations(file_path, product):
    return list_locations_for_product(file_path, product)


def format_eta(seconds):
    if seconds is None or seconds < 0:
        return 'Estimating…'
    if seconds < 60:
        return f'~{int(seconds)} sec remaining'
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f'~{minutes} min {secs} sec remaining'
    hours = minutes // 60
    minutes = minutes % 60
    return f'~{hours} h {minutes} min remaining'


def render_sidebar():
    st.sidebar.header('Configuration')

    st.sidebar.subheader('Data Selection')
    try:
        products = cached_products(DATA_FILE)
    except Exception as e:
        st.sidebar.error(f'Could not load data file: {e}')
        st.stop()

    default_product = 'Ice Cream Strawberry Flavor'
    product_index = products.index(default_product) if default_product in products else 0
    product = st.sidebar.selectbox('Product', products, index=product_index)

    try:
        locations = cached_locations(DATA_FILE, product)
    except Exception as e:
        st.sidebar.error(str(e))
        st.stop()

    default_location = 'Logistics Hub Lissabon'
    location_index = locations.index(default_location) if default_location in locations else 0
    location = st.sidebar.selectbox('Location', locations, index=location_index)

    st.sidebar.caption(f'Data file: `{DEFAULT_FILE_PATH}`')

    st.sidebar.subheader('Training')
    timesteps = st.sidebar.number_input('Timesteps', min_value=1000, value=10000, step=1000)
    learning_rate = st.sidebar.number_input('Learning rate', min_value=1e-5, max_value=1e-1, value=1e-3, format='%.5f')

    st.sidebar.subheader('Cost Model')
    holding_cost = st.sidebar.number_input('Holding cost (€/unit)', min_value=0.0, value=13.0)
    ordering_cost = st.sidebar.number_input('Ordering cost (€/unit)', min_value=0.0, value=60.0)
    lost_sales_cost = st.sidebar.number_input('Lost sales cost (€/unit)', min_value=0.0, value=2500.0)

    st.sidebar.subheader('Environment')
    try:
        demand_data, *_ = load_data(DATA_FILE, product, location)
        _suggested_qty = suggest_max_order_qty(demand_data)
    except Exception:
        _suggested_qty = 200
    max_order_qty = st.sidebar.number_input(
        'Max order qty (units)', min_value=1, max_value=5000, value=_suggested_qty,
        help=f'Suggested: {_suggested_qty} (3× peak weekly demand). '
             'If set much higher than needed the agent must find a needle-in-a-haystack '
             'action and will likely produce excess lost sales.',
    )
    n_forecast_weeks = st.sidebar.number_input('Forecast horizon (weeks)', min_value=1, max_value=12, value=4)

    with st.sidebar.expander('Advanced PPO settings'):
        gamma = st.number_input('Gamma', min_value=0.0, max_value=1.0, value=0.99, format='%.4f')
        n_steps = st.number_input('n_steps', min_value=64, max_value=8192, value=2048, step=64)
        batch_size = st.number_input('Batch size', min_value=32, max_value=4096, value=64, step=32)

    return TrainingConfig(
        file_path=DATA_FILE,
        product=product,
        location=location,
        timesteps=int(timesteps),
        learning_rate=float(learning_rate),
        holding_cost=float(holding_cost),
        ordering_cost=float(ordering_cost),
        lost_sales_cost=float(lost_sales_cost),
        max_order_qty=int(max_order_qty),
        n_forecast_weeks=int(n_forecast_weeks),
        gamma=float(gamma),
        n_steps=int(n_steps),
        batch_size=int(batch_size),
        verbose=0,
    )


def execute_training(config, progress_bar, status_text):
    st.session_state.training = True
    progress_state = {'start': time.time()}

    def on_progress(current, total):
        pct = min(current / total, 1.0)
        progress_bar.progress(pct)
        elapsed = time.time() - progress_state['start']
        if current > max(total * 0.02, 50):
            eta = elapsed / current * (total - current)
        else:
            eta = None
        status_text.markdown(
            f'**Training:** step {current:,} / {total:,} · {format_eta(eta)}'
        )

    try:
        with st.spinner('Training and evaluating model…'):
            result = run_training_pipeline(
                config,
                progress_callback=on_progress,
                verbose=False,
            )
        st.session_state.last_result = result
        progress_bar.progress(1.0)
        status_text.success(
            f'Training complete in {result.duration_seconds:.1f}s · '
            f'artifacts saved to `{result.run_dir}`'
        )
        cached_list_runs.clear()
        cached_load_run.clear()
    except Exception as e:
        status_text.error(f'Training failed: {e}')
    finally:
        st.session_state.training = False


def render_current_run_tab(result):
    if result is None:
        st.info('No training run yet. Adjust parameters in the sidebar and click **Start Training**.')
        return

    st.subheader('Key Performance Indicators')

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric('Total Cost (hist.)', f'€{result.total_cost:,.0f}')
    k2.metric('Service Level', f'{result.service_level:.1f}%')
    k3.metric('Total Ordered', f'{result.total_ordered:,} units')
    k4.metric('Avg Inventory', f'{result.avg_inventory:,.0f} units')
    k5.metric('Historical Weeks', len(result.records))
    k6.metric('Projected Weeks', len(result.future_records))

    st.caption(f'Run directory: `{result.run_dir}`')

    st.subheader('Interactive Dashboard')
    visible = st.multiselect(
        'Visible series',
        options=ALL_SERIES,
        default=DEFAULT_VISIBLE,
        help='Toggle which data series appear in the charts. You can also click legend items in the chart.',
        key='current_visible_series',
    )

    fig, _ = build_dashboard_figure(
        result.records,
        result.product,
        result.location,
        future_records=result.future_records,
        visible_series=visible,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_compare_tab():
    all_runs = cached_list_runs()
    if not all_runs:
        st.info('No saved runs found in `runs/`. Train a model first, then compare runs here.')
        return

    st.subheader('Filter saved runs')

    f1, f2, f3, f4 = st.columns(4)
    products = [ALL_FILTER] + distinct_products(all_runs)
    with f1:
        filter_product = st.selectbox('Product', products, key='compare_filter_product')
    loc_options = [ALL_FILTER] + distinct_locations(all_runs, filter_product)
    with f2:
        filter_location = st.selectbox('Location', loc_options, key='compare_filter_location')
    with f3:
        use_ts_min = st.checkbox('Min timesteps', value=False, key='compare_use_ts_min')
        ts_min = st.number_input(
            'Min timesteps', min_value=0, value=1000, step=1000,
            disabled=not use_ts_min, key='compare_ts_min',
            label_visibility='collapsed',
        )
    with f4:
        use_ts_max = st.checkbox('Max timesteps', value=False, key='compare_use_ts_max')
        ts_max = st.number_input(
            'Max timesteps', min_value=0, value=100000, step=1000,
            disabled=not use_ts_max, key='compare_ts_max',
            label_visibility='collapsed',
        )

    filtered = filter_runs(
        all_runs,
        product=filter_product,
        location=filter_location,
        timesteps_min=int(ts_min) if use_ts_min else None,
        timesteps_max=int(ts_max) if use_ts_max else None,
    )

    if not filtered:
        st.warning('No saved runs match the current filters.')
        return

    id_to_summary = {s.run_id: s for s in filtered}
    selected_ids = st.multiselect(
        f'Select runs to compare (max {MAX_COMPARE_RUNS})',
        options=[s.run_id for s in filtered],
        format_func=lambda rid: id_to_summary[rid].label,
        max_selections=MAX_COMPARE_RUNS,
        key='compare_selected_runs',
    )

    if len(selected_ids) < 2:
        st.info('Select at least **2 runs** to compare.')
        return

    try:
        loaded_runs = [cached_load_run(rid) for rid in selected_ids]
    except Exception as e:
        st.error(f'Failed to load run data: {e}')
        return

    hist_weeks = [len(r.records) for r in loaded_runs]
    if len(set(hist_weeks)) > 1:
        st.warning(
            f'Runs have different historical week counts ({min(hist_weeks)}–{max(hist_weeks)}). '
            f'Charts are trimmed to the shortest common period ({min(hist_weeks)} weeks).'
        )

    products_sel = {r.config.get('product') for r in loaded_runs}
    locations_sel = {r.config.get('location') for r in loaded_runs}
    if len(products_sel) > 1 or len(locations_sel) > 1:
        st.warning('Selected runs use different product/location combinations. Demand reference uses the first run.')

    st.subheader('KPI Comparison')
    kpi_df = compute_comparison_kpis(loaded_runs)
    st.dataframe(kpi_df, use_container_width=True, hide_index=True)

    best_cost = min(loaded_runs, key=lambda r: float(r.config.get('total_cost', float('inf'))))
    st.caption(
        f'Lowest total cost: **{best_cost.summary.label}** '
        f'(€{best_cost.config.get("total_cost", 0):,.0f})'
    )

    st.subheader('Overlay Dashboard')
    compare_visible = st.multiselect(
        'Visible series',
        options=COMPARE_SERIES,
        default=DEFAULT_COMPARE_VISIBLE,
        help='Toggle metric groups for all selected runs. Use the chart legend for individual runs.',
        key='compare_visible_series',
    )

    fig = build_comparison_figure(loaded_runs, visible_series=compare_visible)
    st.plotly_chart(fig, use_container_width=True)


def main():
    inject_tum_styles()
    render_tum_header()
    st.markdown(
        'Configure parameters, train the PPO agent, and explore results in an interactive dashboard.'
    )

    config = render_sidebar()

    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    if 'training' not in st.session_state:
        st.session_state.training = False
    if 'awaiting_large_run_confirm' not in st.session_state:
        st.session_state.awaiting_large_run_confirm = False

    tab_current, tab_compare = st.tabs(['Current Run', 'Compare Runs'])

    with tab_current:
        col_btn, _col_info = st.columns([1, 3])
        with col_btn:
            start = st.button(
                'Start Training',
                type='primary',
                disabled=st.session_state.training or st.session_state.awaiting_large_run_confirm,
                use_container_width=True,
            )

        progress_bar = st.progress(0)
        status_text = st.empty()

        if start:
            if config.timesteps > LARGE_TIMESTEPS_THRESHOLD:
                st.session_state.awaiting_large_run_confirm = True
            else:
                execute_training(config, progress_bar, status_text)

        if st.session_state.awaiting_large_run_confirm:
            st.warning(
                f'You selected **{config.timesteps:,} timesteps** (more than '
                f'{LARGE_TIMESTEPS_THRESHOLD:,}). Training may take a very long time and '
                f'the UI will be blocked until it finishes. Are you sure you want to continue?'
            )
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button('Yes, start training', type='primary', use_container_width=True):
                    st.session_state.awaiting_large_run_confirm = False
                    execute_training(config, progress_bar, status_text)
            with cancel_col:
                if st.button('Cancel', use_container_width=True):
                    st.session_state.awaiting_large_run_confirm = False
                    status_text.info('Training cancelled.')

        st.divider()
        render_current_run_tab(st.session_state.last_result)

    with tab_compare:
        render_compare_tab()


if __name__ == '__main__':
    main()
