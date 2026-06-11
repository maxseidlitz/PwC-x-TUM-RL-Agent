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
    run_training_pipeline,
)
from dashboard import ALL_SERIES, DEFAULT_VISIBLE, build_dashboard_figure  # noqa: E402
from tum_theme import inject_tum_styles, render_tum_header  # noqa: E402

st.set_page_config(
    page_title='PPO Inventory Optimizer | TUM',
    page_icon='🔷',
    layout='wide',
    initial_sidebar_state='expanded',
)

DATA_FILE = str(ROOT / DEFAULT_FILE_PATH)
LARGE_TIMESTEPS_THRESHOLD = 500_000


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
    max_order_qty = st.sidebar.number_input('Max order qty (units)', min_value=1, max_value=5000, value=200)
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
    except Exception as e:
        status_text.error(f'Training failed: {e}')
    finally:
        st.session_state.training = False


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

    col_btn, col_info = st.columns([1, 3])
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

    result = st.session_state.last_result
    if result is None:
        st.info('No training run yet. Adjust parameters in the sidebar and click **Start Training**.')
        return

    st.divider()
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
    )

    fig, _ = build_dashboard_figure(
        result.records,
        result.product,
        result.location,
        future_records=result.future_records,
        visible_series=visible,
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()
