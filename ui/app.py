"""Streamlit UI for PPO inventory optimization training and interactive dashboard."""

from __future__ import annotations

import sys
import time
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
UI_DIR = Path(__file__).resolve().parent
for path in (ROOT, UI_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inventory_ppo import (  # noqa: E402
    DEFAULT_CSV_PATH,
    DEFAULT_FILE_PATH,
    TrainingConfig,
    compute_kpis,
    list_locations_for_product,
    list_locations_for_product_csv,
    list_product_location_pairs,
    list_products,
    list_products_from_csv,
    list_scenarios,
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
    build_method_comparison_figure,
    build_policy_comparison_df,
    compute_comparison_kpis,
    format_metric_delta,
    ppo_kpis_for_table,
)
from benchmark_methods import (  # noqa: E402
    COMPARISON_FILENAME,
    METHOD_PPO,
    generate_benchmarks_for_selection,
    write_comparison_excel,
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
DEFAULT_CSV_FILE = str(ROOT / DEFAULT_CSV_PATH) if DEFAULT_CSV_PATH else ''
LARGE_TIMESTEPS_THRESHOLD = 500_000
MAX_COMPARE_RUNS = 5
TRAINING_GRAPH_MIN_WEEK = (2026, 18)
DEFAULT_TRAINING_PAIRS = [
    ('Ice Cream Strawberry Flavor', 'Logistics Hub Lissabon'),
    ('Ice Cream Chocolate Flavor', 'Logistics Hub Lissabon'),
    ('Ice Cream Strawberry Flavor', 'Logistics Hub Porto'),
    ('Ice Cream Chocolate Flavor', 'Logistics Hub Porto'),
]
OPTIMIZED_PARAMETERS = {
    'lead_time': 2,
    'initial_inventory': 0,
    'timesteps': 10_000,
    'learning_rate': 1e-3,
    'holding_cost': 13.0,
    'ordering_cost': 60.0,
    'lost_sales_cost': 2500.0,
    'max_order_qty': 200,
    'n_forecast_weeks': 4,
    'gamma': 0.99,
    'n_steps': 2048,
    'batch_size': 64,
}


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


@st.cache_data
def cached_scenarios(file_path):
    return list_scenarios(file_path)


@st.cache_data
def cached_products_csv(csv_path):
    return list_products_from_csv(csv_path)


@st.cache_data
def cached_locations_csv(csv_path, product):
    return list_locations_for_product_csv(csv_path, product)


@st.cache_data
def cached_product_location_pairs(file_path):
    return list_product_location_pairs(file_path)


@st.cache_data
def cached_product_location_pairs_csv(csv_path):
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    if len(cols) < 2:
        return []
    product_col, location_col = cols[0], cols[1]
    pairs = (
        df[[product_col, location_col]]
        .dropna()
        .astype(str)
        .apply(lambda col: col.str.strip())
        .drop_duplicates()
    )
    return sorted((row[product_col], row[location_col]) for _, row in pairs.iterrows())


def _pair_label(product, location):
    return f'{product} @ {location}'


def _default_pair_labels(available_pairs):
    available = set(available_pairs)
    defaults = [pair for pair in DEFAULT_TRAINING_PAIRS if pair in available]
    if defaults:
        return [_pair_label(product, location) for product, location in defaults]
    if not available_pairs:
        return []
    return [_pair_label(*available_pairs[0])]


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


def base_stock_baseline(base_stock_results):
    return base_stock_results[0] if base_stock_results else None


def _parse_week_label(value):
    try:
        week, year = str(value).split('.', 1)
        return int(year), int(week)
    except (TypeError, ValueError):
        return None


def _is_training_graph_week(value):
    parsed = _parse_week_label(value)
    if parsed is None:
        return True
    return parsed >= TRAINING_GRAPH_MIN_WEEK


def filter_records_for_training_graph(records):
    return [
        record
        for record in (records or [])
        if _is_training_graph_week(record.get('week'))
    ]


def filter_hist_for_training_graph(hist_demand, hist_week_labels):
    hist_demand = list(hist_demand) if hist_demand is not None else []
    hist_week_labels = list(hist_week_labels) if hist_week_labels is not None else []
    if not hist_demand or not hist_week_labels:
        return hist_demand, hist_week_labels
    pairs = [
        (demand, label)
        for demand, label in zip(hist_demand, hist_week_labels)
        if _is_training_graph_week(label)
    ]
    if not pairs:
        return [], []
    demand, labels = zip(*pairs)
    return list(demand), list(labels)


def filter_run_for_training_graph(run):
    hist_demand, hist_week_labels = filter_hist_for_training_graph(
        getattr(run, 'hist_demand', None),
        getattr(run, 'hist_week_labels', None),
    )
    return SimpleNamespace(
        **{
            **run.__dict__,
            'records': filter_records_for_training_graph(run.records),
            'future_records': filter_records_for_training_graph(run.future_records),
            'hist_demand': hist_demand,
            'hist_week_labels': hist_week_labels,
        }
    )


def _config_to_dict(config):
    if isinstance(config, dict):
        return dict(config)
    if is_dataclass(config):
        return asdict(config)
    return {
        key: getattr(config, key)
        for key in dir(config)
        if not key.startswith('_') and not callable(getattr(config, key))
    }


def _format_param_value(value, suffix=''):
    if value is None or value == '':
        return 'N/A'
    if isinstance(value, (list, tuple, set)):
        return ', '.join(str(v) for v in value) if value else 'None'
    if isinstance(value, float):
        text = f'{value:.5f}'.rstrip('0').rstrip('.')
        return f'{text}{suffix}'
    if isinstance(value, int):
        return f'{value:,}{suffix}'
    return f'{value}{suffix}'


def _run_parameter_rows(config, run_dir=None, lead_time=None):
    cfg = _config_to_dict(config)
    effective_lead_time = lead_time if lead_time is not None else cfg.get('lead_time')

    rows = [
        ('Data', 'Product', cfg.get('product')),
        ('Data', 'Location', cfg.get('location')),
        ('Data', 'Data source', cfg.get('csv_path') or cfg.get('file_path')),
        ('Data', 'Scenarios', cfg.get('scenarios')),
        ('Inventory', 'Lead time', effective_lead_time, ' weeks'),
        ('Inventory', 'Initial inventory', cfg.get('initial_inventory'), ' units'),
        ('Environment', 'Max order qty', cfg.get('max_order_qty'), ' units'),
        ('Environment', 'Forecast horizon', cfg.get('n_forecast_weeks'), ' weeks'),
        ('Training', 'Timesteps', cfg.get('timesteps')),
        ('Training', 'Learning rate', cfg.get('learning_rate')),
        ('Training', 'Gamma', cfg.get('gamma')),
        ('Training', 'n_steps', cfg.get('n_steps')),
        ('Training', 'Batch size', cfg.get('batch_size')),
        ('Cost model', 'Holding cost', cfg.get('holding_cost'), ' €/unit'),
        ('Cost model', 'Ordering cost', cfg.get('ordering_cost'), ' €/unit'),
        ('Cost model', 'Lost sales cost', cfg.get('lost_sales_cost'), ' €/unit'),
        ('Run', 'Started at', cfg.get('started_at')),
        ('Run', 'Finished at', cfg.get('finished_at')),
        ('Run', 'Duration', cfg.get('duration_seconds'), ' sec'),
        ('Run', 'Run directory', run_dir),
    ]
    parameter_rows = []
    for row in rows:
        group, name, value, *rest = row
        suffix = rest[0] if rest else ''
        parameter_rows.append({
            'Group': group,
            'Parameter': name,
            'Value': _format_param_value(value, suffix),
        })
    return parameter_rows


def render_run_parameters(config, run_dir=None, lead_time=None):
    with st.expander('Run Parameters', expanded=False):
        st.dataframe(
            pd.DataFrame(_run_parameter_rows(config, run_dir, lead_time)),
            width='stretch',
            hide_index=True,
        )


def build_run_parameters_comparison(runs):
    rows = []
    for run in runs:
        cfg = _config_to_dict(run.config)
        rows.append({
            row['Parameter']: row['Value']
            for row in _run_parameter_rows(
                cfg,
                run_dir=run.summary.path,
                lead_time=run.lead_time,
            )
        })
    return pd.DataFrame(rows)


def add_parameter_rows_to_kpi_table(kpi_df, runs):
    parameter_table = build_run_parameters_comparison(runs)
    rows = []
    run_columns = list(kpi_df.columns[1:])
    for parameter in parameter_table.columns:
        row = {'Metric': f'Parameter · {parameter}'}
        for run_column, value in zip(run_columns, parameter_table[parameter].tolist()):
            row[run_column] = value
        rows.append(row)
    return pd.concat([kpi_df, pd.DataFrame(rows)], ignore_index=True)


def latest_run_per_product_location(summaries):
    latest = {}
    for summary in summaries:
        key = (summary.product, summary.location)
        if all(key) and key not in latest:
            latest[key] = summary
    return list(latest.values())


def _aggregate_records(record_sets):
    if not record_sets:
        return []
    min_len = min(len(records) for records in record_sets)
    if min_len == 0:
        return []

    sum_keys = [
        'actual_demand',
        'arriving_qty',
        'inventory_after_arrival',
        'order_qty',
        'unmet_demand',
        'inventory',
        'reward',
        'holding_cost_total',
        'ordering_cost_total',
        'lost_sales_cost_total',
    ]
    aggregated = []
    for index in range(min_len):
        first = record_sets[0][index]
        row = {'week': first.get('week', f'W{index}')}
        if 'due' in first:
            row['due'] = first.get('due', '')
        for key in sum_keys:
            row[key] = sum(float(records[index].get(key, 0)) for records in record_sets)
        for key in ('actual_demand', 'arriving_qty', 'inventory_after_arrival', 'order_qty', 'unmet_demand', 'inventory'):
            row[key] = int(round(row[key]))
        aggregated.append(row)
    return aggregated


def _aggregate_hist_demand(runs):
    demand_sets = [list(getattr(run, 'hist_demand', []) or []) for run in runs]
    demand_sets = [values for values in demand_sets if values]
    if not demand_sets:
        return []
    min_len = min(len(values) for values in demand_sets)
    return [
        int(round(sum(values[index] for values in demand_sets)))
        for index in range(min_len)
    ]


def build_aggregated_graph_runs(loaded_runs, group_by, product_label=None, location_label=None):
    groups = {}
    for run in loaded_runs:
        key = run.config.get(group_by, '')
        if key:
            groups.setdefault(key, []).append(run)

    aggregated_runs = []
    for key, runs in groups.items():
        records = _aggregate_records([run.records for run in runs])
        if not records:
            continue
        future_records = _aggregate_records([run.future_records for run in runs if run.future_records])
        hist_demand = _aggregate_hist_demand(runs)
        hist_week_labels = list(getattr(runs[0], 'hist_week_labels', []) or [])
        if hist_demand:
            hist_week_labels = hist_week_labels[:len(hist_demand)]
        total_cost = sum(float(run.config.get('total_cost', 0)) for run in runs)
        total_demand = sum(sum(record.get('actual_demand', 0) for record in run.records) for run in runs)
        total_unmet = sum(sum(record.get('unmet_demand', 0) for record in run.records) for run in runs)
        total_ordered = sum(sum(record.get('order_qty', 0) for record in run.records) for run in runs)
        avg_inventory = (
            sum(sum(record.get('inventory', 0) for record in run.records) for run in runs)
            / max(sum(len(run.records) for run in runs), 1)
        )
        service_level = 100.0 * (1 - total_unmet / max(total_demand, 1))
        aggregate_label = key
        config = {
            **runs[0].config,
            'product': key if group_by == 'product' else (product_label or 'All products'),
            'location': key if group_by == 'location' else (location_label or 'All locations'),
            'aggregate_label': aggregate_label,
            'aggregate_group_by': group_by,
            'total_cost': total_cost,
            'service_level': service_level,
            'total_ordered': total_ordered,
            'avg_inventory': avg_inventory,
        }
        summary = SimpleNamespace(
            run_id=f'aggregate-{group_by}-{key}',
            path='',
            label=aggregate_label,
            product=config['product'],
            location=config['location'],
            timesteps=max(run.summary.timesteps for run in runs),
            total_cost=total_cost,
            service_level=service_level,
            started_at=max((run.summary.started_at for run in runs), default=''),
            has_records=True,
        )
        aggregated_runs.append(SimpleNamespace(
            summary=summary,
            config=config,
            records=records,
            future_records=future_records,
            lead_time=max((run.lead_time for run in runs), default=0),
            hist_demand=hist_demand,
            hist_week_labels=hist_week_labels,
            base_stock_results=[],
            per_scenario_records={},
        ))
    return aggregated_runs


def render_product_location_buttons(key_prefix):
    session_key = f'{key_prefix}_graph_view'
    if session_key not in st.session_state:
        st.session_state[session_key] = 'Product'

    view_btn_product, view_btn_location = st.columns(2)
    with view_btn_product:
        if st.button(
            'Product',
            type='primary' if st.session_state[session_key] == 'Product' else 'secondary',
            width='stretch',
            key=f'{key_prefix}_graph_product_button',
        ):
            st.session_state[session_key] = 'Product'
    with view_btn_location:
        if st.button(
            'Location',
            type='primary' if st.session_state[session_key] == 'Location' else 'secondary',
            width='stretch',
            key=f'{key_prefix}_graph_location_button',
        ):
            st.session_state[session_key] = 'Location'
    return st.session_state[session_key]


def latest_pair_runs_for_graph():
    eligible_runs = [s for s in cached_list_runs() if s.has_records]
    latest_pair_summaries = latest_run_per_product_location(eligible_runs)
    return [cached_load_run(s.run_id) for s in latest_pair_summaries]


def aggregated_graph_runs_for_view(view, selected_products=None, selected_locations=None):
    latest_pair_runs = latest_pair_runs_for_graph()
    product_label = None
    location_label = None
    if selected_products is not None:
        selected_set = set(selected_products)
        latest_pair_runs = [
            run for run in latest_pair_runs
            if run.config.get('product') in selected_set
        ]
        if len(selected_set) == 1:
            product_label = next(iter(selected_set))
        elif selected_set:
            product_label = 'Selected products'
    if selected_locations is not None:
        selected_set = set(selected_locations)
        latest_pair_runs = [
            run for run in latest_pair_runs
            if run.config.get('location') in selected_set
        ]
        if len(selected_set) == 1:
            location_label = next(iter(selected_set))
        elif selected_set:
            location_label = 'Selected locations'
    group_by = 'location' if view == 'Location' else 'product'
    return build_aggregated_graph_runs(
        latest_pair_runs,
        group_by=group_by,
        product_label=product_label,
        location_label=location_label,
    )


def _aggregate_option_label(run):
    return run.config.get('aggregate_label') or run.summary.label


def _run_performance_label(run, view):
    if view == 'Location':
        return run.config.get('location') or getattr(run.summary, 'location', '') or _aggregate_option_label(run)
    return run.config.get('product') or getattr(run.summary, 'product', '') or _aggregate_option_label(run)


def _performance_rows(view, graph_runs):
    rows = []
    label_col = 'Location' if view == 'Location' else 'Product'
    for run in graph_runs:
        records = list(getattr(run, 'records', []) or [])
        demand = sum(float(record.get('actual_demand', 0)) for record in records)
        inventory = sum(float(record.get('inventory', 0)) for record in records)
        quantity = sum(float(record.get('order_qty', 0)) for record in records)
        cost = sum(float(-record.get('reward', 0)) for record in records)
        unmet = sum(float(record.get('unmet_demand', 0)) for record in records)
        service_level = 100.0 * (1 - unmet / max(demand, 1))
        rows.append({
            label_col: _run_performance_label(run, view),
            'Cumulative Demand': int(round(demand)),
            'Cumulative Inventory': int(round(inventory)),
            'Cumulative Quantity': int(round(quantity)),
            'Cumulative Cost (€)': round(cost, 2),
            'Service Level (%)': round(service_level, 2),
        })
    return rows


def render_performance_summary(view, graph_runs):
    if not graph_runs:
        return
    st.caption(f'{view} performance summary')
    st.dataframe(
        pd.DataFrame(_performance_rows(view, graph_runs)),
        width='stretch',
        hide_index=True,
    )


def render_aggregate_dropdown(view, key_prefix):
    if view == 'Location':
        latest_pair_runs = latest_pair_runs_for_graph()
        product_options = sorted({
            run.config.get('product', '')
            for run in latest_pair_runs
            if run.config.get('product')
        })
        if not product_options:
            return []
        selected_products = st.multiselect(
            'Products',
            options=product_options,
            default=product_options,
            key=f'{key_prefix}_aggregate_product_filter',
            help='Filter the location lines by product. Each location line sums the selected product set.',
        )
        if not selected_products:
            return []
        if len(selected_products) == len(product_options):
            return aggregated_graph_runs_for_view(view)
        return aggregated_graph_runs_for_view(view, selected_products=selected_products)

    latest_pair_runs = latest_pair_runs_for_graph()
    location_options = sorted({
        run.config.get('location', '')
        for run in latest_pair_runs
        if run.config.get('location')
    })
    if not location_options:
        return []
    selected_locations = st.multiselect(
        'Locations',
        options=location_options,
        default=location_options,
        key=f'{key_prefix}_aggregate_location_filter',
        help='Filter the product lines by location. Each product line sums the selected location set.',
    )
    if not selected_locations:
        return []
    if len(selected_locations) == len(location_options):
        return aggregated_graph_runs_for_view(view)
    return aggregated_graph_runs_for_view(view, selected_locations=selected_locations)


def render_aggregate_graph_note(view, graph_runs):
    if not graph_runs:
        st.info(f'No saved runs are available to aggregate by {view.lower()}.')
    elif view == 'Location':
        st.caption('Location view: each line sums the selected product set for that location.')
    else:
        st.caption('Product view: each line sums the selected location set for that product.')


def reset_to_optimized_parameters(default_pair_labels):
    st.session_state['optimized_pair_labels'] = list(default_pair_labels)
    st.session_state['optimized_lead_time'] = OPTIMIZED_PARAMETERS['lead_time']
    st.session_state['optimized_initial_inventory'] = OPTIMIZED_PARAMETERS['initial_inventory']
    st.session_state['optimized_timesteps'] = OPTIMIZED_PARAMETERS['timesteps']
    st.session_state['optimized_learning_rate'] = OPTIMIZED_PARAMETERS['learning_rate']
    st.session_state['optimized_holding_cost'] = OPTIMIZED_PARAMETERS['holding_cost']
    st.session_state['optimized_ordering_cost'] = OPTIMIZED_PARAMETERS['ordering_cost']
    st.session_state['optimized_lost_sales_cost'] = OPTIMIZED_PARAMETERS['lost_sales_cost']
    st.session_state['optimized_max_order_qty'] = OPTIMIZED_PARAMETERS['max_order_qty']
    st.session_state['optimized_n_forecast_weeks'] = OPTIMIZED_PARAMETERS['n_forecast_weeks']
    st.session_state['optimized_gamma'] = OPTIMIZED_PARAMETERS['gamma']
    st.session_state['optimized_n_steps'] = OPTIMIZED_PARAMETERS['n_steps']
    st.session_state['optimized_batch_size'] = OPTIMIZED_PARAMETERS['batch_size']


def render_sidebar():
    import tempfile, os  # noqa: E401

    st.sidebar.header('Configuration')

    # ------------------------------------------------------------------
    # Data source: default CSV, optional upload, or legacy Excel fallback
    # ------------------------------------------------------------------
    st.sidebar.subheader('Data Source')
    uploaded_csv = st.sidebar.file_uploader(
        'Scenario CSV (optional — overrides default)',
        type=['csv'],
        help=(
            'Upload a custom scenario CSV. '
            'Leave empty to use the default file: '
            f'`{DEFAULT_CSV_PATH}`.'
        ),
    )

    csv_path = ''
    if uploaded_csv is not None:
        # Persist uploaded CSV to a temp file so pandas can read it by path.
        if 'csv_tmp_path' not in st.session_state or not Path(st.session_state['csv_tmp_path']).exists():
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            tmp.write(uploaded_csv.getvalue())
            tmp.close()
            st.session_state['csv_tmp_path'] = tmp.name
        else:
            with open(st.session_state['csv_tmp_path'], 'wb') as f:
                f.write(uploaded_csv.getvalue())
        csv_path = st.session_state['csv_tmp_path']
    elif DEFAULT_CSV_FILE and Path(DEFAULT_CSV_FILE).exists():
        csv_path = DEFAULT_CSV_FILE

    use_csv = bool(csv_path)

    # ------------------------------------------------------------------
    # Product / location combinations
    # ------------------------------------------------------------------
    if use_csv:
        try:
            product_location_pairs = cached_product_location_pairs_csv(csv_path)
        except Exception as e:
            st.sidebar.error(f'Could not parse CSV: {e}')
            st.stop()
    else:
        try:
            product_location_pairs = cached_product_location_pairs(DATA_FILE)
        except Exception as e:
            st.sidebar.error(f'Could not load data file: {e}')
            st.stop()

    if use_csv:
        csv_label = uploaded_csv.name if uploaded_csv is not None else DEFAULT_CSV_PATH
        st.sidebar.caption(f'CSV mode · `{csv_label}`')
    else:
        available_scenarios = cached_scenarios(DATA_FILE)
        if available_scenarios:
            selected_scenarios = st.sidebar.multiselect(
                'Scenarios',
                options=available_scenarios,
                default=available_scenarios,
                help=(
                    'Select one or more forecast scenarios to run. '
                    'When multiple are selected the results are averaged week-by-week '
                    'across all selected scenarios.'
                ),
            )
            if not selected_scenarios:
                st.sidebar.warning('Select at least one scenario.')
                selected_scenarios = available_scenarios[:1]
        else:
            selected_scenarios = []
        st.sidebar.caption(f'Data file: `{DEFAULT_FILE_PATH}`')

    default_pair_labels = _default_pair_labels(product_location_pairs)
    st.sidebar.subheader('Optimized Parameters')
    st.sidebar.caption('Stored defaults for the current PPO setup.')
    st.sidebar.button(
        'Reset to optimized parameters',
        use_container_width=True,
        on_click=reset_to_optimized_parameters,
        args=(default_pair_labels,),
    )

    # ------------------------------------------------------------------
    # Inventory parameters (only shown in CSV mode; Excel has them in the file)
    # ------------------------------------------------------------------
    if use_csv:
        st.sidebar.subheader('Inventory Parameters')
        lead_time = st.sidebar.number_input(
            'Lead time (weeks)',
            min_value=0,
            max_value=52,
            value=OPTIMIZED_PARAMETERS['lead_time'],
            key='optimized_lead_time',
        )
        initial_inventory = st.sidebar.number_input(
            'Initial inventory (units)',
            min_value=0,
            value=OPTIMIZED_PARAMETERS['initial_inventory'],
            key='optimized_initial_inventory',
        )
    else:
        lead_time = 2        # not used in Excel mode (read from file)
        initial_inventory = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    st.sidebar.subheader('Training')
    pair_options = [_pair_label(product, location) for product, location in product_location_pairs]
    pair_lookup = {
        _pair_label(product, location): (product, location)
        for product, location in product_location_pairs
    }
    selected_pair_labels = st.sidebar.multiselect(
        'Product/location combinations',
        options=pair_options,
        default=default_pair_labels,
        key='optimized_pair_labels',
        help=(
            'Each selected combination is trained as a separate saved run. '
            'The defaults cover two products at the same location and two locations '
            'for the same product, so the product and location aggregate views have '
            'cumulative data to compare.'
        ),
    )
    if not selected_pair_labels:
        st.sidebar.warning('Select at least one product/location combination.')
        st.stop()

    selected_pairs = [pair_lookup[label] for label in selected_pair_labels]
    product, location = selected_pairs[0]

    _suggested_qty = 200
    if not use_csv:
        try:
            demand_data, *_ = load_data(DATA_FILE, product, location)
            _suggested_qty = suggest_max_order_qty(demand_data)
        except Exception:
            pass

    timesteps = st.sidebar.number_input(
        'Timesteps',
        min_value=1000,
        value=OPTIMIZED_PARAMETERS['timesteps'],
        step=1000,
        key='optimized_timesteps',
    )
    learning_rate = st.sidebar.number_input(
        'Learning rate',
        min_value=1e-5,
        max_value=1e-1,
        value=OPTIMIZED_PARAMETERS['learning_rate'],
        format='%.5f',
        key='optimized_learning_rate',
    )

    st.sidebar.subheader('Cost Model')
    holding_cost = st.sidebar.number_input(
        'Holding cost (€/unit)',
        min_value=0.0,
        value=OPTIMIZED_PARAMETERS['holding_cost'],
        key='optimized_holding_cost',
    )
    ordering_cost = st.sidebar.number_input(
        'Ordering cost (€/unit)',
        min_value=0.0,
        value=OPTIMIZED_PARAMETERS['ordering_cost'],
        key='optimized_ordering_cost',
    )
    lost_sales_cost = st.sidebar.number_input(
        'Lost sales cost (€/unit)',
        min_value=0.0,
        value=OPTIMIZED_PARAMETERS['lost_sales_cost'],
        key='optimized_lost_sales_cost',
    )

    st.sidebar.subheader('Environment')
    max_order_qty = st.sidebar.number_input(
        'Max order qty (units)',
        min_value=1,
        max_value=5000,
        value=OPTIMIZED_PARAMETERS['max_order_qty'],
        key='optimized_max_order_qty',
        help=f'Suggested: {_suggested_qty} (3× peak weekly demand). '
             'If set much higher than needed the agent must find a needle-in-a-haystack '
             'action and will likely produce excess lost sales.',
    )
    n_forecast_weeks = st.sidebar.number_input(
        'Forecast horizon (weeks)',
        min_value=1,
        max_value=12,
        value=OPTIMIZED_PARAMETERS['n_forecast_weeks'],
        key='optimized_n_forecast_weeks',
    )

    with st.sidebar.expander('Advanced PPO settings'):
        gamma = st.number_input(
            'Gamma',
            min_value=0.0,
            max_value=1.0,
            value=OPTIMIZED_PARAMETERS['gamma'],
            format='%.4f',
            key='optimized_gamma',
        )
        n_steps = st.number_input(
            'n_steps',
            min_value=64,
            max_value=8192,
            value=OPTIMIZED_PARAMETERS['n_steps'],
            step=64,
            key='optimized_n_steps',
        )
        batch_size = st.number_input(
            'Batch size',
            min_value=32,
            max_value=4096,
            value=OPTIMIZED_PARAMETERS['batch_size'],
            step=32,
            key='optimized_batch_size',
        )

    if use_csv:
        return [
            TrainingConfig(
                csv_path=csv_path,
                product=pair_product,
                location=pair_location,
                lead_time=int(lead_time),
                initial_inventory=int(initial_inventory),
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
            for pair_product, pair_location in selected_pairs
        ]
    return [
        TrainingConfig(
            file_path=DATA_FILE,
            csv_path='',
            product=pair_product,
            location=pair_location,
            scenarios=selected_scenarios,
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
        for pair_product, pair_location in selected_pairs
    ]


def execute_training(configs, progress_bar, status_text):
    configs = list(configs if isinstance(configs, (list, tuple)) else [configs])
    if not configs:
        status_text.error('Training failed: no product/location combinations selected.')
        return

    st.session_state.training = True
    total_configs = len(configs)
    total_steps = sum(config.timesteps for config in configs)
    progress_state = {
        'start': time.time(),
        'completed_steps': 0,
        'combo_index': 1,
        'combo_label': '',
    }

    def on_progress(current, total):
        completed = progress_state['completed_steps'] + current
        pct = min(completed / total_steps, 1.0) if total_steps else 0.0
        progress_bar.progress(pct)
        elapsed = time.time() - progress_state['start']
        if completed > max(total_steps * 0.02, 50):
            eta = elapsed / completed * (total_steps - completed)
        else:
            eta = None
        status_text.markdown(
            f'**Training {progress_state["combo_index"]}/{total_configs}:** '
            f'{progress_state["combo_label"]} · '
            f'step {current:,} / {total:,} · {format_eta(eta)}'
        )

    try:
        results = []
        with st.spinner('Training and evaluating selected combinations…'):
            for index, config in enumerate(configs, start=1):
                progress_state['combo_index'] = index
                progress_state['combo_label'] = _pair_label(config.product, config.location)
                result = run_training_pipeline(
                    config,
                    progress_callback=on_progress,
                    verbose=False,
                )
                results.append(result)
                progress_state['completed_steps'] += config.timesteps
        st.session_state.last_result = results[-1]
        st.session_state.last_results = results
        progress_bar.progress(1.0)
        total_duration = sum(result.duration_seconds for result in results)
        if total_configs == 1:
            result = results[0]
            status_text.success(
                f'Training complete in {result.duration_seconds:.1f}s · '
                f'artifacts saved to `{result.run_dir}`'
            )
        else:
            status_text.success(
                f'{total_configs} training runs complete in {total_duration:.1f}s · '
                'open Compare Runs to view the aggregate product/location graphs.'
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

    base_stock_results = getattr(result, 'base_stock_results', None) or []

    # --- Szenario-Auswahl: einzelnes Szenario oder Durchschnitt über alle ---
    per_sc = getattr(result, 'per_scenario_records', {}) or {}
    scenario_names = list(per_sc.keys())
    has_multiple = len(scenario_names) > 1

    AVERAGE_LABEL = f'Average (all {len(scenario_names)} scenarios)'

    if has_multiple:
        options = [AVERAGE_LABEL] + scenario_names
        selected = st.selectbox(
            'Scenario',
            options=options,
            index=0,
            key='current_scenario_select',
            help='View results for a specific scenario or the week-by-week average across all selected scenarios.',
        )
        if selected == AVERAGE_LABEL:
            display_records = result.records
            kpis = {
                'total_cost': result.total_cost,
                'service_level': result.service_level,
                'total_ordered': result.total_ordered,
                'avg_inventory': result.avg_inventory,
            }
        else:
            display_records = per_sc[selected]
            kpis = compute_kpis(display_records)
    else:
        display_records = result.records
        kpis = {
            'total_cost': result.total_cost,
            'service_level': result.service_level,
            'total_ordered': result.total_ordered,
            'avg_inventory': result.avg_inventory,
        }

    # --- KPI-Tabelleninput, jetzt szenario-abhängig gespeist ---
    ppo_table_kpis = ppo_kpis_for_table(
        kpis['total_cost'], kpis['service_level'],
        kpis['total_ordered'], kpis['avg_inventory'],
    )

    st.subheader('Key Performance Indicators')

    ref_bs = base_stock_baseline(base_stock_results)
    ref_kpis = ref_bs['kpis'] if ref_bs else None

    k1, k2, k3, k4, k5 = st.columns(5)

    if ref_kpis:
        d_cost, c_cost = format_metric_delta(
            ppo_table_kpis['Total Cost (€)'], ref_kpis['Total Cost (€)'], lower_is_better=True,
        )
        d_sl, c_sl = format_metric_delta(
            ppo_table_kpis['Service Level (%)'], ref_kpis['Service Level (%)'], lower_is_better=False,
        )
        d_ord, c_ord = format_metric_delta(
            ppo_table_kpis['Total Ordered'], ref_kpis['Total Ordered'], lower_is_better=True,
        )
        d_inv, c_inv = format_metric_delta(
            ppo_table_kpis['Avg Inventory'], ref_kpis['Avg Inventory'], lower_is_better=True,
        )
        k1.metric('Total Cost (PPO)', f'€{kpis["total_cost"]:,.0f}', delta=d_cost, delta_color=c_cost)
        k2.metric('Service Level (PPO)', f'{kpis["service_level"]:.1f}%', delta=d_sl, delta_color=c_sl)
        k3.metric('Total Ordered (PPO)', f'{kpis["total_ordered"]:,} units', delta=d_ord, delta_color=c_ord)
        k4.metric('Avg Inventory (PPO)', f'{kpis["avg_inventory"]:,.0f} units', delta=d_inv, delta_color=c_inv)
    else:
        k1.metric('Total Cost (PPO)', f'€{kpis["total_cost"]:,.0f}')
        k2.metric('Service Level (PPO)', f'{kpis["service_level"]:.1f}%')
        k3.metric('Total Ordered (PPO)', f'{kpis["total_ordered"]:,} units')
        k4.metric('Avg Inventory (PPO)', f'{kpis["avg_inventory"]:,.0f} units')
    k5.metric('Forecast Weeks', len(display_records))

    if base_stock_results:
        st.caption('Policy comparison (PPO vs. Base Stock baseline)')
        policy_df = build_policy_comparison_df(ppo_table_kpis, base_stock_results)
        st.dataframe(policy_df, width='stretch', hide_index=True)

    cfg = result.config
    parameter_config = {
        **_config_to_dict(cfg),
        'started_at': result.started_at,
        'finished_at': result.finished_at,
        'duration_seconds': result.duration_seconds,
    }
    render_run_parameters(parameter_config, run_dir=result.run_dir, lead_time=result.lead_time)

    csv_path_used = cfg.get('csv_path', '') if isinstance(cfg, dict) else getattr(cfg, 'csv_path', '')
    scenarios_used = cfg.get('scenarios') if isinstance(cfg, dict) else getattr(cfg, 'scenarios', [])
    if csv_path_used:
        st.caption(f'CSV mode · Run directory: `{result.run_dir}`')
    elif scenarios_used:
        scenario_label = ', '.join(scenarios_used)
        avg_note = ' (averaged)' if len(scenarios_used) > 1 else ''
        st.caption(f'Scenarios: **{scenario_label}**{avg_note} · Run directory: `{result.run_dir}`')
    else:
        st.caption(f'Run directory: `{result.run_dir}`')

    st.subheader('Interactive Dashboard')
    current_graph_view = render_product_location_buttons('current')
    visible = st.multiselect(
        'Visible series',
        options=COMPARE_SERIES,
        default=DEFAULT_COMPARE_VISIBLE,
        help='Toggle metric groups in the chart. Use the dropdown above to choose products or locations.',
        key='current_visible_series',
    )

    graph_runs = [
        filter_run_for_training_graph(run)
        for run in render_aggregate_dropdown(current_graph_view, 'current')
    ]
    graph_runs = [run for run in graph_runs if run.records]
    render_aggregate_graph_note(current_graph_view, graph_runs)
    if current_graph_view == 'Location' and not graph_runs:
        st.info('Select at least one product to show location performance.')
        return
    if current_graph_view == 'Product' and not graph_runs:
        st.info('Select at least one location to show product performance.')
        return
    if graph_runs:
        render_performance_summary(current_graph_view, graph_runs)
        fig = build_comparison_figure(
            graph_runs,
            visible_series=visible,
            base_stock_results=[],
        )
        st.plotly_chart(fig, width='stretch')
        return

    try:
        benchmark_config = result.config
        if has_multiple and selected != AVERAGE_LABEL:
            benchmark_config = replace(result.config, scenarios=[selected])
        benchmark_records = generate_benchmarks_for_selection(benchmark_config)
        method_records = {
            METHOD_PPO: filter_records_for_training_graph(display_records),
            **{
                method: filter_records_for_training_graph(records)
                for method, records in benchmark_records.items()
            },
        }
        graph_hist_demand, graph_hist_week_labels = filter_hist_for_training_graph(
            getattr(result, 'hist_demand', None),
            getattr(result, 'hist_week_labels', None),
        )

        comparison_path = Path(result.run_dir) / COMPARISON_FILENAME
        write_comparison_excel(result.product, result.location, method_records, comparison_path)
        write_comparison_excel(result.product, result.location, method_records, ROOT / COMPARISON_FILENAME)

        fig = build_method_comparison_figure(
            result.product,
            result.location,
            method_records,
            hist_demand=graph_hist_demand,
            hist_week_labels=graph_hist_week_labels,
            visible_series=visible,
        )
        st.plotly_chart(fig, width='stretch')
    except Exception as e:
        st.warning(f'Could not generate benchmark comparison: {e}')
        fig, _ = build_dashboard_figure(
            filter_records_for_training_graph(display_records),
            result.product,
            result.location,
            future_records=filter_records_for_training_graph(result.future_records),
            visible_series=ALL_SERIES,
        )
        st.plotly_chart(fig, width='stretch')


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
    show_parameters = st.checkbox(
        'Show run parameters',
        value=True,
        key='compare_show_run_parameters',
    )
    kpi_df = compute_comparison_kpis(loaded_runs)
    if show_parameters:
        kpi_df = add_parameter_rows_to_kpi_table(kpi_df, loaded_runs)
    st.dataframe(kpi_df, width='stretch', hide_index=True)

    ref_bs_results = getattr(loaded_runs[0], 'base_stock_results', None) or []
    if ref_bs_results:
        st.caption('Base Stock reference (from first selected run)')
        ref_rows = []
        for bs in ref_bs_results:
            ref_rows.append({
                'Policy': f'Base Stock (S={bs["S"]})',
                **bs.get('kpis', {}),
            })
        st.dataframe(
            pd.DataFrame(ref_rows),
            width='stretch',
            hide_index=True,
        )

    best_cost = min(loaded_runs, key=lambda r: float(r.config.get('total_cost', float('inf'))))
    st.caption(
        f'Lowest total cost: **{best_cost.summary.label}** '
        f'(€{best_cost.config.get("total_cost", 0):,.0f})'
    )

    st.subheader('Overlay Dashboard')
    compare_graph_view = render_product_location_buttons('compare')
    graph_runs = [filter_run_for_training_graph(run) for run in loaded_runs]
    graph_ref_bs_results = ref_bs_results
    aggregate_runs = render_aggregate_dropdown(compare_graph_view, 'compare')
    if compare_graph_view == 'Location':
        graph_runs = [filter_run_for_training_graph(run) for run in aggregate_runs]
        graph_ref_bs_results = []
    elif aggregate_runs:
        graph_runs = [filter_run_for_training_graph(run) for run in aggregate_runs]
        graph_ref_bs_results = []
    render_aggregate_graph_note(compare_graph_view, aggregate_runs)
    if compare_graph_view == 'Location' and not graph_runs:
        st.info('Select at least one product to show location performance.')
        return
    if compare_graph_view == 'Product' and not graph_runs:
        st.info('Select at least one location to show product performance.')
        return
    render_performance_summary(compare_graph_view, graph_runs)

    compare_visible = st.multiselect(
        'Visible series',
        options=COMPARE_SERIES,
        default=DEFAULT_COMPARE_VISIBLE,
        help='Toggle metric groups in the chart. Use the dropdown above to choose products or locations.',
        key='compare_visible_series',
    )

    fig = build_comparison_figure(
        graph_runs,
        visible_series=compare_visible,
        base_stock_results=graph_ref_bs_results,
    )
    st.plotly_chart(fig, width='stretch')


def main():
    inject_tum_styles()
    render_tum_header()
    st.markdown(
        'Configure parameters, train the PPO agent, and explore results in an interactive dashboard.'
    )

    configs = render_sidebar()
    selected_run_count = len(configs)
    selected_timesteps = sum(config.timesteps for config in configs)

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
                width='stretch',
            )

        progress_bar = st.progress(0)
        status_text = st.empty()

        if start:
            if selected_timesteps > LARGE_TIMESTEPS_THRESHOLD:
                st.session_state.awaiting_large_run_confirm = True
            else:
                execute_training(configs, progress_bar, status_text)

        if st.session_state.awaiting_large_run_confirm:
            st.warning(
                f'You selected **{selected_run_count} run(s)** with '
                f'**{selected_timesteps:,} total timesteps** (more than '
                f'{LARGE_TIMESTEPS_THRESHOLD:,}). Training may take a very long time and '
                f'the UI will be blocked until it finishes. Are you sure you want to continue?'
            )
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button('Yes, start training', type='primary', width='stretch'):
                    st.session_state.awaiting_large_run_confirm = False
                    execute_training(configs, progress_bar, status_text)
            with cancel_col:
                if st.button('Cancel', width='stretch'):
                    st.session_state.awaiting_large_run_confirm = False
                    status_text.info('Training cancelled.')

        st.divider()
        render_current_run_tab(st.session_state.last_result)

    with tab_compare:
        render_compare_tab()


if __name__ == '__main__':
    main()
