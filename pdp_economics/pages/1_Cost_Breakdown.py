import streamlit as st
import streamlit.components.v1 as components
# import st_debug as d
import altair as alt

from typing import Union

from datetime import date, timedelta

import numpy as np
import pandas as pd
import jax.numpy as jnp

import mechafil_jax.data as data
import mechafil_jax.sim as sim
import mechafil_jax.constants as C
import mechafil_jax.minting as minting
import mechafil_jax.date_utils as du

import scenario_generator.utils as u

import utils  # streamlit runs from root directory, so we can import utils directly

st.set_page_config(
    page_title="Cost Breakdown",
    page_icon="🚀",  # TODO: can update this to the FIL logo
    layout="wide",
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# local_css("debug.css")

def compute_costs(scenario2erpt=None):
    filp_multiplier = st.session_state['filp_multiplier']
    rd_multiplier = st.session_state['rd_multiplier']
    cc_multiplier = st.session_state['cc_multiplier']
    pdp_multiplier = st.session_state['pdp_multiplier']

    onboarding_scenario = st.session_state['onboarding_scenario'].lower()
    
    exchange_rate =  st.session_state['filprice_slider']
    borrowing_cost_pct = st.session_state['borrow_cost_pct'] / 100.0
    filp_bd_cost_tib_per_yr = st.session_state['filp_bizdev_cost']
    rd_bd_cost_tib_per_yr = st.session_state['rd_bizdev_cost']
    deal_income_tib_per_yr = st.session_state['deal_income']
    pdp_deal_income_tib_per_yr = st.session_state['pdp_deal_income']
    data_prep_cost_tib_per_yr = st.session_state['data_prep_cost']
    penalty_tib_per_yr = st.session_state['cheating_penalty']

    power_cost_tib_per_yr = st.session_state['power_cost']
    bw_cost_tib_per_yr = st.session_state['bw_cost']
    staff_cost_tib_per_yr = st.session_state['staff_cost']

    df = utils.compute_costs(
        scenario2erpt=scenario2erpt,
        filp_multiplier=filp_multiplier, rd_multiplier=rd_multiplier, cc_multiplier=cc_multiplier, pdp_multiplier=pdp_multiplier,
        onboarding_scenario=onboarding_scenario,
        exchange_rate=exchange_rate, borrowing_cost_pct=borrowing_cost_pct,
        filp_bd_cost_tib_per_yr=filp_bd_cost_tib_per_yr, rd_bd_cost_tib_per_yr=rd_bd_cost_tib_per_yr,
        deal_income_tib_per_yr=deal_income_tib_per_yr, pdp_deal_income_tib_per_yr=pdp_deal_income_tib_per_yr,
        data_prep_cost_tib_per_yr=data_prep_cost_tib_per_yr, penalty_tib_per_yr=penalty_tib_per_yr,
        power_cost_tib_per_yr=power_cost_tib_per_yr, bandwidth_10gbps_tib_per_yr=bw_cost_tib_per_yr,
        staff_cost_tib_per_yr=staff_cost_tib_per_yr
    )
    print(df)
    plot_costs(df)

def plot_costs(df):
    acounting_chart = alt.Chart(df, title="Net Income").mark_bar().encode(
        x=alt.X('SP Type', sort='-y', title=""),
        y=alt.Y('profit', title="($/TiB/Yr)"),
        color=alt.Color('SP Type', scale=alt.Scale(scheme='tableau20')),
    ).configure_axis(
        labelAngle=0,
        labelFontSize=20,
        titleFontSize=20
    ).properties(height=300)
    st.altair_chart(acounting_chart, use_container_width=True)
    
    df_copy = df.copy()
    for c in df_copy.columns:
        if 'cost' in c:
            df_copy[c] = df_copy[c] * -1
    df_copy = df_copy.drop(columns=['profit'])
    df_positive = df_copy[['SP Type', 'block_rewards', 'deal_income']]
    df_negative = df_copy[['SP Type', 'pledge_cost', 'gas_cost', 'power_cost', 
                 'bandwidth_cost', 'staff_cost', 'sealing_cost', 'data_prep_cost', 
                 'bd_cost', 'extra_copy_cost', 'cheating_cost']]
    dff_positive = pd.melt(df_positive, id_vars=['SP Type'])
    dff_negative = pd.melt(df_negative, id_vars=['SP Type'])
    # dff = pd.melt(df_copy, id_vars=['SP Type'])
    # angelo_chart = alt.Chart(dff).mark_bar().encode(
    #         x=alt.X("value:Q", title=""),
    #         y=alt.Y("SP Type:N", title=""),
    #         color=alt.Color("variable", type="nominal", title=""),
    #         order=alt.Order("variable", sort="descending"),
    #     )
    chart1 = (
    alt.Chart(dff_positive, title="Cost Breakdown").mark_bar().encode(
            x=alt.X("value:Q", title="($/TiB/Yr)"),
            y=alt.Y("SP Type:N", title=""),
            color=alt.Color(
                'variable',
                scale=alt.Scale(
                    scheme='greenblue'
                ),
                legend=alt.Legend(title='Revenue')
            ),
            order=alt.Order("variable", sort="descending")
        )
    )
    chart2 = (
        alt.Chart(dff_negative).mark_bar().encode(
            x=alt.X("value:Q", title="($/TiB/Yr)"),
            y=alt.Y("SP Type:N", title=""),
            color=alt.Color(
                'variable',
                scale=alt.Scale(
                    scheme='goldred'
                ),
                legend=alt.Legend(title='Costs')
            ),
            order=alt.Order("variable", sort="descending"),
        )
    )
    vline = (
        alt.Chart(pd.DataFrame({'x':[0]})).mark_rule(color='black').encode(x='x', strokeWidth=alt.value(2))
    )
    st.altair_chart((chart1+chart2).properties(height=500).configure_axis(labelFontSize=20, titleFontSize=20).resolve_scale(color='independent')+vline, use_container_width=True)

    # NOTE: not sure why formatting is not working
    format_mapping = {}
    for c in df.columns:
        if c != 'SP Type':
            format_mapping[c] = "{:.2f}"
    formatted_df = df.T.style.format(format_mapping)
    st.markdown("###### Cost Breakdown Table")
    st.write(formatted_df)

current_date = date.today() - timedelta(days=3)
mo_start = max(current_date.month - 1 % 12, 1)
start_date = date(current_date.year, mo_start, 1)
forecast_length_days=365*3
end_date = current_date + timedelta(days=forecast_length_days)
scenario2erpt = utils.get_offline_data(start_date, current_date, end_date)
compute_costs_kwargs = {
    'scenario2erpt':scenario2erpt
}

with st.sidebar:
    st.slider(
        "FIL Exchange Rate ($/FIL)", 
        min_value=3., max_value=50., value=4.0, step=.1, format='%0.02f', key="filprice_slider",
        on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
    )
    st.selectbox(
        'Onboarding Scenario', ('Status-Quo', 'Pessimistic', 'Optimistic'), key="onboarding_scenario",
        on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
    )                
    with st.expander("Revenue Settings", expanded=False):
        st.slider(
            'Deal Income ($/TiB/Yr)', 
            min_value=0.0, max_value=100.0, value=16.0, step=1.0, format='%0.02f', key="deal_income",
            on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'PDP Deal Income ($/TiB/Yr)',
            min_value=0.0, max_value=100.0, value=32.0, step=1.0, format='%0.02f', key="pdp_deal_income",
            on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
        )
    with st.expander("Cost Settings", expanded=False):
        st.slider(
            'Borrowing Costs (Pct. of Pledge)', 
            min_value=0.0, max_value=100.0, value=50.0, step=1.00, format='%0.02f', key="borrow_cost_pct",
            on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'FIL+ Biz Dev Cost ($/TiB/Yr)', 
            min_value=1.0, max_value=50.0, value=8.0, step=1.0, format='%0.02f', key="filp_bizdev_cost",
            on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'RD Biz Dev Cost ($/TiB/Yr)', 
            min_value=1.0, max_value=50.0, value=3.2, step=1.0, format='%0.02f', key="rd_bizdev_cost",
            on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'Data Prep Cost ($/TiB/Yr)', 
            min_value=0.0, max_value=50.0, value=1.0, step=1.0, format='%0.02f', key="data_prep_cost",
            on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'FIL+ Slashing Penalty ($/TiB/Yr)', 
            min_value=0.0, max_value=50.0, value=18.0, step=1.0, format='%0.02f', key="cheating_penalty",
            on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'Power+COLO Cost ($/TiB/Yr)', 
            min_value=0.0, max_value=50.0, value=6.0, step=1.0, format='%0.02f', key="power_cost",
            on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'Bandwidth [10GBPS] Cost ($/TiB/Yr)', 
            min_value=0.0, max_value=50.0, value=6.0, step=1.0, format='%0.02f', key="bw_cost",
            on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'Staff Cost ($/TiB/Yr)', 
            min_value=0.0, max_value=50.0, value=8.0, step=1.0, format='%0.02f', key="staff_cost",
            on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
        )
    with st.expander("Multipliers", expanded=False):
        st.slider(
            'CC', min_value=1, max_value=20, value=1, step=1, key="cc_multiplier",
            on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'RD', min_value=1, max_value=20, value=1, step=1, key="rd_multiplier",
            on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'PDP', min_value=1, max_value=20, value=1, step=1, key="pdp_multiplier",
            on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'FIL+', min_value=1, max_value=20, value=10, step=1, key="filp_multiplier",
            on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
        )
    
    st.button("Compute!", on_click=compute_costs, kwargs=compute_costs_kwargs, key="forecast_button")

# if "debug_string" in st.session_state:
#     st.markdown(
#         f'<div class="debug">{ st.session_state["debug_string"]}</div>',
#         unsafe_allow_html=True,
#     )
# components.html(
#     d.js_code(),
#     height=0,
#     width=0,
# )
