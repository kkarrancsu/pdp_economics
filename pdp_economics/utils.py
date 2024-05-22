import streamlit as st
import streamlit.components.v1 as components

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

@st.cache_data
def get_offline_data(start_date, current_date, end_date):
    PUBLIC_AUTH_TOKEN='Bearer ghp_EviOPunZooyAagPPmftIsHfWarumaFOUdBUZ'
    offline_data = data.get_simulation_data(PUBLIC_AUTH_TOKEN, start_date, current_date, end_date)

    _, hist_rbp = u.get_historical_daily_onboarded_power(current_date-timedelta(days=180), current_date)
    _, hist_rr = u.get_historical_renewal_rate(current_date-timedelta(days=180), current_date)
    _, hist_fpr = u.get_historical_filplus_rate(current_date-timedelta(days=180), current_date)

    smoothed_last_historical_rbp = float(np.median(hist_rbp[-30:]))
    smoothed_last_historical_rr = float(np.median(hist_rr[-30:]))
    smoothed_last_historical_fpr = float(np.median(hist_fpr[-30:]))

    # run mechafil and compute expected block rewards. we only need to do this once
    scenarios = ['pessimistic', 'status-quo', 'optimistic']
    scenario_scalers = [0.5, 1.0, 1.5]

    forecast_length = (end_date-start_date).days
    sector_duration = 365
    lock_target = 0.3

    scenario2erpt = {}
    for ii, scenario_scaler in enumerate(scenario_scalers):    
        scenario = scenarios[ii]
        
        rbp = jnp.ones(forecast_length) * smoothed_last_historical_rbp * scenario_scaler
        rr = jnp.ones(forecast_length) * smoothed_last_historical_rr * scenario_scaler
        fpr = jnp.ones(forecast_length) * smoothed_last_historical_fpr
        
        simulation_results = sim.run_sim(
            rbp,
            rr,
            fpr,
            lock_target,
            start_date,
            current_date,
            forecast_length,
            sector_duration,
            offline_data
        )
        # need to get the index associated w/ the start of simulation
        ix_current_date = (current_date-start_date).days
        expected_rewards_per_sector_today = float(simulation_results['1y_return_per_sector'][ix_current_date])
    
        # extract the block-rewards per tib for each scenario
        sectors_per_tib = (1024**4) / C.SECTOR_SIZE
        brpt = expected_rewards_per_sector_today * sectors_per_tib
        scenario2erpt[scenario] = brpt
    
    return scenario2erpt

def get_negligible_costs(bandwidth_10gbps_tib_per_yr):
    # Definitions (we can make these configurable later, potentially)
    sealing_costs_tib_per_yr = 1.3

    gas_cost_tib_per_yr = (2250.+108.)/1024.
    gas_cost_without_psd_tib_per_yr = 108./1024.
    bandwidth_1gbps_tib_per_yr=bandwidth_10gbps_tib_per_yr/10.0

    return sealing_costs_tib_per_yr, gas_cost_tib_per_yr, gas_cost_without_psd_tib_per_yr, bandwidth_1gbps_tib_per_yr


def compute_costs(scenario2erpt=None, 
                  filp_multiplier=10, rd_multiplier=1, cc_multiplier=1, pdp_multiplier=1,
                  onboarding_scenario='status-quo',
                  exchange_rate=4.0, borrowing_cost_pct=50,
                  filp_bd_cost_tib_per_yr=8.0, rd_bd_cost_tib_per_yr=3.2,
                  deal_income_tib_per_yr=16.0,
                  pdp_deal_income_tib_per_yr=32.0,
                  data_prep_cost_tib_per_yr=1.0, penalty_tib_per_yr=0.0,
                  power_cost_tib_per_yr=6, 
                  bandwidth_10gbps_tib_per_yr=6, 
                  staff_cost_tib_per_yr=10
                  ):
    erpt = scenario2erpt[onboarding_scenario]
    
    sealing_costs_tib_per_yr, gas_cost_tib_per_yr, gas_cost_without_psd_tib_per_yr, bandwidth_1gbps_tib_per_yr = get_negligible_costs(bandwidth_10gbps_tib_per_yr)
    
    # create a dataframe for each of the miner profiles
    filp_miner = {
        'SP Type': 'FIL+',
        'block_rewards': erpt*exchange_rate*filp_multiplier,
        'deal_income': deal_income_tib_per_yr,
        'pledge_cost': erpt*exchange_rate*filp_multiplier*borrowing_cost_pct,
        'gas_cost': gas_cost_tib_per_yr,
        'power_cost': power_cost_tib_per_yr,
        'bandwidth_cost': bandwidth_10gbps_tib_per_yr,
        'staff_cost': staff_cost_tib_per_yr,
        'sealing_cost': sealing_costs_tib_per_yr,
        'data_prep_cost': data_prep_cost_tib_per_yr,
        'bd_cost': filp_bd_cost_tib_per_yr,
        'extra_copy_cost': (staff_cost_tib_per_yr+power_cost_tib_per_yr)*0.5,
        'cheating_cost': 0
    }
    rd_miner = {
        'SP Type': 'Regular Deal',
        'block_rewards': erpt*exchange_rate*rd_multiplier,
        'deal_income': deal_income_tib_per_yr,
        'pledge_cost': erpt*exchange_rate*rd_multiplier*borrowing_cost_pct,
        'gas_cost': gas_cost_tib_per_yr,
        'power_cost': power_cost_tib_per_yr,
        'bandwidth_cost': bandwidth_10gbps_tib_per_yr,
        'staff_cost': staff_cost_tib_per_yr,
        'sealing_cost': sealing_costs_tib_per_yr,
        'data_prep_cost': data_prep_cost_tib_per_yr,
        'bd_cost': rd_bd_cost_tib_per_yr,
        'extra_copy_cost': (staff_cost_tib_per_yr+power_cost_tib_per_yr)*0.5,
        'cheating_cost': 0
    }
    pdp_miner = {
        'SP Type': 'PDP',
        'block_rewards': erpt*exchange_rate*pdp_multiplier,
        'deal_income': pdp_deal_income_tib_per_yr,
        'pledge_cost': erpt*exchange_rate*pdp_multiplier*borrowing_cost_pct,
        'gas_cost': gas_cost_tib_per_yr,
        'power_cost': power_cost_tib_per_yr,
        'bandwidth_cost': bandwidth_10gbps_tib_per_yr,
        'staff_cost': staff_cost_tib_per_yr,
        'sealing_cost': 0,
        'data_prep_cost': 0,
        'bd_cost': rd_bd_cost_tib_per_yr,
        'extra_copy_cost': 0,
        'cheating_cost': 0
    }
    # filp_exploit_miner = {
    #     'SP Type':'V1-ExploitFIL+',
    #     'block_rewards': erpt*exchange_rate*filp_multiplier,
    #     'deal_income': 0,
    #     'pledge_cost': erpt*exchange_rate*filp_multiplier*borrowing_cost_pct,
    #     'gas_cost': gas_cost_tib_per_yr,
    #     'power_cost': power_cost_tib_per_yr,
    #     'bandwidth_cost': bandwidth_1gbps_tib_per_yr,
    #     'staff_cost': staff_cost_tib_per_yr,
    #     'sealing_cost': sealing_costs_tib_per_yr,
    #     'data_prep_cost': 1,
    #     'bd_cost': 0,
    #     'extra_copy_cost': 0,
    #     'cheating_cost': 0
    # }
    # filp_exploit_with_retrieval = {
    #     'SP Type':'V2-ExploitFIL+',
    #     'block_rewards': erpt*exchange_rate*filp_multiplier,
    #     'deal_income': 0,
    #     'pledge_cost': erpt*exchange_rate*filp_multiplier*borrowing_cost_pct,
    #     'gas_cost': gas_cost_tib_per_yr,
    #     'power_cost': power_cost_tib_per_yr,
    #     'bandwidth_cost': bandwidth_10gbps_tib_per_yr,
    #     'staff_cost': staff_cost_tib_per_yr,
    #     'sealing_cost': sealing_costs_tib_per_yr,
    #     'data_prep_cost': 1,
    #     'bd_cost': 0,
    #     'extra_copy_cost': (staff_cost_tib_per_yr*0.5+bandwidth_10gbps_tib_per_yr)*0.5,
    #     'cheating_cost': 0
    # }
    # filp_exploit_with_retrieval_and_slash = {
    #     'SP Type':'V3-ExploitFIL+',
    #     'block_rewards': erpt*exchange_rate*filp_multiplier,
    #     'deal_income': 0,
    #     'pledge_cost': erpt*exchange_rate*filp_multiplier*borrowing_cost_pct,
    #     'gas_cost': gas_cost_tib_per_yr,
    #     'power_cost': power_cost_tib_per_yr,
    #     'bandwidth_cost': bandwidth_10gbps_tib_per_yr,
    #     'staff_cost': staff_cost_tib_per_yr,
    #     'sealing_cost': sealing_costs_tib_per_yr,
    #     'data_prep_cost': 1,
    #     'bd_cost': 0,
    #     'extra_copy_cost': (staff_cost_tib_per_yr*0.5+bandwidth_10gbps_tib_per_yr)*0.5,
    #     'cheating_cost': penalty_tib_per_yr
    # }
    cc_miner = {
        'SP Type':'CC',
        'block_rewards': erpt*exchange_rate*cc_multiplier,
        'deal_income': 0,
        'pledge_cost': erpt*exchange_rate*borrowing_cost_pct*cc_multiplier,
        'gas_cost': gas_cost_without_psd_tib_per_yr,
        'power_cost': power_cost_tib_per_yr,
        'bandwidth_cost': bandwidth_1gbps_tib_per_yr,
        'staff_cost': staff_cost_tib_per_yr,
        'sealing_cost': sealing_costs_tib_per_yr,
        'data_prep_cost': 0,
        'bd_cost': 0,
        'extra_copy_cost': 0,
        'cheating_cost': 0
    }
    aws = {
        'SP Type':'AWS Reference',
        'block_rewards': 0,
        'deal_income': 6.6,
        'pledge_cost': 0,
        'gas_cost': 0,
        'power_cost': 0,
        'bandwidth_cost': 0,
        'staff_cost': 0,
        'sealing_cost': 0,
        'data_prep_cost': 0,
        'bd_cost': 0,
        'extra_copy_cost': 0,
        'cheating_cost': 0
    }
    # df = pd.DataFrame([filp_miner, rd_miner, filp_exploit_miner, filp_cheat_miner, cc_miner, aws])
    # df = pd.DataFrame([filp_miner, rd_miner, filp_exploit_miner, filp_exploit_with_retrieval, filp_exploit_with_retrieval_and_slash, cc_miner])
    df = pd.DataFrame([filp_miner, rd_miner, cc_miner, pdp_miner])
    # add final accounting to the DF
    revenue = df['block_rewards'] + df['deal_income']
    cost = (
        df['pledge_cost'] 
        + df['gas_cost'] 
        + df['power_cost'] 
        + df['bandwidth_cost'] 
        + df['staff_cost'] 
        + df['sealing_cost'] 
        + df['data_prep_cost'] 
        + df['bd_cost'] 
        + df['extra_copy_cost'] 
        + df['cheating_cost']
    )
    df['profit'] = revenue-cost

    return df