#!/usr/bin/env python
# coding: utf-8

# SWIS-100 Batch run
# 
# This is a script for running a SWIS-100 "experiment" - a batch
# of single runs, driven from an array of run configurations
# provided via a single external (currently hard coded) .ods
# spreadsheet file.

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
import numpy as np
import pandas as pd
import swis100 as swis

# Arguably, this mapping should be put in the configs.ods input file and read from there
# as a preliminary step... but this works for the moment.

config_types = {
    'use_pyomo' : bool,
    'solver_name' : str,
    'assumptions_src' : str,
    'assumptions_year' : int,
    'usd_to_eur' : float,
    'constant_load_flag' : bool,
    'load_year_start' : int,
    'load_scope' : str,
    'snapshot_interval' : int,
    'nuclear_SMR_min_p (GW)' : float,
    'nuclear_SMR_max_p (GW)' : float,
    'weather_year_start' : int,
    'Nyears' : int,
    'solar_marginal_cost' : float,
    'onshore_wind_marginal_cost' : float,
    'offshore_wind_marginal_cost' : float,
    'offshore_wind_min_p (GW)' : float,
    'offshore_wind_max_p (GW)' : float,
    'onshore_wind_min_p (GW)' : float,
    'onshore_wind_max_p (GW)' : float,
    'solar_min_p (GW)' : float,
    'solar_max_p (GW)' : float,
    'IC_min_p (GW)' : float,
    'IC_max_p (GW)' : float,
    'IC_max_e (TWh)' : float,
    'Battery_max_p (MW)' : float,
    'Battery_max_e (MWh)' : float,
    'H2_electrolysis_tech' : str,
    'H2_electrolysis_max_p (GW)' : float,
    'H2_CCGT_max_p (GW)' : float,
    'H2_OCGT_max_p (GW)' : float,
    'H2_storage_tech' : str,
    'H2_store_max_e (TWh)' : float
}

# Note that we read the configs in a "transposed" form. This transposition hack is needed to 
# allow casting of types on the basis of config var: pd.read_excel() only allows this by col, 
# not row...

batch_configs = pd.read_excel('runs/batch-run/test_configs.ods',
                              #dtype=object,
                              header=0,
                              index_col=0,
                              sheet_name='transpose',
                              converters=config_types
                             )

#print(batch_configs)

batch_configs_dict = batch_configs.to_dict(orient='index')

batch_id = 'scratch'
batch_dir='runs/batch-run/'+batch_id
os.makedirs(batch_dir,exist_ok=True)

# ## Solve the system (batch-run, no parallelization support)

batch_stats = pd.DataFrame(dtype=object)
for run_id in batch_configs_dict.keys() :
    logger.info('run_id: '+run_id)
    run_config = batch_configs_dict[run_id]
    network = swis.solve_network(run_config)
    network.export_to_netcdf(batch_dir+'/'+run_id+'-network.nc') # Comment out if full network object data not to be saved
    batch_stats[run_id] = swis.gather_run_stats(run_config, network)

# Dump (volatile) run data to (persistent) file storage

batch_configs.to_excel(batch_dir+'/batch_config.ods')
batch_stats.to_excel(batch_dir+'/batch_stats.ods')




