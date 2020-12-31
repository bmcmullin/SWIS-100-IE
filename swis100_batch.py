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
import sys

def main(argv):

    def usage():
        logger.error('Usage: "'+scriptname+' <path-to-batch-dir>"')
        sys.exit(os.EX_USAGE)

    scriptname=os.path.basename(argv[0])
    try :
        batch_dir=argv[1].strip('/')
    except :
        usage()
    if (not os.path.isdir(batch_dir)) :
        usage()
    if (not os.path.isfile(batch_dir+'/batch_config.ods')) :
        logger.error('batch_config.ods not present in dir: '+batch_dir)
        usage()

    import numpy as np
    import pandas as pd
    import swis100 as swis

    batch_configs = pd.read_excel('runs/batch-run/test_configs.ods',
                                  #dtype=object,
                                  header=0,
                                  index_col=0,
                                  sheet_name='swis-config',
                                  converters=swis.config_converters
                                 )

    #print(batch_configs)

    batch_configs_dict = batch_configs.to_dict(orient='index')

    # Solve the system (batch-run, no parallelization support)

    batch_stats = pd.DataFrame(dtype=object)
    for run_id in batch_configs_dict.keys() :
        logger.info('run_id: '+run_id)
        run_config = batch_configs_dict[run_id]
        network = swis.solve_network(run_config)
        network.export_to_netcdf(batch_dir+'/'+run_id+'-network.nc') # Comment out if full network object data not to be saved
        batch_stats[run_id] = swis.gather_run_stats(run_config, network)

    #batch_configs.to_excel(batch_dir+'/batch_config.ods')
    batch_stats.to_excel(batch_dir+'/batch_stats.ods')

if __name__ == "__main__":
   main(sys.argv)
