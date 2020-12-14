#!/usr/bin/env python
# coding: utf-8

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import pypsa

# Allow use of pyomo=False version of lopf() extra_functionality
from pypsa.linopt import get_var, linexpr, define_constraints

# Allow use of pyomo=True version of lopf() extra_functionality
from pyomo.environ import Constraint

import numpy as np
import pandas as pd

def fmt_float(x) :
    float_fmt_str = "{:6.2f}"
    return (float_fmt_str.format(x))

pd.set_option('float_format', fmt_float)
idx = pd.IndexSlice

# ## Required data
# 
# ### Wind and solar resource variabilities
# 
# From [Renewables.ninja Downloads](https://www.renewables.ninja/downloads):
# 
# - Solar time series "ninja_pv_europe_v1.1_sarah.csv" from [PV v1.1 Europe (.zip)](https://www.renewables.ninja/static/downloads/ninja_europe_pv_v1.1.zip)
# - Wind time series "ninja_wind_europe_v1.1_current_on-offshore.csv" from [Wind v1.1 Europe (.zip)](https://www.renewables.ninja/static/downloads/ninja_europe_wind_v1.1.zip)
# 
# ### IE Load (electricity demand) variability
# 
# From [eirgrid System and Renewable Data Reports](http://www.eirgridgroup.com/how-the-grid-works/renewables/):
# 
# - [System-Data-Qtr-Hourly-2018-2019.xlsx](http://www.eirgridgroup.com/site-files/library/EirGrid/System-Data-Qtr-Hourly-2018-2019.xlsx) 
# - [System-Data-Qtr-Hourly-2016-2017.xlsx](http://www.eirgridgroup.com/site-files/library/EirGrid/System-Data-Qtr-Hourly-2016-2017.xlsx)
# - [System-Data-Qtr-Hourly-2014-2015.xlsx](http://www.eirgridgroup.com/site-files/library/EirGrid/System-Data-Qtr-Hourly-2014-2015.xlsx)
# 

# ## Read in wind and solar variability data
# 
# **TODO:** Ideally, recode this to check for local copy, and, if not available, automatically download 
# and extract the required .csv from the .zip in each case; but for the moment, just assume there is are local copies of the .csv files already available.
# 
# **Alternative approach?** An alterative to using renewables ninja (specifically for wind) would be to extract the variability data (of actual wind generation) from historical eirgrid data. This would reflect the performance of the IE wind fleet as of whatever historical date was used: which may be a good thing or a bad thing of course (since that is almost 100% onshore for the moment, it is "biased against" offshore - arguably?).
# 
# **Validate/calibrate?** Would be good to calibrate/compare the (normalised) *wind* availability projected from the renewables ninja data with the actual recorded availability in the eirgrid data, for those years where both are available!
# 

logger.info("Loading solar and wind variability timeseries data (renewables.ninja)")

#rninja_base_url = "https://www.renewables.ninja/static/downloads/"
r_ninja_base_url = 'ninja/' # Actually already downloaded...

#solar_pv_zip_file = 'ninja_europe_pv_v1.1.zip'
#solar_pv_zip_url = r_ninja_base_url + solar_pv_zip_file

solar_pv_csv_file = 'ninja_pv_europe_v1.1_sarah.csv'
solar_pv_csv_url = r_ninja_base_url + solar_pv_csv_file

#read in renewables.ninja solar time series
solar_pu_raw = pd.read_csv(solar_pv_csv_url,
                       index_col=0,parse_dates=True)

#wind_zip_file = 'ninja_europe_wind_v1.1.zip'
#wind_zip_url = r_ninja_base_url + wind_zip_file

wind_csv_file = 'ninja_wind_europe_v1.1_current_on-offshore.csv'
wind_csv_url = r_ninja_base_url + wind_csv_file

#read in renewables.ninja wind time series
wind_pu_raw = pd.read_csv(wind_csv_url,
                       index_col=0,parse_dates=True)


# ## Read in and preprocess load variability data (via Ireland TSO, [EirGrid](http://www.eirgridgroup.com/))
# 
# We start with [historical data inputs from EirGrid](http://www.eirgridgroup.com/how-the-grid-works/renewables/) which show 15-minute time series for:
# 
# - wind availability
# - wind generation
# - total generation
# - total load
# 
# broken out by:
# 
# - IE (Republic of Ireland) only
# - NI (Northern Ireland) only
# 

# Retrieve example eirgrid load data into a pd.DataFrame

# If file already available locally, can point at that; otherwise use the web url
# (i.e. uncomment one or the other of the following two statements).

logger.info("Loading electricity demand timeseries data (eirgrid)")
#eirgrid_base_url = "http://www.eirgridgroup.com/site-files/library/EirGrid/"
eirgrid_base_url = "eirgrid/"

# Columns of interest:
cols = ['DateTime', 'GMT Offset', 'IE Demand', 'NI Demand']

load_data_raw = pd.DataFrame()
for base_year in [2014, 2016, 2018] :
    load_data_filename = F"System-Data-Qtr-Hourly-{base_year:4}-{(base_year+1):4}.xlsx"
    load_data_url = eirgrid_base_url + load_data_filename
    load_data_raw = pd.concat([load_data_raw, pd.read_excel(load_data_url, usecols = cols)], axis=0)

load_data_raw = load_data_raw.rename(columns={'IE Demand':'IE', 'NI Demand':'NI'})
load_data_raw['IE+NI'] = load_data_raw['IE']+load_data_raw['NI']
#display(load_data_raw)


# ## Fix the timestamps...
# 
# The raw eirgrid data has one column showing localtime (`DateTime`, type `pd.Timestamp`, holding "naive" timestamps - no recorded timezone) and a separate column showing the offset, in hours, from UTC for each individual row (`GMT Offset`). It will be simpler here to convert all the `DateTime` values to UTC (and explicitly having the UTC timezone).
# 
# We can then dispense with the `GMT Offset` column as it is redundant.

from datetime import timedelta

def tz_fix(row):
  try:
    naive_timestamp = row['DateTime']
    gmt_offset = row['GMT Offset'] 
    utc_timestamp = naive_timestamp - timedelta(hours=float(gmt_offset))
        # float() conversion required for timedelta() argument!
        # Must SUBTRACT the GMT Offset to get GMT/UTC
    row['DateTime'] = utc_timestamp.tz_localize('UTC')
  except Exception as inst:
    print(F"Exception:\n {row}")
    print(inst)
  return row

# This may be rather be slow for a big dataset...
# Is there a more efficient way of doing this?
load_data_raw = load_data_raw.apply(tz_fix, axis=1).drop(columns='GMT Offset')
load_data_raw.set_index('DateTime', verify_integrity=True, inplace=True)


# ## Data quality checks?
# 
# Minimal data quality check: make sure [we have no missing values](https://chartio.com/resources/tutorials/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe/) (either `None` or `NaN`).

assert(not load_data_raw.isnull().values.any())

#display(load_data_raw)


# ## Show some (raw) load profile stats?

def print_load_profile(load_col):
    print(F"\n\nLoad data col: {load_col:s}")
    
    load = load_data_raw.loc[:,load_col] # convert to pd.Series
    load_max = load.max()
    load_mean = load.mean()
    load_min = load.min()
    load_e = load.sum()*0.25 # Assume time interval is 15m == 0.25h

    #display(load)
    print(F"load_max: {(load_max/1.0e3) : 6.3f} GW")
    print(F"load_mean: {(load_mean/1.0e3) : 6.3f} GW")
    print(F"load_min: {(load_min/1.0e3) : 6.3f} GW")
    print(F"load_e: {(load_e/1.0e6) : 6.3f} TWh")

#print_load_profile('IE')
#print_load_profile('NI')
#print_load_profile('IE+NI')


# ## Required functions

# FIXME: need some docs/explanation/sources for this calculation (extract from WHOBS)?

def annuity(lifetime, rate):
    if rate == 0.0 :
        return 1.0/lifetime
    else:
        return rate/(1.0 - (1.0 / (1.0 + rate)**lifetime))

def prepare_assumptions(src="assumptions.csv",Nyears=1,usd_to_eur=1/1.2,assumptions_year=2020):
    """set all asset assumptions and other parameters"""

    assumptions = pd.read_csv('assumptions/'+src,index_col=list(range(3))).sort_index()

    #correct units to MW and EUR
    assumptions.loc[assumptions.unit.str.contains("/kW"),"value"]*=1e3
    assumptions.loc[assumptions.unit.str.contains("USD"),"value"]*=usd_to_eur

    assumptions = assumptions.loc[idx[:,assumptions_year,:],
                                  "value"].unstack(level=2).groupby(level="technology").sum(min_count=1)

    #fill defaults
    assumptions = assumptions.fillna({"FOM" : assumptions.at["default","FOM"],
                                      "discount rate" : assumptions.at["default","discount rate"],
                                      "lifetime" : assumptions.at["default","lifetime"]})

    #annualise investment costs, add FOM
    # (FOM = estimated "Follow On Maintenance", as % of initial capex, per annum?)
    assumptions["fixed"] = [(annuity(v["lifetime"],v["discount rate"]) + 
                             v["FOM"]/100.)*v["investment"]*Nyears for i,v in assumptions.iterrows()]

    return assumptions

def solve_network(run_configs, r_id):

    snapshot_interval = int(run_configs.loc['snapshot_interval',r_id])
    use_pyomo = bool(run_configs.loc['use_pyomo',r_id])
    solver_name = bool(run_configs.loc['solver_name',r_id])
    Nyears = int(run_configs.loc['Nyears',r_id])

    # Available year(s) for weather data: solar 1985-2015 inclusive, wind 1980-2016
    weather_year_start = int(run_configs.loc['weather_year_start',r_id])
    assert(weather_year_start >= 1985)
    weather_year_end = weather_year_start + (Nyears - 1)
    assert(weather_year_end <= 2015)

    if (run_configs.loc['constant_load_flag',r_id]) :
        load = run_configs.loc['constant_load (GW)',r_id]*1.0e3 # GW -> MW
    else :
        # Available year(s) for eirgrid load data: 2014-2019 inclusive
        load_year_start = int(run_configs.loc['load_year_start',r_id])
        assert(load_year_start >= 2014)
        load_year_end = load_year_start + (Nyears - 1)
        assert(load_year_end <= 2019)

        load_date_start = "{}-01-01 00:00".format(load_year_start)
        load_date_end = "{}-12-31 23:59".format(load_year_end)
        load_scope = run_configs.loc['load_scope',r_id]
        load = load_data_raw.loc[load_date_start:load_date_end, load_scope]
        load = load.resample(str(snapshot_interval)+"H").mean()

    solar_pu = solar_pu_raw.resample(str(snapshot_interval)+"H").mean()
    wind_pu = wind_pu_raw.resample(str(snapshot_interval)+"H").mean()
    # All this (re-)sampling may be a bit inefficient if doing multiple runs with the 
    # same snapshot_interval; but for the moment at least, we don't try to optimise around that
    # (e.g. by caching resampled timeseries for later use...)

    # CHECKME/FIXME: there *may* be a bug that can be triggered relating to the interaction 
    # of this resampling and leap-day filtering: vague memory of seeing that at some point. 
    # But don't currently have a test case demonstrating this... caveat modeller
    
    # Could just skip resampling by (instead) uncommenting:
    #solar_pu = solar_pu_raw
    #wind_pu = wind_pu_raw

    
    assumptions_src = run_configs.loc['assumptions_src',r_id]
    assumptions_year = int(run_configs.loc['assumptions_year',r_id])
    assert (assumptions_year in [2020, 2030, 2050])
    assumptions = prepare_assumptions(src=assumptions_src,Nyears=Nyears,
                                      assumptions_year=assumptions_year,
                                      usd_to_eur=run_configs.loc['usd_to_eur',r_id])

    network = pypsa.Network()

    snaps_df = pd.date_range("{}-01-01".format(weather_year_start),
                              "{}-12-31 23:00".format(weather_year_end),
                              freq=str(snapshot_interval)+"H").to_frame()

    snapshots = snaps_df[~((snaps_df.index.month == 2) & (snaps_df.index.day == 29))].index
    if (not run_configs.loc['constant_load_flag',r_id]) :
        load = load[~((load.index.month == 2) & (load.index.day == 29))]
        # Kludge to filter out "leap days" (29th Feb in any year)
        # https://stackoverflow.com/questions/34966422/remove-leap-year-day-from-pandas-dataframe
        # Necessary because we will want to combine arbitrary load years with arbitrary weather years...
        assert(load.count() == snapshots.size)
        load = load.values

    #display(snapshots)
    
    network.set_snapshots(snapshots)

    network.snapshot_weightings = pd.Series(float(snapshot_interval),index=network.snapshots)

    network.add("Bus","local-elec-grid")
    network.add("Load","local-elec-demand",
                bus="local-elec-grid",
                p_set= load)

    network.add("Generator","nuclear-SMR",
                bus="local-elec-grid",
                p_nom_extendable = True,
                p_nom_min = run_configs.at['nuclear_SMR_min_p (GW)', r_id]*1e3, #GW -> MW
                p_nom_max = run_configs.at['nuclear_SMR_max_p (GW)', r_id]*1e3, #GW -> MW
                marginal_cost = 1.0, # €/MWh FIXME!! This should probably come via "assumptions"?
                capital_cost = assumptions.at['Nuclear SMR','fixed'],
               )

    # Set very small VRE marginal cost to prefer curtailment to destroying energy in storage
    # (not sure of the rationale?).
    solar_marginal_cost = run_configs.loc['solar_marginal_cost',r_id] # €/MWh
    onshore_wind_marginal_cost = run_configs.loc['onshore_wind_marginal_cost',r_id] # €/MWh
    offshore_wind_marginal_cost = run_configs.loc['offshore_wind_marginal_cost',r_id]  # €/MWh
       
    network.add("Generator","solar",
                bus="local-elec-grid",
                p_max_pu = solar_pu["IE"], # Hardwired choice of IE location for renewables.ninja
                p_nom_extendable = True,
                p_nom_min = run_configs.at['solar_min_p (GW)', r_id]*1e3, #GW -> MW
                p_nom_max = run_configs.at['solar_max_p (GW)', r_id]*1e3, #GW -> MW
                marginal_cost = solar_marginal_cost, 
                #Small cost to prefer curtailment to destroying energy in storage
                capital_cost = assumptions.at['utility solar PV','fixed'],
               )

    network.add("Generator","onshore wind",
                bus="local-elec-grid",
                p_max_pu = wind_pu["IE_ON"], 
                    # Hardwired choice of IE location for renewables.ninja
                    # "_ON" codes for "onshore" in renewables.ninja wind data
                p_nom_extendable = True,
                p_nom_min = run_configs.at['onshore_wind_min_p (GW)', r_id]*1e3, #GW -> MW
                p_nom_max = run_configs.at['onshore_wind_max_p (GW)', r_id]*1e3, #GW -> MW
                marginal_cost = onshore_wind_marginal_cost, 
                #Small cost to prefer curtailment to destroying energy in storage, wind curtails before solar
                capital_cost = assumptions.at['onshore wind','fixed'])

    network.add("Generator","offshore wind",
                bus="local-elec-grid",
                p_max_pu = wind_pu["IE_OFF"], 
                    # Hardwired choice of IE location for renewables.ninja
                    # "_OFF" codes for "onshore" in renewables.ninja wind data
                p_nom_extendable = True,
                p_nom_min = run_configs.at['offshore_wind_min_p (GW)', r_id]*1e3, #GW -> MW
                p_nom_max = run_configs.at['offshore_wind_max_p (GW)', r_id]*1e3, #GW -> MW
                marginal_cost = offshore_wind_marginal_cost, 
                #Small cost to prefer curtailment to destroying energy in storage, wind curtails before solar
                capital_cost = assumptions.at['offshore wind','fixed'])

    # Model interconnection *very* crudely as an indefinitely (?) large external store, imposing
    # no local cost except for the interconnector to it. Set e_cyclic=True so that, over the 
    # modelled period, zero nett exchange, so that we don't have to 
    # pick (guess?) relative pricing for market-based import/export modelling.
    # from the local perspective we are just exploiting it as a "cheap" way to do temporal
    # shifting, once the interconnector is built and subject to the efficiency losses of the
    # interconnector (only). Of course this effectively excludes nett exports as a trade opportunity...
    # We do impose a (local system) capital charge on the interconnector itself; and assume that 
    # this is shared ~80:50 between the local system and remote-elec-grid (between IE state and European 
    # Union Funding in case of EWIC of 460:110 M€).
    # (https://www.irishtimes.com/news/east-west-interconnector-is-opened-1.737858)
    # This all skates over the NI integration connection, which arguably deserves finer 
    # grained representation (given similar wind var profile).

    network.add("Bus","remote-elec-grid")

    network.add("Store","remote-elec-grid-buffer",
                bus = "remote-elec-grid",
                e_nom_extendable = True,
                e_nom_max = run_configs.loc['IC_max_e (TWh)',r_id]*1.0e6, 
                         # TWh -> MWh
                e_cyclic=True,
                capital_cost=0.0) # Assume no local cost for existence of arbitrarily large ext grid

    # ic-export and ic-import links are two logical representations of the *same*
    # underlying hardware, operating in different directions. A single bi-directional link
    # representation is not possible if there are any losses, i.e., efficiency < 1.0. Note
    # addition below of global constraint, via extra_functionality(), to ensure import and
    # export "links" have the same p_nom (on their respective input sides).
    network.add("Link","ic-export",
                bus0 = "local-elec-grid",
                bus1 = "remote-elec-grid",
                efficiency = assumptions.at['interconnector','efficiency'],
                p_nom_extendable = True,
                p_nom_min = run_configs.at['IC_min_p (GW)', r_id]*1e3, # GW -> MW
                p_nom_max = run_configs.at['IC_max_p (GW)', r_id]*1e3, # GW -> MW
                capital_cost=assumptions.at['interconnector','fixed']*0.8
                 # Capital cost shared somewhat with remote-elec-grid operator(s)
                )
 
    network.add("Link","ic-import",
                bus0 = "remote-elec-grid",
                bus1 = "local-elec-grid",
                efficiency = assumptions.at['interconnector','efficiency'],
                p_nom_extendable = True,
                capital_cost=0.0
                 # Capital cost already accounted in ic-export view of link
                )
 
    # Battery storage
    network.add("Bus","battery")

    network.add("Store","battery storage",
                bus = "battery",
                e_nom_extendable = True,
                e_nom_max = run_configs.loc['Battery_max_e (MWh)',r_id],
                e_cyclic=True,
                capital_cost=assumptions.at['battery storage','fixed'])

    # "battery charge" and "battery discharge" links are two logical representations of the *same*
    # underlying hardware, operating in different directions. Note
    # addition below of global constraint, via extra_functionality(), to ensure charge and
    # discharge "links" have the same p_nom (on the network/grid side).
    network.add("Link","battery charge",
                bus0 = "local-elec-grid",
                bus1 = "battery",
                efficiency = assumptions.at['battery inverter','efficiency'],
                p_nom_extendable = True,
                p_nom_max = run_configs.at['Battery_max_p (MW)', r_id],
                capital_cost=assumptions.at['battery inverter','fixed'])

    network.add("Link","battery discharge",
                bus0 = "battery",
                bus1 = "local-elec-grid",
                efficiency = assumptions.at['battery inverter','efficiency'],
                p_nom_extendable = True,
                capital_cost=0.0
                 # Capital cost already accounted in battery charge view of link
                )

    network.add("Bus", "H2",
                     carrier="H2")

    h2_electrolysis_tech = 'H2 electrolysis ' + run_configs.loc['H2_electrolysis_tech',r_id]

    network.add("Link",
                    "H2 electrolysis",
                    bus1="H2",
                    bus0="local-elec-grid",
                    p_nom_extendable=True,
                    p_nom_max = run_configs.at['H2_electrolysis_max_p (GW)', r_id]*1e3, # GW -> MW
                    efficiency=assumptions.at["H2 electrolysis","efficiency"],
                    capital_cost=assumptions.at[h2_electrolysis_tech,"fixed"])

    network.add("Link",
                     "H2 CCGT",
                     bus0="H2",
                     bus1="local-elec-grid",
                     p_nom_extendable=True,
                     p_nom_max = run_configs.at['H2_CCGT_max_p (GW)', r_id]*1e3, # GW -> MW
                     efficiency=assumptions.at["H2 CCGT","efficiency"],
                     capital_cost=assumptions.at["H2 CCGT","fixed"]*assumptions.at["H2 CCGT","efficiency"])  
                     #NB: fixed (capital) cost for H2 CCGT in assumptions is per MWel (p1 of link)

    network.add("Link",
                     "H2 OCGT",
                     bus0="H2",
                     bus1="local-elec-grid",
                     p_nom_extendable=True,
                     p_nom_max = run_configs.at['H2_OCGT_max_p (GW)', r_id]*1e3, # GW -> MW
                     efficiency=assumptions.at["H2 OCGT","efficiency"],
                     capital_cost=assumptions.at["H2 OCGT","fixed"]*assumptions.at["H2 OCGT","efficiency"])  
                     #NB: fixed (capital) cost for H2 CCGT in assumptions is per MWel (p1 of link)

    h2_storage_tech = 'H2 ' + run_configs.loc['H2_storage_tech',r_id] + ' storage'

    network.add("Store",
                     "H2 store",
                     bus="H2",
                     e_nom_extendable=True,
                     e_nom_max = run_configs.loc['H2_store_max_e (TWh)',r_id]*1.0e6,
                         # TWh -> MWh
                     e_cyclic=True,
                     capital_cost=assumptions.at[h2_storage_tech,"fixed"])

    # Global constraints:
    
    # Interconnector import and export links are constrained so that rated power capacity at the 
    # *input* side (p0) is equal for both directions; so max available *output* power (p1) will 
    # be less, in both directions, via the configured efficiency.
    
    # Battery charge and discharge links are constrained so that rated power capacity at the 
    # network/grid bus (as opposed to the store bus) is equal for both charge and discharge.
    # (The implies that the rated power on the *input* side of the *discharge* link will be
    # correspondingly higher, via the configured efficiency.)
    
    def extra_functionality(network,snapshots):
        if use_pyomo :
            def ic(model):
                return (model.link_p_nom["ic-export"] 
                        == model.link_p_nom["ic-import"])

            network.model.ic = Constraint(rule=ic)

            def battery(model):
                return (model.link_p_nom["battery charge"] 
                        == (model.link_p_nom["battery discharge"] * 
                            network.links.at["battery charge","efficiency"]))
            network.model.battery = Constraint(rule=battery)

        else : # not use_pyomo
            link_p_nom = get_var(network, "Link", "p_nom")

            lhs = linexpr((1.0, link_p_nom["ic-export"]),
                           (-1.0, link_p_nom["ic-import"]))
            define_constraints(network, lhs, "=", 0.0, 'Link', 'ic_ratio')
 
            lhs = linexpr((1.0,link_p_nom["battery charge"]),
                          (-network.links.loc["battery discharge", "efficiency"],
                           link_p_nom["battery discharge"]))
            define_constraints(network, lhs, "=", 0.0, 'Link', 'battery_charger_ratio')

      
    if solver_name == "gurobi":
        solver_options = {"threads" : 4,
                          "method" : 2,
                          "crossover" : 0,
                          "BarConvTol": 1.e-5,
                          "FeasibilityTol": 1.e-6 }
    else:
        solver_options = {}


    network.consistency_check()

    network.lopf(solver_name=run_configs.loc['solver_name',r_id],
                 solver_options=solver_options,
                 pyomo=use_pyomo,
                 extra_functionality=extra_functionality)

    return network


def gather_run_stats(run_configs,run_stats,r_id,network):

    # FIXME: Add a sanity check that there are no snapshots where *both* electrolysis and 
    # H2 to power (whether CCGT or OCGT) are simultaneously dispatched!? (Unless there is
    # some conceivable circumstance in which it makes sense to take power over the interconnector
    # for electrolysis??) Of course, if we adding ramping constraints the situation would be 
    # quite different...
    
    snapshot_interval = run_configs.at['snapshot_interval',r_id]

    max_load_p = network.loads_t.p.sum(axis='columns').max()
    mean_load_p = network.loads_t.p.sum(axis='columns').mean()
    min_load_p = network.loads_t.p.sum(axis='columns').min()

    for g in network.generators.index :
        if (not(g in network.generators_t.p_max_pu.columns)) :
            # network.generators_t.p_max_pu is not defined for gens with static p_max_pu
            # but we want to do various generic calculations for *all* generators using this
            # so add it in for any such generators...
            network.generators_t.p_max_pu[g] = network.generators.at[g,'p_max_pu']
   
    total_load_e = (network.loads_t.p.sum().sum() * snapshot_interval)
    available_e = (network.generators_t.p_max_pu.multiply(network.generators.p_nom_opt).sum() 
        * snapshot_interval)
    total_available_e = available_e.sum()
    dispatched_e = network.generators_t.p.sum() * snapshot_interval
    total_dispatched_e = dispatched_e.sum()
    undispatched_e = (available_e - dispatched_e)
    total_undispatched_e = undispatched_e.sum()
    undispatched_frac = undispatched_e/available_e
    
    run_stats.at["System total load (TWh)",r_id] = total_load_e/1.0e6
    run_stats.at["System mean load (GW)",r_id] = mean_load_p/1.0e3

    run_stats.at["System available (TWh)",r_id] = total_available_e/1.0e6
    run_stats.at["System efficiency gross (%)",r_id] = (total_load_e/total_available_e)*100.0
        # "gross" includes dispatch down
    run_stats.at["System dispatched (TWh)",r_id] = total_dispatched_e/1.0e6
    run_stats.at["System dispatched down (TWh)",r_id] = total_undispatched_e/1.0e6
    run_stats.at["System dispatched down (%)",r_id] = (total_undispatched_e/total_available_e)*100.0
    run_stats.at["System storage loss (TWh)",r_id] = (total_dispatched_e-total_load_e)/1.0e6

    run_stats.at["System efficiency net (%)",r_id] = (total_load_e/total_dispatched_e)*100.0
        # "net" of dispatch down

    total_hours = network.snapshot_weightings.sum()

    # FIXME: add calculation of "min" LCOE for all gens (based on 100% capacity running)
    # Note that this doesn't depend on lopf() results - it is statically determined by
    # fixed and marginal costs of each gen.

    for g in network.generators.index :
        g_idx =  g
        run_stats.at[g+" capacity nom (GW)",r_id] = (
            network.generators.p_nom_opt[g_idx]/1.0e3)
        run_stats.at[g+" available (TWh)",r_id] = available_e[g_idx]/1.0e6
        run_stats.at[g+" dispatched (TWh)",r_id] = dispatched_e[g_idx]/1.0e6
        run_stats.at[g+" penetration (%)",r_id] = (dispatched_e[g_idx]/total_dispatched_e)*100.0 
        run_stats.at[g+" dispatched down (TWh)",r_id] = (undispatched_e[g_idx])/1.0e6
        run_stats.at[g+" dispatched down (%)",r_id] = (undispatched_frac[g_idx])*100.0
        run_stats.at[g+" capacity factor max (%)",r_id] = (
            network.generators_t.p_max_pu[g_idx].mean())*100.0
        run_stats.at[g+" capacity factor act (%)",r_id] = (
            dispatched_e[g_idx]/(network.generators.p_nom_opt[g_idx]*total_hours))*100.0
        
    links_e0 = network.links_t.p0.sum() * snapshot_interval
    links_e1 = network.links_t.p1.sum() * snapshot_interval

    ic_p = network.links.p_nom_opt["ic-export"]
    run_stats.at["IC power (GW)",r_id] = ic_p/1.0e3
        # NB: interconnector export and import p_nom are constrained to be equal
        # (at the input side of the respective links)
    ic_total_e = links_e0["ic-export"] - links_e1["ic-import"] # On IE grid side
    run_stats.at["IC transferred (TWh)",r_id] = ic_total_e/1.0e6
    run_stats.at["IC capacity factor (%)",r_id] = ic_total_e/(
        network.links.p_nom_opt["ic-export"]*total_hours)*100.0
    remote_elec_grid_e = network.stores.e_nom_opt["remote-elec-grid-buffer"]
    run_stats.at["remote-elec-grid 'store' (TWh)",r_id] = remote_elec_grid_e/1.0e6
    remote_elec_grid_h = remote_elec_grid_e/ic_p
    run_stats.at["remote-elec-grid 'store' time (h)",r_id] = remote_elec_grid_h
    run_stats.at["remote-elec-grid 'store' time (d)",r_id] = remote_elec_grid_h/24.0

    # Battery "expected" to be "relatively" small so we represent stats as MW (power) or MWh (energy)
    battery_charge_p = network.links.p_nom_opt["battery charge"]
    run_stats.at["Battery charge/discharge capacity nom (MW)",r_id] = battery_charge_p
        # NB: battery charge and discharge p_nom are constrained to be equal (grid side)
    battery_total_e = links_e0["battery charge"] - links_e1["battery discharge"] # on grid side
    run_stats.at["Battery transferred (GWh)",r_id] = battery_total_e/1.0e3
    run_stats.at["Battery capacity factor (%)",r_id] = (battery_total_e/(
        network.links.p_nom_opt["battery charge"]*total_hours))*100.0
    battery_store_e = network.stores.e_nom_opt["battery storage"]
    run_stats.at["Battery store (MWh)",r_id] = battery_store_e
    battery_discharge_p = network.links.p_nom_opt["battery discharge"]
    battery_store_h = battery_store_e/battery_discharge_p
    run_stats.at["Battery store time (h)",r_id] = battery_store_h
    #run_stats.at["Battery storage time (d)",r_id] = battery_store_h/24.0

    # P2H and H2P represent separate plant with separate capacity factors (0-100%); albeit, with 
    # no independent H2 load on the H2 bus, the sum of their respective capacity factors still 
    # has to be <=100% (as they will never run at the same time - that would always increase
    # system cost, as well as being just silly!)
    links = ["H2 electrolysis", "H2 OCGT", "H2 CCGT"]
    for l in links:
        l_idx =  l
        run_stats.at[l+" i/p capacity nom (GW)",r_id] = (network.links.p_nom_opt[l_idx]/1.0e3)
        run_stats.at[l+" o/p capacity nom (GW)",r_id] = (
            (network.links.p_nom_opt[l_idx]*network.links.efficiency[l_idx])/1.0e3)
        run_stats.at[l+" capacity factor (%)",r_id] = (
            links_e0[l_idx]/(network.links.p_nom_opt[l_idx]*total_hours))*100.0

    p2h2p_total_e = links_e0["H2 electrolysis"] - (links_e1["H2 OCGT"]+links_e1["H2 CCGT"]) 
                                                   # OCGT, CCGT both on grid side (e1)
    run_stats.at["P2H2P transferred (TWh)",r_id] = p2h2p_total_e/1.0e6    
    h2_store_e = network.stores.e_nom_opt["H2 store"]
    run_stats.at["H2 store (TWh)",r_id] = (h2_store_e/1.0e6)
    h2_store_CCGT_p = network.links.p_nom_opt["H2 CCGT"]
    h2_store_CCGT_h = h2_store_e/h2_store_CCGT_p
    run_stats.at["H2 store time (CCGT, h)",r_id] = h2_store_CCGT_h 
    run_stats.at["H2 store time (CCGT, d)",r_id] = h2_store_CCGT_h/24.0
    h2_store_OCGT_p = network.links.p_nom_opt["H2 OCGT"]
    h2_store_OCGT_h = h2_store_e/h2_store_OCGT_p
    run_stats.at["H2 store time (OCGT, h)",r_id] = h2_store_OCGT_h 
    run_stats.at["H2 store time (OCGT, d)",r_id] = h2_store_OCGT_h/24.0

    run_stats.at["System total raw store I+B+H2 (TWh)",r_id] = (
        h2_store_e+battery_store_e+remote_elec_grid_e)/1.0e6
    
    # Do a somewhat crude/ad hoc calculation of how much electricity can be generated
    # from the available storage, based on the efficiencies of the respective
    # conversion paths. This is further complicated for H2 storage in that there are two
    # possible conversion pathways (OCGT and CCGT). Since CCGT has higher efficiency, we
    # use it *unless* the deployed amount of CCGT is "negligible" compared to (mean) load_p...
    if (h2_store_CCGT_p > 0.01*mean_load_p) :
        h2_store_gen_efficiency = network.links.at["H2 CCGT","efficiency"]
    else :
        h2_store_gen_efficiency = network.links.at["H2 OCGT","efficiency"]
    total_avail_store_gen = ((h2_store_e*h2_store_gen_efficiency) +
                    (battery_store_e*network.links.at["battery discharge","efficiency"]) +
                    (remote_elec_grid_e*network.links.at["ic-import","efficiency"]))
    
    run_stats.at["System total usable store I+B+H2 (TWh)",r_id] = total_avail_store_gen/1.0e6
    total_avail_store_gen_h = total_avail_store_gen/mean_load_p
    run_stats.at["System total usable store/load (%) ",r_id] = (total_avail_store_gen/total_load_e)*100.0
    run_stats.at["System total usable store time (h)",r_id] = total_avail_store_gen_h
    run_stats.at["System total usable store time (d)",r_id] = total_avail_store_gen_h/24.0
                                                        
    run_stats.at["System notional cost (B€)",r_id] = network.objective/1.0e9 # Scale (by Nyears) to p.a.?
    run_stats.at["System notional LCOE (€/MWh)",r_id] = network.objective/total_load_e

    run_stats.at["Load weighted mean notional shadow price (€/MWh)",r_id] = (
        ((network.buses_t.marginal_price["local-elec-grid"]*network.loads_t.p["local-elec-demand"]).sum() * snapshot_interval)
                              / total_load_e)
        # This uses the WHOBS approach, based on shadow price at the ct bus, but now
        # (correctly?) weighted by the load at each snapshot. This implicitly assumes that all 
        # loads are indeed connected to the ct bus. This cost will, presumably, be
        # consistently higher than the "naive" (constant load) cost. Absent other constraints, it
        # should equal the system notional LCOE as calculated above. But constraints may give rise to
        # localised "profit" in certain sub-systems. See discussion here:
        # https://groups.google.com/g/pypsa/c/xXHmChzd8o8
    run_stats.at["Load max notional shadow price (€/MWh)",r_id] = (
        network.buses_t.marginal_price["local-elec-grid"].max())
    run_stats.at["Load min notional shadow price (€/MWh)",r_id] = (
        network.buses_t.marginal_price["local-elec-grid"].min())

    # All the following are "weighted means"
    run_stats.at["Offshore wind notional shadow cost (€/MWh)",r_id] = (
        ((network.buses_t.marginal_price["local-elec-grid"]*network.generators_t.p["offshore wind"]).sum())
                              / network.generators_t.p["offshore wind"].sum())

    run_stats.at["Onshore wind notional shadow cost (€/MWh)",r_id] = (
        ((network.buses_t.marginal_price["local-elec-grid"]*network.generators_t.p["onshore wind"]).sum())
                              / network.generators_t.p["onshore wind"].sum())

    run_stats.at["Solar notional shadow cost (€/MWh)",r_id] = (
        ((network.buses_t.marginal_price["local-elec-grid"]*network.generators_t.p["solar"]).sum())
                              / network.generators_t.p["solar"].sum())

    run_stats.at["Battery charge notional shadow cost (€/MWh)",r_id] = (
        ((network.buses_t.marginal_price["local-elec-grid"]*network.links_t.p0["battery charge"]).sum())
                              / network.links_t.p0["battery charge"].sum())

    run_stats.at["Battery discharge notional shadow cost (€/MWh)",r_id] = (
        ((network.buses_t.marginal_price["local-elec-grid"]*network.links_t.p1["battery discharge"]).sum())
                              / network.links_t.p1["battery discharge"].sum())

    run_stats.at["IC export notional shadow cost (€/MWh)",r_id] = (
        ((network.buses_t.marginal_price["local-elec-grid"]*network.links_t.p0["ic-export"]).sum())
                              / network.links_t.p0["ic-export"].sum())

    run_stats.at["IC import notional shadow cost (€/MWh)",r_id] = (
        ((network.buses_t.marginal_price["local-elec-grid"]*network.links_t.p1["ic-import"]).sum())
                              / network.links_t.p1["ic-import"].sum())

    run_stats.at["Elec. for H2 notional shadow cost (€/MWh)",r_id] = (
        ((network.buses_t.marginal_price["local-elec-grid"]*network.links_t.p0["H2 electrolysis"]).sum())
                              / network.links_t.p0["H2 electrolysis"].sum())

    run_stats.at["H2 for CCGT notional shadow cost (€/MWh)",r_id] = (
        ((network.buses_t.marginal_price["H2"]*network.links_t.p0["H2 CCGT"]).sum())
                              / network.links_t.p0["H2 CCGT"].sum())

    run_stats.at["Elec. from H2 CCGT notional shadow cost (€/MWh)",r_id] = (
        ((network.buses_t.marginal_price["local-elec-grid"]*network.links_t.p1["H2 CCGT"]).sum())
                              / network.links_t.p1["H2 CCGT"].sum())

    run_stats.at["H2 for OCGT notional shadow cost (€/MWh)",r_id] = (
        ((network.buses_t.marginal_price["H2"]*network.links_t.p0["H2 OCGT"]).sum())
                              / network.links_t.p0["H2 OCGT"].sum())

    run_stats.at["Elec. from H2 OCGT notional shadow cost (€/MWh)",r_id] = (
        ((network.buses_t.marginal_price["local-elec-grid"]*network.links_t.p1["H2 OCGT"]).sum())
                              / network.links_t.p1["H2 OCGT"].sum())

    run_stats.at["H2 notional shadow cost (€/MWh)",r_id] = (
        ((network.buses_t.marginal_price["H2"]*network.links_t.p1["H2 electrolysis"]).sum())
                              / network.links_t.p1["H2 electrolysis"].sum())


    return
