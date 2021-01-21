#!/usr/bin/env python
# coding: utf-8

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings

import pypsa

# Support use of lopf() extra_functionality() [pyomo=False version] 
from pypsa.linopt import get_var, linexpr, define_constraints

import numpy as np
import pandas as pd
def fmt_float(x) :
    float_fmt_str = "{:6.2f}"
    return (float_fmt_str.format(x))

pd.set_option('float_format', fmt_float)
idx = pd.IndexSlice

from datetime import timedelta

config_converters = {

    # Utility mapping dict for stype casting when reading in spreadsheet configuration data.
    # Arguably, this information should be put in the spreadsheet file and read from there
    # as a preliminary step... but this works for the moment.

    'ASHP_max_p (GW)' : float,
    'ASHP_min_p (GW)' : float,
    'BEV_max_p (GW)' : float,
    'BEV_min_p (GW)' : float,
    'Battery_max_e (MWh)' : float,
    'Battery_max_p (MW)' : float,
    'delta_CO2_atm_max (MtCO2)' : float,
    'FCEV_max_p (GW)' : float,
    'FCEV_min_p (GW)' : float,
    'H2_CCGT_max_p (GW)' : float,
    'H2_OCGT_max_p (GW)' : float,
    'H2_boiler_max_p (GW)' : float,
    'H2_boiler_min_p (GW)' : float,
    'H2_electrolysis_max_p (GW)' : float,
    'H2_electrolysis_tech' : str,
    'H2_storage_tech' : str,
    'H2_store_max_e (TWh)' : float,
    'IC_max_e (TWh)' : float,
    'IC_max_p (GW)' : float,
    'IC_min_p (GW)' : float,
    'Nyears' : int,
    'assumptions_year' : int,
    'constant_elec_load_flag' : bool,
    'constant_lo_temp_heat_load_flag' : bool,
    'constant_surface_transport_load_flag' : bool,
    'elec_load_scope' : str,
    'elec_load_year_start' : int,
    'heat_load_year_start' : int,
    'heat_year_start' : int,
    'nuclear_SMR_max_p (GW)' : float,
    'nuclear_SMR_min_p (GW)' : float,
    'offshore_wind_marginal_cost' : float,
    'offshore_wind_max_p (GW)' : float,
    'offshore_wind_min_p (GW)' : float,
    'onshore_wind_marginal_cost' : float,
    'onshore_wind_max_p (GW)' : float,
    'onshore_wind_min_p (GW)' : float,
    'snapshot_interval' : int,
    'solar_marginal_cost' : float,
    'solar_max_p (GW)' : float,
    'solar_min_p (GW)' : float,
    'solver_name' : str,
    'transport_load_year_start' : int,
    'usd_to_eur' : float,
    'weather_year_start' : int,
}


# ## Read in required static data

# ### Wind and solar resource, pu_raw variability data
logger.info("Reading solar and wind variability (pu) timeseries data (via renewables.ninja)")

# From [Renewables.ninja Downloads](https://www.renewables.ninja/downloads):
# 
# - Solar time series "ninja_pv_europe_v1.1_sarah.csv" from [PV v1.1 Europe (.zip)](https://www.renewables.ninja/static/downloads/ninja_europe_pv_v1.1.zip)
# - Wind time series "ninja_wind_europe_v1.1_current_on-offshore.csv" from [Wind v1.1 Europe (.zip)](https://www.renewables.ninja/static/downloads/ninja_europe_wind_v1.1.zip)

# **TODO:** Ideally, recode this to check for local file
# copy, and, if not available, automatically download and
# extract the required .csv from the .zip in each case; but
# for the moment, just assume there is are local copies of
# the .csv files already available.

# **Alternative approach?** An alterative to using renewables
# ninja (specifically for wind) would be to extract the
# variability data (of actual wind generation) from
# historical eirgrid data. This would reflect the performance
# of the IE wind fleet as of whatever historical date was
# used: which may be a good thing or a bad thing of course
# (since that is almost 100% onshore for the moment, it is
# "biased against" offshore - arguably?).

# Validate/calibrate? Would be good to calibrate/compare
# the (normalised) *wind* availability projected from the
# renewables ninja data with the actual recorded availability
# in the eirgrid data, for those years where both are
# available!

#rninja_base_url = "https://www.renewables.ninja/static/downloads/"
r_ninja_base_url = 'ninja/' # Actually already downloaded...

#solar_pv_zip_file = 'ninja_europe_pv_v1.1.zip'
#solar_pv_zip_url = r_ninja_base_url + solar_pv_zip_file

solar_pv_csv_file = 'ninja_pv_europe_v1.1_sarah.csv'
solar_pv_csv_url = r_ninja_base_url + solar_pv_csv_file

#read in renewables.ninja solar time series
solar_pu_raw = pd.read_csv(solar_pv_csv_url,
                           usecols=['time','IE'],
                           index_col='time',
                           parse_dates=True).tz_localize('UTC')

#wind_zip_file = 'ninja_europe_wind_v1.1.zip'
#wind_zip_url = r_ninja_base_url + wind_zip_file

wind_csv_file = 'ninja_wind_europe_v1.1_current_on-offshore.csv'
wind_csv_url = r_ninja_base_url + wind_csv_file

#read in renewables.ninja wind time series
wind_pu_raw = pd.read_csv(wind_csv_url,
                          usecols=['time','IE_ON','IE_OFF'],
                          index_col='time',
                          parse_dates=True).tz_localize('UTC')

# ### IE/NI electricity load (demand) data
logger.info("Reading electricity demand timeseries data (via eirgrid)")

# We start with [historical data inputs from
# EirGrid](http://www.eirgridgroup.com/how-the-grid-works/renewables/):
#
# - [System-Data-Qtr-Hourly-2018-2019.xlsx](http://www.eirgridgroup.com/site-files/library/EirGrid/System-Data-Qtr-Hourly-2018-2019.xlsx) 
# - [System-Data-Qtr-Hourly-2016-2017.xlsx](http://www.eirgridgroup.com/site-files/library/EirGrid/System-Data-Qtr-Hourly-2016-2017.xlsx)
# - [System-Data-Qtr-Hourly-2014-2015.xlsx](http://www.eirgridgroup.com/site-files/library/EirGrid/System-Data-Qtr-Hourly-2014-2015.xlsx)

# There show 15-minute time series for:
# - wind availability
# - wind generation
# - total generation
# - total load
# 
# broken out by:
# - IE (Republic of Ireland) only
# - NI (Northern Ireland) only

# If file already available locally, can point at that; otherwise use the web url
# (i.e. uncomment one or the other of the following two statements).

#eirgrid_base_url = "http://www.eirgridgroup.com/site-files/library/EirGrid/"
eirgrid_base_url = "eirgrid/"

# Columns of interest:
cols = ['DateTime', 'GMT Offset', 'IE Demand', 'NI Demand']

elec_load_data_raw = pd.DataFrame()
for base_year in [2014, 2016, 2018] :
    elec_load_data_filename = F"System-Data-Qtr-Hourly-{base_year:4}-{(base_year+1):4}.xlsx"
    elec_load_data_url = eirgrid_base_url + elec_load_data_filename
    elec_load_data_raw = pd.concat([elec_load_data_raw, pd.read_excel(elec_load_data_url, usecols = cols)], axis=0)

elec_load_data_raw = elec_load_data_raw.rename(columns={'IE Demand':'IE', 'NI Demand':'NI'})
elec_load_data_raw['IE+NI'] = elec_load_data_raw['IE']+elec_load_data_raw['NI']

# ## Fix the timestamps...

# The raw eirgrid data has one column showing localtime
# (`DateTime`, type `pd.Timestamp`, holding "naive" timestamps -
# no recorded timezone) and a separate column showing the offset,
# in hours, from UTC for each individual row (`GMT Offset`). It
# will be simpler here to convert all the `DateTime` values to
# UTC (and explicitly having the UTC timezone). We can then
# dispense with the `GMT Offset` column as it is redundant.

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
elec_load_data_raw = elec_load_data_raw.apply(tz_fix, axis=1).drop(columns='GMT Offset')
elec_load_data_raw.set_index('DateTime', verify_integrity=True, inplace=True)

# ## Electricity load data quality checks?

# Minimal data quality check: make sure [we have no missing
# values](https://chartio.com/resources/tutorials/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe/)
# (either `None` or `NaN`).
assert(not elec_load_data_raw.isnull().values.any())

# ## Show some (raw) electricity load profile stats?

def print_elec_load_profile(elec_load_col):
    print(F"\n\nElectricity load data col: {elec_load_col:s}")
    
    elec_load = elec_load_data_raw.loc[:,elec_load_col] # convert to pd.Series
    elec_load_max = elec_load.max()
    elec_load_mean = elec_load.mean()
    elec_load_min = elec_load.min()
    elec_load_e = elec_load.sum()*0.25 # Assume time interval is 15m == 0.25h

    #print(elec_load)
    print(F"elec_load_max: {(elec_load_max/1.0e3) : 6.3f} GW")
    print(F"elec_load_mean: {(elec_load_mean/1.0e3) : 6.3f} GW")
    print(F"elec_load_min: {(elec_load_min/1.0e3) : 6.3f} GW")
    print(F"elec_load_e: {(elec_load_e/1.0e6) : 6.3f} TWh")

#print_elec_load_profile('IE')
#print_elec_load_profile('NI')
#print_elec_load_profile('IE+NI')

# IE transport load (demand) data (via IE energy statistics agency, [SEAI](http://www.seai.ie/))
logger.info("Loading transport demand annual timeseries data (seai)")

# We start with [historical "energy flow" data inputs from
# SEAI](https://www.seai.ie/publications/Energy-by-Fuel.xlsx)
# which show annual resolution time series (from 1990) for energy
# flows by sector (Republic of Ireland only). These are
# subdivided by fuels: but as transport is currently dominated by
# liquid hydrocarbons fuels we assume the totals are essentially
# equal to this.  The raw data is in ktoe units, for flows "into"
# these transport sectors:

# Road Light Goods Vehicle
# Road Private Car
# Road Freight
# Public Passenger Services
# Rail
# Domestic Aviation
# International Aviation
# Fuel Tourism
# Navigation [assume dominated by international?]
# Unspecified

# If file already available locally, can point at that; otherwise use the web url
# (i.e. uncomment one or the other of the following two statements).

#seai_base_url = "https://www.seai.ie/publications/Energy-by-Fuel.xlsx"
seai_base_url = 'seai/'

transport_load_data_filename = 'Energy-by-Fuel.xlsx'
transport_load_data_url = seai_base_url + transport_load_data_filename
transport_load_data_sheet = 'Total'
transport_load_data_raw = pd.read_excel(
    transport_load_data_url,
    sheet_name=transport_load_data_sheet,
    index_col=0)['Transport':'Unspecified']
transport_load_data_raw = transport_load_data_raw.drop(columns='NACE').transpose()
transport_load_data_raw.index=pd.to_datetime(transport_load_data_raw.index,format="%Y")
transport_load_data_raw.index = transport_load_data_raw.index+timedelta(hours=(365.0*12.0))
        # anchor at mid-year (for later interpolation across years)
        # (this is off by 12 hours in leap years - but we neglect that!)
ktoe_to_MWh=11630.0 # https://www.unitjuggler.com/convert-energy-from-ktoe-to-MWh.html
transport_load_data_raw = (
    (transport_load_data_raw*ktoe_to_MWh)/(365.0*24))
    # Annual ktoe -> average continuous MW
    # (Neglect slight conversion error in leap years...)

transport_load_data_raw.columns.rename('Total (MWh)',inplace=True)
assert(not transport_load_data_raw.isnull().values.any())

logger.info("Loading low temperature (space and water) heating timeseries data (when2heat)")

# when2heat_base_url = 'https://data.open-power-system-data.org/when2heat/2019-08-06/'

# The when2heat dataset provides high time-resolution estimated
# timeseries for aggregate national (here, IE) water and space heating
# timeseries (in MW), as well as heat pump COP estimates. For
# full details on the methodology see:
# https://www.nature.com/articles/s41597-019-0199-y

# This does *not* include heat demand for purposes other than
# space and water heating: particularly higher temperature
# industrial process heat requirements...

when2heat_base_url = 'when2heat/'
when2heat_data_filename = 'when2heat.csv'
when2heat_data_url = when2heat_base_url + when2heat_data_filename

usecols = ['utc_timestamp',
           'IE_heat_demand_total', 'IE_heat_demand_space', 'IE_heat_demand_water',
           'IE_COP_ASHP_radiator', 'IE_COP_ASHP_water']

when2heat_data = pd.read_csv(
    when2heat_data_url,
    delimiter=';',
    decimal=',',
    usecols=usecols,
    index_col='utc_timestamp')
when2heat_data.index = pd.to_datetime(when2heat_data.index)
when2heat_data = when2heat_data.dropna()['2008-01-01':]
    # Discard initial small number of rows for very end of 2007;
    # and trailing NaN rows (presumably due to missing data in
    # the when2heat upstream data sources?). End result is that
    # we only have this data for the full years 2008-2013
    # inclusive.

# For simplicity in the coarse grained modelling, we aggregate
# space and water heating demand as "low_temp" demand; we further
# assume only ASHP, and only radiator use for space heating, and
# calculate an equivalent COP aggregated across both space and
# water (weighted by the relative demand levels in each
# snapshot).
when2heat_data['IE_lo_temp_heat_demand'] = (
    when2heat_data['IE_heat_demand_space'] + when2heat_data['IE_heat_demand_water'])
when2heat_data['IE_lo_temp_heat_supply'] = (
    (when2heat_data['IE_heat_demand_space']/when2heat_data['IE_COP_ASHP_radiator']) +
    (when2heat_data['IE_heat_demand_water']/when2heat_data['IE_COP_ASHP_water']))
when2heat_data['IE_COP_ASHP'] = (
    when2heat_data['IE_lo_temp_heat_demand'] / when2heat_data['IE_lo_temp_heat_supply'])
when2heat_data=when2heat_data.loc[:,['IE_lo_temp_heat_demand','IE_COP_ASHP']]
assert(not when2heat_data.isnull().values.any())

# Read raw technology assumptions data (will be further
# processed/refined for each run)
logger.info("Reading raw technology assumptions data (via assumptions/SWIS.ods)")

assumptions_raw = pd.read_excel('assumptions/SWIS.ods',
                                usecols=['technology','year','parameter','value','unit'],
                                index_col=list(range(3)),
                                header=0,
                                sheet_name='SWIS').sort_index()

# Required functions

def annuity(lifetime, rate):
    # FIXME: need some docs/explanation/sources for this calculation (extract from WHOBS)?
    if rate == 0.0 :
        return 1.0/lifetime
    else:
        return rate/(1.0 - (1.0 / (1.0 + rate)**lifetime))

def prepare_assumptions(Nyears=1,usd_to_eur=1/1.2,assumptions_year=2020):
    """set all asset assumptions and other parameters for specific run_config"""

    assumptions = assumptions_raw.copy(deep=True)

    #correct units to MW, EUR, tCO2
    assumptions.loc[assumptions.unit.str.contains("/kgCO2"),"value"]*=1e3
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

def solve_network(run_config):

    snapshot_interval = int(run_config['snapshot_interval'])
    solver_name = bool(run_config['solver_name'])
    Nyears = int(run_config['Nyears'])
    assumptions_year = int(run_config['assumptions_year'])
    assert (assumptions_year in [2020, 2030, 2050])

    assumptions = prepare_assumptions(Nyears=Nyears,
                                      assumptions_year=assumptions_year,
                                      usd_to_eur=run_config['usd_to_eur'])

    # Available year(s) for vre (pu) resource data: solar 1985-2015 inclusive, wind 1980-2016
    weather_year_start = int(run_config['weather_year_start'])
    assert(weather_year_start >= 1985)
    weather_year_end = weather_year_start + (Nyears - 1)
    assert(weather_year_end <= 2015)

    solar_pu = solar_pu_raw.resample(str(snapshot_interval)+"H").mean()
    wind_pu = wind_pu_raw.resample(str(snapshot_interval)+"H").mean()
    # All this (re-)sampling may be a bit inefficient if doing multiple runs with the 
    # same snapshot_interval; but for the moment at least, we don't try to optimise around that
    # (e.g. by caching resampled timeseries for later use...)

    # CHECKME/FIXME: there *may* be a bug that can be triggered relating to the interaction 
    # of this resampling and leap-day filtering: vague memory of seeing that at some point. 
    # But don't currently have a test case demonstrating this... caveat modeller
    
    # Could just skip resampling and just let pypsa sub-sample
    # (within lopf()); though note that this will no longer be
    # expected to exactly match overall average reseource pu
    # availability. But if such subsampling is preferred,
    # uncomment:

    #solar_pu = solar_pu_raw
    #wind_pu = wind_pu_raw

    # Configure Links to have multiple outputs by overriding the
    # component_attrs. This can be done for as many buses as you
    # need with format busi for i = 2,3,4,5,....  See
    # https://pypsa.org/doc/components.html#link-with-multiple-outputs-or-inputs
    override_component_attrs = pypsa.descriptors.Dict({k : v.copy() for k,v in pypsa.components.component_attrs.items()})
    override_component_attrs["Link"].loc["bus2"] = ["string",np.nan,np.nan,"2nd bus","Input (optional)"]
    override_component_attrs["Link"].loc["efficiency2"] = ["static or series","per unit",1.,"2nd bus efficiency","Input (optional)"]
    override_component_attrs["Link"].loc["p2"] = ["series","MW",0.,"2nd bus output","Output"]
    
    network = pypsa.Network(override_component_attrs=override_component_attrs)

    snapshots_df = pd.date_range("{}-01-01".format(weather_year_start),
                              "{}-12-31 23:00".format(weather_year_end),
                              freq=str(snapshot_interval)+"H", tz='UTC').to_frame()

    # Filter out leap days...
    snapshots = snapshots_df[~((snapshots_df.index.month == 2) & (snapshots_df.index.day == 29))].index

    #print(snapshots)
    
    network.set_snapshots(snapshots)

    network.snapshot_weightings = pd.Series(float(snapshot_interval),index=network.snapshots)

    network.add("Bus","local-elec-grid")

    # Configure required elec_load (constant or timeseries)
    if (run_config['constant_elec_load_flag']) :
        elec_load = run_config['constant_elec_load (GW)']*1.0e3 # GW -> MW
    else :
        # Available year(s) for eirgrid load data: 2014-2019 inclusive
        elec_load_year_start = int(run_config['elec_load_year_start'])
        assert(elec_load_year_start >= 2014)
        elec_load_year_end = elec_load_year_start + (Nyears - 1)
        assert(elec_load_year_end <= 2019)

        elec_load_date_start = "{}-01-01 00:00".format(elec_load_year_start)
        elec_load_date_end = "{}-12-31 23:59".format(elec_load_year_end)
        elec_load_scope = run_config['elec_load_scope']
        elec_load = elec_load_data_raw.loc[elec_load_date_start:elec_load_date_end, elec_load_scope]
        elec_load = elec_load.resample(str(snapshot_interval)+"H").mean()

        elec_load = elec_load[~((elec_load.index.month == 2) & (elec_load.index.day == 29))]
        # Kludge to filter out "leap days" (29th Feb in any year)
        # https://stackoverflow.com/questions/34966422/remove-leap-year-day-from-pandas-dataframe
        # Necessary because we will want to combine arbitrary electricity load years with
        # arbitrary weather years...
        assert(elec_load.count() == snapshots.size)
        elec_load = elec_load.values
       
    network.add("Load","local-elec-demand",
                bus="local-elec-grid",
                p_set= elec_load)

    network.add("Generator","nuclear-SMR",
                bus="local-elec-grid",
                p_nom_extendable = True,
                p_nom_min = run_config['nuclear_SMR_min_p (GW)']*1e3, #GW -> MW
                p_nom_max = run_config['nuclear_SMR_max_p (GW)']*1e3, #GW -> MW
                marginal_cost = 1.0, # €/MWh FIXME!! This should probably come via "assumptions"?
                capital_cost = assumptions.at['Nuclear SMR','fixed'],
               )

    # Set very small VRE marginal cost to prefer curtailment to destroying energy in storage
    # (not sure of the rationale?).
    solar_marginal_cost = run_config['solar_marginal_cost'] # €/MWh
    onshore_wind_marginal_cost = run_config['onshore_wind_marginal_cost'] # €/MWh
    offshore_wind_marginal_cost = run_config['offshore_wind_marginal_cost']  # €/MWh
       
    network.add("Generator","solar",
                bus="local-elec-grid",
                p_max_pu = solar_pu["IE"], # Hardwired choice of IE location for renewables.ninja
                p_nom_extendable = True,
                p_nom_min = run_config['solar_min_p (GW)']*1e3, #GW -> MW
                p_nom_max = run_config['solar_max_p (GW)']*1e3, #GW -> MW
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
                p_nom_min = run_config['onshore_wind_min_p (GW)']*1e3, #GW -> MW
                p_nom_max = run_config['onshore_wind_max_p (GW)']*1e3, #GW -> MW
                marginal_cost = onshore_wind_marginal_cost, 
                #Small cost to prefer curtailment to destroying energy in storage, wind curtails before solar
                capital_cost = assumptions.at['onshore wind','fixed'])

    network.add("Generator","offshore wind",
                bus="local-elec-grid",
                p_max_pu = wind_pu["IE_OFF"], 
                    # Hardwired choice of IE location for renewables.ninja
                    # "_OFF" codes for "onshore" in renewables.ninja wind data
                p_nom_extendable = True,
                p_nom_min = run_config['offshore_wind_min_p (GW)']*1e3, #GW -> MW
                p_nom_max = run_config['offshore_wind_max_p (GW)']*1e3, #GW -> MW
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
                e_nom_max = run_config['IC_max_e (TWh)']*1.0e6, 
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
                p_nom_min = run_config['IC_min_p (GW)']*1e3, # GW -> MW
                p_nom_max = run_config['IC_max_p (GW)']*1e3, # GW -> MW
                capital_cost=assumptions.at['interconnector','fixed']*0.8
                 # Capital cost "shared" somewhat (20%?) by remote-elec-grid operator(s)
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
                e_nom_max = run_config['Battery_max_e (MWh)'],
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
                p_nom_max = run_config['Battery_max_p (MW)'],
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

    h2_electrolysis_tech = 'H2 electrolysis ' + run_config['H2_electrolysis_tech']

    network.add("Link", "H2 electrolysis",
                bus1="H2",
                bus0="local-elec-grid",
                p_nom_extendable=True,
                p_nom_max = run_config['H2_electrolysis_max_p (GW)']*1e3, # GW -> MW
                efficiency=assumptions.at["H2 electrolysis","efficiency"],
                capital_cost=assumptions.at[h2_electrolysis_tech,"fixed"])

    network.add("Link", "H2 CCGT",
                bus0="H2",
                bus1="local-elec-grid",
                p_nom_extendable=True,
                p_nom_max = run_config['H2_CCGT_max_p (GW)']*1e3, # GW -> MW
                efficiency=assumptions.at["H2 CCGT","efficiency"],
                capital_cost=assumptions.at["H2 CCGT","fixed"]*assumptions.at["H2 CCGT","efficiency"])  
                #NB: fixed (capital) cost for H2 CCGT in assumptions is per MWel (p1 of link)

    network.add("Link", "H2 OCGT",
                bus0="H2",
                bus1="local-elec-grid",
                p_nom_extendable=True,
                p_nom_max = run_config['H2_OCGT_max_p (GW)']*1e3, # GW -> MW
                efficiency=assumptions.at["H2 OCGT","efficiency"],
                capital_cost=assumptions.at["H2 OCGT","fixed"]*assumptions.at["H2 OCGT","efficiency"])  
                #NB: fixed (capital) cost for H2 CCGT in assumptions is per MWel (p1 of link)

    h2_storage_tech = 'H2 ' + run_config['H2_storage_tech'] + ' storage'

    network.add("Store", "H2 store",
                bus="H2",
                e_nom_extendable=True,
                e_nom_max = run_config['H2_store_max_e (TWh)']*1.0e6,
                # TWh -> MWh
                e_cyclic=True,
                capital_cost=assumptions.at[h2_storage_tech,"fixed"])

    # Transport subsystem (surface only as yet: excludes aviation)
    network.add("Bus","surface_transport_final")

    # Configure required surface_transport_load

    surface_cols = ['Road Freight', 
                    'Road Light Goods Vehicle', 
                    'Road Private Car', 
                    'Public Passenger Services', 
                    'Rail',
                    'Fuel Tourism',
                    'Navigation',
                    'Unspecified']
    # We aggregate even 'Navigation' and 'Unspecified' into "surface" transport as a 
    # (very coarse!) heuristic.

    # Configure required surface_transport_load (constant or timeseries)
    if (run_config['constant_surface_transport_load_flag']) :
        surface_transport_load = run_config['constant_surface_transport_load (GW)']*1.0e3 # GW -> MW
    else :
        # Available year(s) for seai transport data are 1990-2018, but allowing for
        # interpolation, usable range is 1991-2017 inclusive
        surface_transport_load_year_start = int(run_config['transport_load_year_start'])
        assert(surface_transport_load_year_start >= 1991)
        surface_transport_load_year_end = surface_transport_load_year_start + (Nyears - 1)
        assert(surface_transport_load_year_end <= 2017)

        # We include an extra year before and after the years of interest to smooth the interpolation
        surface_transport_load = (
            transport_load_data_raw.loc[
                    str(surface_transport_load_year_start - 1) : 
                    str(surface_transport_load_year_end +1),
                    surface_cols].sum(axis=1)
                * assumptions.at['ICEV','efficiency'])
            # We count only the "final" ("wheel") energy as load, to
            # allow for deployment of more or less efficient upstream
            # converters (vehicle fleet), relative to current
            # ICE-dominated fleet. In this current instantiation we
            # just use a single, crude, fleet wide, conversion
            # efficiency assuming an "average" ICE conversion.

        surface_transport_load = (
            surface_transport_load.resample(str(snapshot_interval)+"H").interpolate())
        surface_transport_load = (surface_transport_load[
                ~((surface_transport_load.index.month == 2) & 
                  (surface_transport_load.index.day == 29))])
                # Filter out "leap days" (29th Feb in any year)
        surface_transport_load = (surface_transport_load[
            "{}-01-01 00:00".format(surface_transport_load_year_start) :
            "{}-12-31 23:59".format(surface_transport_load_year_end)])
                # Filter just the full years actually in scope
        assert(surface_transport_load.count() == snapshots.size)
        surface_transport_load = surface_transport_load.values

    network.add("Load","surface-transport-demand",
                bus="surface_transport_final",
                p_set= surface_transport_load)

    network.add("Link", "BEV", # tacitly includes possibility of battery electic shipping!?
                bus0="local-elec-grid",
                bus1="surface_transport_final",
                p_nom_extendable=True,
                p_nom_min = run_config['BEV_min_p (GW)']*1e3, # GW -> MW
                p_nom_max = run_config['BEV_max_p (GW)']*1e3, # GW -> MW
                efficiency=assumptions.at["BEV","efficiency"],
                capital_cost=assumptions.at["BEV","fixed"])

    network.add("Link", "FCEV", # tacitly includes possibility of HFC shipping!?
                bus0="H2",
                bus1="surface_transport_final",
                p_nom_extendable=True,
                p_nom_min = run_config['FCEV_min_p (GW)']*1e3, # GW -> MW
                p_nom_max = run_config['FCEV_max_p (GW)']*1e3, # GW -> MW
                efficiency=assumptions.at["FCEV","efficiency"],
                capital_cost=assumptions.at["FCEV","fixed"])

    # Heat subsystem: low temperature (space and water) heating only as yet: excludes industrial process heat.
    network.add("Bus","lo_temp_heat")
    
    # Configure required low_temp_heat_load (constant or timeseries) and COP
    # Available year(s) for when2heat data are 2008-2013 inclusive
    lo_temp_heat_year_start = int(run_config['heat_year_start'])
    assert(lo_temp_heat_year_start >= 2008)
    lo_temp_heat_year_end = lo_temp_heat_year_start + (Nyears - 1)
    assert(lo_temp_heat_year_end <= 2013)

    lo_temp_heat_load_date_start = "{}-01-01 00:00".format(lo_temp_heat_year_start)
    lo_temp_heat_load_date_end = "{}-12-31 23:59".format(lo_temp_heat_year_end)
    lo_temp_heat_data = when2heat_data.loc[lo_temp_heat_load_date_start:lo_temp_heat_load_date_end, ]
    lo_temp_heat_data = lo_temp_heat_data.resample(str(snapshot_interval)+"H").mean()
    lo_temp_heat_data = lo_temp_heat_data[~((lo_temp_heat_data.index.month == 2) & (lo_temp_heat_data.index.day == 29))]
    # Kludge to filter out "leap days" (29th Feb in any year)
    # https://stackoverflow.com/questions/34966422/remove-leap-year-day-from-pandas-dataframe
    # Necessary because we want to be able to combine arbitrary load years with
    # arbitrary weather years...
    assert(len(lo_temp_heat_data.index) == snapshots.size)

    if (run_config['constant_lo_temp_heat_load_flag']) :
        lo_temp_heat_load = run_config['constant_lo_temp_heat_load (GW)']*1.0e3 # GW -> MW
    else :
        lo_temp_heat_load = lo_temp_heat_data['IE_lo_temp_heat_demand'].values

    network.add("Load","lo-temp-heat-demand",
                bus="lo_temp_heat",
                p_set=lo_temp_heat_load)

    network.add("Link", "ASHP",
                bus0="local-elec-grid",
                bus1="lo_temp_heat",
                p_nom_extendable=True,
                p_nom_min = run_config['ASHP_min_p (GW)']*1e3, # GW -> MW
                p_nom_max = run_config['ASHP_max_p (GW)']*1e3, # GW -> MW
                efficiency= lo_temp_heat_data['IE_COP_ASHP'],
                    # NB: generally > 1.0, indicating tacit harvest of environmental heat energy
                capital_cost=assumptions.at["ASHP","fixed"])


    # ## DEFUNCT: legacy representation of harvest of
    # environmental heat energy via an explicit Generator...
    #
    # network.links['COP'] = object
    #     # Non-standard hack to create column for (possible) COP timeseries
    # network.links.at['ASHP', 'COP'] = lo_temp_heat_data['IE_COP_ASHP']

    #network.add("Generator","ASHP_RE",
    #        bus="lo_temp_heat",
    #        p_nom=np.inf # Actual dispatch coupled to ASHP link via custom constraint
    #       )

    network.add("Link", "H2_boiler",
                bus0="H2",
                bus1="lo_temp_heat",
                p_nom_extendable=True,
                p_nom_min = run_config['H2_boiler_min_p (GW)']*1e3, # GW -> MW
                p_nom_max = run_config['H2_boiler_max_p (GW)']*1e3, # GW -> MW
                efficiency = assumptions.at["H2 boiler","efficiency"],
                capital_cost = assumptions.at["H2 boiler","fixed"])

    # DAC subsystem
    network.add("Bus", "CO2_atm_bus",
                carrier="CO2")
    network.add("Store", "CO2_atm_store",
                bus="CO2_atm_bus",
                e_nom_extendable=True,
                e_nom_max = +np.inf,
                e_min_pu = -1.0, # We want to allow net CDR against inital "zero" reference
                capital_cost = 0.0, # €/tCO2
                marginal_cost = 0.0, # €/(tCO2/h)
                e_initial = 0.0) # Just track *changes* in atm CO2 stock

    network.add("Bus", "CO2_conc_bus", # Concentrated/"pure" CO2
                carrier="CO2")
    network.add("Store", "CO2_conc_store",
                bus="CO2_conc_bus",
                e_nom_extendable=True,
                e_nom_max = +np.inf,
                e_nom_min = 0.0,
                capital_cost = assumptions.at['CO2_conc_store','fixed'], # €/tCO2
                marginal_cost = 0.0, # €/(tCO2/h)
                e_initial = 0.0) # This may impose a startup artefact in CO2 *utilisation*

    # No config var limits for DAC capacity: we assume this can be freely driven to meet
    # custom constraint on total CO2 removal (if any).
    network.add("Link", "DAC",
                bus0 = "local-elec-grid", # Primary input: electricity, MW
                bus1 = "CO2_conc_bus", # Primary output, conc. CO2, tCO2/h
                bus2 = "CO2_atm_bus",
                    # Secondary input (neg efficiency!):
                    # dilute CO2 from atm, tCO2/h
                efficiency = assumptions.at['DAC','efficiency'], # tCO2/MWh
                efficiency2 = -assumptions.at['DAC','efficiency'], # Conservation of CO2 mass
                p_nom_extendable = True,
                p_nom_min = 0.0,
                    # Default, but stated explicitly to emphasise that DAC plant can't be
                    # operated in reverse (somehow generate power by releasing CO2 to atmosphere!?)
                capital_cost=assumptions.at['DAC','fixed'] # /MW input capacity (p_nom)
                )
                
    # Custom constraints:
    
    def extra_functionality(network,snapshots):
        link_p_nom = get_var(network, "Link", "p_nom")

        # Interconnector import and export links are constrained so that rated power capacity at the 
        # *input* side (p0) is equal for both directions; so max available *output* power (p1) will 
        # be less, in both directions, via the configured efficiency.
        lhs = linexpr((1.0, link_p_nom["ic-export"]),
                       (-1.0, link_p_nom["ic-import"]))
        define_constraints(network, lhs, "=", 0.0, 'Link', 'ic_ratio')

        # Battery charge and discharge links are constrained so that rated power capacity at the 
        # network/grid bus (as opposed to the store bus) is equal for both charge and discharge.
        # (The implies that the rated power on the *input* side of the *discharge* link will be
        # correspondingly higher, via the configured efficiency.)
        lhs = linexpr((1.0,link_p_nom["battery charge"]),
                      (-network.links.loc["battery discharge", "efficiency"],
                       link_p_nom["battery discharge"]))
        define_constraints(network, lhs, "=", 0.0, 'Link', 'battery_charger_ratio')

        # Atmospheric CO2 constraint
        delta_CO2_atm_max = run_config['delta_CO2_atm_max (MtCO2)']*1e6 # MtCO2 -> t
        atm_CO2_store_e = get_var(network, "Store", "e")["CO2_atm_store"].iloc[-1]
            # Scalar var: *final* value of e for CO2_atm_store
        lhs = linexpr((1.0, atm_CO2_store_e))
        define_constraints(network, lhs, "<=", delta_CO2_atm_max, 'Store', 'delta CO2 atm (max)')
        
        # ## DEFUNCT: legacy representation of harvest of
        # environmental heat energy via an explicit Generator.
        # ASHP Link and ASHP_RE Generator are coupled together so
        # that the amount of environmental heat "pumped" is
        # determined by the amount of electricity flowing into
        # the ASHP link modulo the (snapshot-specific) COP, here
        # coded via the series ashp_RE_factor.
        
        # ashp_cop = network.links.at['ASHP','COP']
        # link_p = get_var(network, "Link", "p")
        # gen_p = get_var(network, "Generator", "p")

        # lhs = linexpr(((ashp_cop - 1.0), link_p["ASHP"]),
        #               (-1.0, gen_p["ASHP_RE"]))
        # define_constraints(network, lhs, "=", 0.0, 'Link', 'ASHP RE')

      
    if solver_name == "gurobi":
        solver_options = {"threads" : 4,
                          "method" : 2,
                          "crossover" : 0,
                          "BarConvTol": 1.e-5,
                          "FeasibilityTol": 1.e-6 }
    else:
        solver_options = {}

    network.consistency_check()

    network.lopf(solver_name=run_config['solver_name'],
                  solver_options=solver_options,
                  pyomo=False,
                  extra_functionality=extra_functionality)

    return network


def gather_run_stats(run_config, network):

    # FIXME: Add a sanity check that there are no snapshots where *both* electrolysis and 
    # H2 to power (whether CCGT or OCGT) are simultaneously dispatched!? (Unless there is
    # some conceivable circumstance in which it makes sense to take power over the interconnector
    # for electrolysis??) Of course, if we adding ramping constraints the situation would be 
    # quite different...

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Blunt instrument!!
        logger.info("NB - RuntimeWarning(s) suppressed (if any)...")

        run_stats = pd.Series()

        snapshot_interval = run_config['snapshot_interval']
        total_hours = network.snapshot_weightings.sum()

        # Stash some preliminary data as extra columns in network
        # components. NB: possibly some of this might be better
        # done via override_component_attrs at time of network
        # creation...

        network.loads['e'] = (network.loads_t.p.sum() * snapshot_interval)

        network.generators['e_avail'] = (
            network.generators_t.p_max_pu.multiply(network.generators.p_nom_opt).sum() 
            * snapshot_interval)
        network.generators['e_dispatched'] = network.generators_t.p.sum() * snapshot_interval

        # network.generators_t.p_max_pu is not returned by lopf()
        # for gens with static p_max_pu (since it doesn't change
        # in time) but we want to do various generic calculations
        # for *all* generators using this so add it in for such
        # generators, if any...
        for g in network.generators.index :
            if (not(g in network.generators_t.p_max_pu.columns)) :
                network.generators_t.p_max_pu[g] = network.generators.at[g,'p_max_pu']

        # We want timeseries of link power gain/loss for potential use in power balance plots
        links_net_p = -(network.links_t.p0 + network.links_t.p1)
        network.links_t['p_gain'] = links_net_p.clip(lower=0.0)
            # +ve => Gain/tacit Generator (typically environmental heat via HP)
        network.links_t['p_loss'] = links_net_p.clip(upper=0.0)
            # -ve -> Loss/tacit Load

        # Hack to re-classify power consumed in DAC link as
        # (useful!?) "load" rather than (useless!) "loss".
        network.links_t['p_load'] = pd.DataFrame(0.0,
                                    columns=network.links_t.p0.columns,
                                    index=network.links_t.p0.index)
        network.links_t.p_load['DAC'] = network.links_t.p0['DAC']
        network.links_t.p_loss['DAC'] = 0.0
       
        network.links['e0'] = network.links_t.p0.sum() * snapshot_interval
        network.links['e1'] = network.links_t.p1.sum() * snapshot_interval
        network.links['e_gain'] = network.links_t.p_gain.sum() * snapshot_interval
        network.links['e_loss'] = network.links_t.p_loss.sum() * snapshot_interval
        network.links['e_load'] = network.links_t.p_load.sum() * snapshot_interval
        
        # Summary stats on aggregate "final" energy use, across
        # all pypsa "load" components, plus any "p_load" from "link" components
        #max_load_p = network.loads_t.p.sum(axis='columns').max()
        #mean_load_p = network.loads_t.p.sum(axis='columns').mean()
        #min_load_p = network.loads_t.p.sum(axis='columns').min()
        total_load_p = network.loads_t.p.sum(axis='columns') + network.links_t.p_load.sum(axis='columns')
        max_load_p = total_load_p.max()
        mean_load_p = total_load_p.mean()
        min_load_p = total_load_p.min()

        total_e_load = network.loads['e'].sum() + network.links['e_load'].sum()
        total_e_available = network.generators['e_avail'].sum()
        total_e_dispatched =  network.generators['e_dispatched'].sum()
        total_generated_e = total_e_dispatched + network.links['e_gain'].sum()
        total_losses_e = -network.links['e_loss'].sum()
        total_consumed_e = total_e_load + total_losses_e
        assert((total_generated_e - total_consumed_e) < 1.0) # Notional tolerance on system balance, MWh
        total_e_undispatched = total_e_available - total_e_dispatched 

        run_stats["System total load (TWh)"] = total_e_load/1.0e6
        run_stats["System mean load (GW)"] = mean_load_p/1.0e3

        run_stats["System available primary (TWh)"] = total_e_available/1.0e6
        run_stats["System dispatched (TWh)"] = total_e_dispatched/1.0e6
        run_stats["System dispatched down (TWh)"] = total_e_undispatched/1.0e6
        run_stats["System dispatched down (%)"] = (total_e_undispatched/total_e_available)*100.0

        run_stats["System generated (TWh)"] = total_generated_e/1.0e6 # Includes "link gains" (essentially HP?)
        run_stats["System losses (TWh)"] = total_losses_e/1.0e6
        run_stats["System efficiency (%)"] = (total_e_load/total_generated_e)*100.0

        for l in network.loads.index :
            total_e = network.loads.at[l,'e']
            run_stats[l+" total_e (TWh)"] = (total_e/1.0e6)
            run_stats[l+" max_p (GW)"] = network.loads_t.p[l].max()/1.0e3
            run_stats[l+" mean_p (GW)"] = (total_e/(total_hours*1.0e3))
            run_stats[l+" min_p (GW)"] = network.loads_t.p[l].min()/1.0e3

        # Special stats for DAC "load"; some duplication with generic link stats
        dac_p_nom = network.links.p_nom_opt['DAC']
        dac_total_e = network.links.loc['DAC','e_load']
        run_stats["DAC i/p capacity nom (GW)"] = (dac_p_nom/1.0e3)
        run_stats["DAC total_e (TWh)"] = dac_total_e/1.0e6
        run_stats["DAC capacity factor (%)"] = (
                dac_total_e/(dac_p_nom*total_hours))*100.0
        run_stats["DAC max_p (GW)"] = network.links_t.p_load['DAC'].max()/1.0e3
        run_stats["DAC mean_p (GW)"] = (dac_total_e/(total_hours*1.0e3))
        run_stats["DAC min_p (GW)"] = network.links_t.p_load['DAC'].min()/1.0e3

        for g in network.generators.index :
            # FIXME? Add calculation of "min" LCOE for all gens (based on 100% capacity running)
            # Note that this doesn't depend on lopf() results - it is statically determined by
            # fixed and marginal costs of each gen.
            e_avail = network.generators.at[g,'e_avail']
            e_dispatched = network.generators.at[g,'e_dispatched']
            e_undispatched = e_avail - e_dispatched
            run_stats[g+" capacity nom (GW)"] = (
                network.generators.p_nom_opt[g]/1.0e3)
            run_stats[g+" available (TWh)"] = e_avail/1.0e6
            run_stats[g+" dispatched (TWh)"] = e_dispatched/1.0e6
            run_stats[g+" penetration (%)"] = (e_dispatched/total_e_dispatched)*100.0 
            run_stats[g+" dispatched down (TWh)"] = e_undispatched/1.0e6
            run_stats[g+" dispatched down (%)"] = (e_undispatched/e_avail)*100.0
            run_stats[g+" capacity factor max (%)"] = (
                network.generators_t.p_max_pu[g].mean())*100.0
            run_stats[g+" capacity factor act (%)"] = (
                e_dispatched/(network.generators.p_nom_opt[g]*total_hours))*100.0

        links_final_conversion = ["BEV", "FCEV", "ASHP", "H2_boiler"]
        for l in links_final_conversion:
            p_nom = network.links.p_nom_opt[l]
            e0=network.links.at[l,'e0']
            e1=network.links.at[l,'e1']
            e_gain=network.links.at[l,'e_gain']
            e_loss=-network.links.at[l,'e_loss']
            #e_load=-network.links.at[l,'e_load']
            run_stats[l+" i/p capacity nom (GW)"] = (p_nom/1.0e3)
            #run_stats[l+" energy input (TWh)"] = links_e0[l]/1.0e6
            run_stats[l+" energy input (TWh)"] = e0/1.0e6
            run_stats[l+" energy output (TWh)"] = -e1/1.0e6
            run_stats[l+" energy gain (TWh)"] = e_gain/1.0e6
            run_stats[l+" energy loss (TWh)"] = e_loss/1.0e6
            run_stats[l+" capacity factor (%)"] = (
                e0/(p_nom*total_hours))*100.0
            if (e_gain > 0.0) : # Presume HP?
                #ashp_cop = network.links.at['ASHP','COP']
                #ashp_spf = (
                #    (ashp_cop * network.links_t.p1["ASHP"]).sum()
                #    / network.links_t.p1["ASHP"].sum())
                #run_stats[l+" SPF"] = ashp_spf
                spf = (-e1)/e0
                run_stats[l+" SPF"] = spf
                run_stats[l+" notional o/p capacity nom (GW)"] = ((p_nom*spf)/1.0e3)
            else :
                run_stats[l+" o/p capacity nom (GW)"] = (
                    (p_nom*network.links.efficiency[l])/1.0e3)

        run_stats["CO2_atm_store e_nom (MtCO2)"] = network.stores.e_nom_opt["CO2_atm_store"]/1.0e6
        run_stats["CO2_conc_store e_nom (MtCO2)"] = network.stores.e_nom_opt["CO2_conc_store"]/1.0e6
        
        ic_p = network.links.p_nom_opt["ic-export"]
        run_stats["IC power (GW)"] = ic_p/1.0e3
            # NB: interconnector export and import p_nom are constrained to be equal
            # (at the input side of the respective links)
        ic_total_e = network.links.at["ic-export","e0"] + (-network.links.at["ic-import","e1"]) # On IE grid side
        run_stats["IC transferred (TWh)"] = ic_total_e/1.0e6
        run_stats["IC capacity factor (%)"] = ic_total_e/(
            network.links.p_nom_opt["ic-export"]*total_hours)*100.0
        remote_elec_grid_e = network.stores.e_nom_opt["remote-elec-grid-buffer"]
        run_stats["remote-elec-grid 'store' (TWh)"] = remote_elec_grid_e/1.0e6
        remote_elec_grid_h = remote_elec_grid_e/ic_p
        run_stats["remote-elec-grid 'store' time (h)"] = remote_elec_grid_h
        run_stats["remote-elec-grid 'store' time (d)"] = remote_elec_grid_h/24.0

        # Battery "expected" to be "relatively" small so we represent stats as MW (power) or MWh (energy)
        battery_charge_p = network.links.p_nom_opt["battery charge"]
        run_stats["Battery charge/discharge capacity nom (MW)"] = battery_charge_p
            # NB: battery charge and discharge p_nom are constrained to be equal (grid side)
        battery_total_e = (network.links.at["battery charge","e0"] +
                           (-network.links.at["battery discharge","e1"])) # on grid side
        run_stats["Battery transferred (GWh)"] = battery_total_e/1.0e3
        run_stats["Battery capacity factor (%)"] = (battery_total_e/(
            network.links.p_nom_opt["battery charge"]*total_hours))*100.0
        battery_store_e = network.stores.e_nom_opt["battery storage"]
        run_stats["Battery store (MWh)"] = battery_store_e
        battery_discharge_p = network.links.p_nom_opt["battery discharge"]
        battery_store_h = battery_store_e/battery_discharge_p
        run_stats["Battery store time (h)"] = battery_store_h
        #run_stats["Battery storage time (d)"] = battery_store_h/24.0
        
        # P2H and H2P represent separate plant with separate capacity factors (0-100%); albeit, with 
        # no independent H2 load on the H2 bus, the sum of their respective capacity factors still 
        # has to be <=100% (as they will never run at the same time - that would always increase
        # system cost, as well as being just silly!)
        links_H2 = ["H2 electrolysis", "H2 OCGT", "H2 CCGT"]
        for l in links_H2:
            run_stats[l+" i/p capacity nom (GW)"] = (network.links.p_nom_opt[l]/1.0e3)
            run_stats[l+" o/p capacity nom (GW)"] = (
                (network.links.p_nom_opt[l]*network.links.efficiency[l])/1.0e3)
            run_stats[l+" capacity factor (%)"] = (
                network.links.at[l,'e0']/(network.links.p_nom_opt[l]*total_hours))*100.0

        p2h2p_total_e = (network.links.at["H2 electrolysis","e0"] +
                         (-(network.links.at["H2 OCGT","e1"]
                            + network.links.at["H2 CCGT","e1"]))) 
                        # OCGT, CCGT both on grid side (e1)
        run_stats["P2H2P transferred (TWh)"] = p2h2p_total_e/1.0e6    
        h2_store_e = network.stores.e_nom_opt["H2 store"]
        run_stats["H2 store (TWh)"] = (h2_store_e/1.0e6)
        h2_store_CCGT_p = network.links.p_nom_opt["H2 CCGT"]
        h2_store_CCGT_h = h2_store_e/h2_store_CCGT_p
        run_stats["H2 store time (CCGT, h)"] = h2_store_CCGT_h 
        run_stats["H2 store time (CCGT, d)"] = h2_store_CCGT_h/24.0
        h2_store_OCGT_p = network.links.p_nom_opt["H2 OCGT"]
        h2_store_OCGT_h = h2_store_e/h2_store_OCGT_p
        run_stats["H2 store time (OCGT, h)"] = h2_store_OCGT_h 
        run_stats["H2 store time (OCGT, d)"] = h2_store_OCGT_h/24.0

        run_stats["System total raw store I+B+H2 (TWh)"] = (
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

        run_stats["System total usable store I+B+H2 (TWh)"] = total_avail_store_gen/1.0e6
        total_avail_store_gen_h = total_avail_store_gen/mean_load_p
        run_stats["System total usable store/load (%) "] = (total_avail_store_gen/total_e_load)*100.0
        run_stats["System total usable store time (h)"] = total_avail_store_gen_h
        run_stats["System total usable store time (d)"] = total_avail_store_gen_h/24.0

        run_stats["System notional cost (B€)"] = network.objective/1.0e9 # Scale (by Nyears) to p.a.?
        run_stats["System notional LCOE (€/MWh)"] = network.objective/total_e_load

        buses = ["local-elec-grid", "H2"]
        for b in buses:
            run_stats[b+" max notional shadow price (€/MWh)"] = (
                network.buses_t.marginal_price[b].max())
            run_stats[b+" unweighted mean notional shadow price (€/MWh)"] = (
                network.buses_t.marginal_price[b].mean())
            run_stats[b+" min notional shadow price (€/MWh)"] = (
                network.buses_t.marginal_price[b].min())

        # All the following are "weighted means": shadow prices at a bus weighted by some flow to or from that bus 

        run_stats["Elec. load weighted mean notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["local-elec-grid"]*network.loads_t.p["local-elec-demand"]).sum())
                                  / network.loads_t.p["local-elec-demand"].sum())
            # This uses the original WHOBS approach, based on shadow
            # price at the local-elec-grid (ct in WHOBS) bus, BUT now
            # corrected (!?) to weight this by the load at each
            # snapshot. This weighted shadow price will, presumably,
            # be consistently higher than the "naive" (constant load)
            # shadow price in original WHOBS. Absent other
            # constraints, it should equal the system notional LCOE
            # as calculated above. But constraints may give rise to
            # localised "profit" in certain sub-systems. See
            # discussion here:
            # https://groups.google.com/g/pypsa/c/xXHmChzd8o8
            
        run_stats["Surface transport load weighted mean notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["surface_transport_final"]*network.loads_t.p["surface-transport-demand"]).sum())
                                  / network.loads_t.p["surface-transport-demand"].sum())
        
        run_stats["Offshore wind notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["local-elec-grid"]*network.generators_t.p["offshore wind"]).sum())
                                  / network.generators_t.p["offshore wind"].sum())

        run_stats["Onshore wind notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["local-elec-grid"]*network.generators_t.p["onshore wind"]).sum())
                                  / network.generators_t.p["onshore wind"].sum())

        run_stats["Solar notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["local-elec-grid"]*network.generators_t.p["solar"]).sum())
                                  / network.generators_t.p["solar"].sum())

        run_stats["Battery charge notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["local-elec-grid"]*network.links_t.p0["battery charge"]).sum())
                                  / network.links_t.p0["battery charge"].sum())

        run_stats["Battery discharge notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["local-elec-grid"]*network.links_t.p1["battery discharge"]).sum())
                                  / network.links_t.p1["battery discharge"].sum())

        run_stats["IC export notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["local-elec-grid"]*network.links_t.p0["ic-export"]).sum())
                                  / network.links_t.p0["ic-export"].sum())

        run_stats["IC import notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["local-elec-grid"]*network.links_t.p1["ic-import"]).sum())
                                  / network.links_t.p1["ic-import"].sum())

        run_stats["Elec. for DAC notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["local-elec-grid"]*network.links_t.p0["DAC"]).sum())
                                  / network.links_t.p0["DAC"].sum())

        run_stats["Atm CO2 to DAC notional shadow price (€/tCO2)"] = (
            ((network.buses_t.marginal_price["CO2_atm_bus"]*network.links_t.p2["DAC"]).sum())
                                  / network.links_t.p2["DAC"].sum())
        
        run_stats["Conc CO2 from DAC notional shadow price (€/tCO2)"] = (
            ((network.buses_t.marginal_price["CO2_conc_bus"]*network.links_t.p1["DAC"]).sum())
                                  / network.links_t.p1["DAC"].sum())

        run_stats["Elec. for H2 electrolysis notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["local-elec-grid"]*network.links_t.p0["H2 electrolysis"]).sum())
                                  / network.links_t.p0["H2 electrolysis"].sum())

        run_stats["H2 from electrolysis notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["H2"]*network.links_t.p1["H2 electrolysis"]).sum())
                                  / network.links_t.p1["H2 electrolysis"].sum())

        run_stats["H2 for CCGT notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["H2"]*network.links_t.p0["H2 CCGT"]).sum())
                                  / network.links_t.p0["H2 CCGT"].sum())

        run_stats["Elec. from H2 CCGT notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["local-elec-grid"]*network.links_t.p1["H2 CCGT"]).sum())
                                  / network.links_t.p1["H2 CCGT"].sum())

        run_stats["H2 for OCGT notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["H2"]*network.links_t.p0["H2 OCGT"]).sum())
                                  / network.links_t.p0["H2 OCGT"].sum())

        run_stats["Elec. from H2 OCGT notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["local-elec-grid"]*network.links_t.p1["H2 OCGT"]).sum())
                                  / network.links_t.p1["H2 OCGT"].sum())

        run_stats["Elec. for BEV notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["local-elec-grid"]*network.links_t.p0["BEV"]).sum())
                                  / network.links_t.p0["BEV"].sum())

        run_stats["Transport final energy from BEV notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["surface_transport_final"]*network.links_t.p1["BEV"]).sum())
                                  / network.links_t.p1["BEV"].sum())

        run_stats["H2 for FCEV notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["H2"]*network.links_t.p0["FCEV"]).sum())
                                  / network.links_t.p0["FCEV"].sum())

        run_stats["Transport final energy from FCEV notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["surface_transport_final"]*network.links_t.p1["FCEV"]).sum())
                                  / network.links_t.p1["FCEV"].sum())

        run_stats["Elec. for ASHP notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["local-elec-grid"]*network.links_t.p0["ASHP"]).sum())
                                  / network.links_t.p0["ASHP"].sum())

        run_stats["Heat final energy from ASHP notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["lo_temp_heat"]*network.links_t.p1["ASHP"]).sum())
                                  / network.links_t.p1["ASHP"].sum())

        run_stats["H2 for H2 boiler notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["H2"]*network.links_t.p0["H2_boiler"]).sum())
                                  / network.links_t.p0["H2_boiler"].sum())

        run_stats["Heat final energy from H2 boiler notional shadow price (€/MWh)"] = (
            ((network.buses_t.marginal_price["lo_temp_heat"]*network.links_t.p1["H2_boiler"]).sum())
                                  / network.links_t.p1["H2_boiler"].sum())

    return run_stats
