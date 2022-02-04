# SWIS-100-IE
## Notionally Optimal, 100% Solar+Wind+Interconnection+Storage Electricity System for Ireland (IE)

- **Research Project:**
  [OESM-IE](http://ecrn.eeng.dcu.ie/projects/oesm-ie)
- **Funding:** [Sustainable Energy Authority of Ireland (SEAI)
  Research, Development and Demonstration
  Programme](https://www.seai.ie/grants/research-funding/research-development-and-demonstration-fund/),
  award reference SEAI RDD/00246 2018.
- **Contributors:** 
    - Barry McMullin barry.mcmullin@dcu.ie
	- James Carton james.carton@dcu.ie
	- Iftekhar Hussain cmiftekhar.h@gmail.com
- **Release:** Pre-release (initial development)
- **© 2021:** [Dublin City University](http://www.dcu.ie/) (unless indicated otherwise)
- **Licence:** [GNU GENERAL PUBLIC LICENSE Version 3](https://www.gnu.org/licenses/gpl-3.0.en.html)

This package is derived from: [Optimal
Wind+Hydrogen+Other+Battery+Solar (WHOBS) electricity systems for
European countries](https://github.com/PyPSA/WHOBS)


## Motivation

The pre-existing [WHOBS](https://github.com/PyPSA/WHOBS) model,
and its associated interactive web application
[model.energy](https://model.energy/), allows modelling of
notionally optimised "firm" electricity generation for a given
level of (constant/"baseload") capacity, based exclusively on
*variable* renewable (VRE: here limited to wind and solar)
sources, coupled with hydrogen and/or battery storage (to cover
"when the wind doesn't blow and the sun doesn't shine"). It is
built using the [PyPSA](https://github.com/PyPSA/PyPSA) (Python
for Power System Analysis) modelling toolbox.  Wind and solar
resource variability is configured for states within Europe using
historical data from
[Renewables.ninja](https://www.renewables.ninja/)
([model.energy](https://model.energy/) extends this coverage to
global geographical locations using other data sources).

**SWIS-100-IE** adapts WHOBS to model delivery of **100%** of
electricity demand from VRE only for the synchronous grid serving
the island of Ireland (incorporating the **Republic of
Ireland** and **Northern Ireland**, a devolved nation of the
United Kingdom), based on [historical load
data](http://www.eirgridgroup.com/how-the-grid-works/renewables/)
from the [Irish Transmission System Operator (TSO)
eirgrid](http://www.eirgridgroup.com/).

This allows illustration and exploration of the (rough) trade-off between:

- Raw VRE *overprovision* (building more VRE capacity than can be
  directly dispatched at all times, but meaning that more demand
  can be covered directly by instantaneous VRE generations)
- Dispatch down (discarding some generation when it is in excess
  of instantaneous load)
- Storage (storing some generation when it is in excess of
  instantaneous load)
    - Short term, high efficiency storage: battery
    - Long term, low efficiency storage: hydrogen
- Interconnection with a larger, external, grid: here modelled
  (simplistically) as a single, aggregated, fixed power capacity
  link to an indefinitely large external energy storage
  (effectively it is assumed that a larger external grid will
  facilitate temporal buffering from the point of view of the
  local system, in a manner analogous to local storage,
  constrained only by the power capacity and efficiency of the
  interconnector(s) themselves).

## Enhancements over WHOBS?

+ Added (crude) representation of external interconnection
+ Added more flexible options on temporal resolution - not just 1
  or 3 hours, but arbitrary number of hours
+ Added/refactored mechanism for flexibly setting options for a
  particular model run, capturing the summary results, and
  accumulating the information on these runs in two data
  structures, `run_configs` and `ru_stats`.
+ Instead of H2 underground vs steel tank storage being mutually
  exclusive, both are made available for deployment
  unconditionally, but user can (optionally) set
  `e_extendable_max` limits on each separately (which could
  potentially be based, even if loosely, on actual available
  geology, such as salt caverns in NI).
+ Similarly, H2 overall is available for deployment
  unconditionally: but obviously if both storage options are set
  to zero capacity, no actual H2 electrolysis or electricity
  generation will actually be provisioned.
+ Added H2-OCGT as an alternative to H2-CCGT

# Licence

Unless indicated otherwise, all code in this repository is
copyright ©2020 by [Dublin City University](http://www.dcu.ie/).

This program is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either [version 3 of
the License](LICENSE.txt), or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
[GNU General Public License](LICENSE.txt) for more details.




