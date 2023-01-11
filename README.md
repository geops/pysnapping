# PySnapping

Snap points to a line string keeping a given order or spacing intact.

The PySnapping python library helps to solve the problem of snapping an ordered sequence of points
to a line string, respecting the given input order.

The motivation for this library is to solve the problem of snapping stops onto vehicle trajectories
in the [GTFS format](https://gtfs.org/) with inaccurate or missing kilometrage.

In order to be able to work with metric parameters and to treat data with large extent located
anywhere on the world, we use the [EPSG:4978](https://epsg.io/4978) Cartesian 3D geocentric coordinate system
for the internal representation.

The library aims to automatically classify kilometrage information as trusted or untrusted and fills missing
data and corrects bogus data by interpolation/extrapolation if possible.
Trusted points are always snapped as given by the kilometrage.
In between trusted points, an approximate iterative scheme is used to snap all untrusted points such that
the order or minimum spacing conditions are not violated.

## Installation

Install the latest stable release from PyPI with pip:

```bash
pip install pysnapping
```

## Modules

### Linear Referencing

The `pysnapping.linear_referencing` module contains low-level classes and functions for referencing points along
linear features in N-dimensions built on top of `numpy`. This contains interpolation/extrapolation along a line string
and data structures optimized for projecting points to substrings of a line string.

### Ordering

The `pysnapping.ordering` module contains functions to check and fix the order of a sequence of floats,
respecting certain boundary conditions and spacings using the method of quadratic programming with the `cvxpy` library.

### Utils

The `pysnapping.util` module contains common helper functions used in other parts of the library.

### Snapping

The `pysanpping.snap` module is the main entry point for users of the `pysnapping` library and
provides the classes needed to use the library.

## Usage

The typical usage pattern is to create a `pysnapping.snap.DubiousTrajectory` instance which represents
a vehicle trajectory with dubious kilometrage and a `pysnapping.snap.DubiousTrajectoryTrip` `dtrip` which represents
a trip along such a trajectory with dubious kilometrage for the stops and dubious timing information.
Then `dtrip.to_trajectory_trip` can be used to get a `pysnapping.snap.TrajectorTrip` `trip` with well defined
metric kilometrage. `trip.trajectory` then refers to a `pysnapping.snap.Trajectory` instance with well defined metric
kilometrage. The conversion can be controlled with parameters given by a `pysnapping.snap.SnappingParams` instance.

The `trip.snap_trip_points` method can be used to snap the trip points onto the trajectory, resulting in
a `pysnapping.snap.SnappedTripPoints` instance `snapped`. Then you can e.g. split the trajectory at the stops
using the `snapped.get_inter_point_ls_coords_in_travel_direction` method.

A more detailed usage example (e.g. how to process GTFS input) is planned but not available yet.
Until then, please also check the docstrings and source code for additional usage hints/possibilities.

## Issue Tracker

Please use [the GitHub issue tracker](https://github.com/geops/pysnapping/issues) to report bugs/issues.

## Development

### Contributing

If you want to contribute to the pysnapping library, you can make a pull request at [GitHub](https://github.com/geops/pysnapping).
Before working on major features/changes, please consider contacting us about your plans.
See [our GitHub page](https://github.com/geops) for contact details.

### Editable Installation

Clone this repo and enter the corresponding directory.
Create a virtual environment, then install frozen requirements, dev-requirements
and this library in editable mode:

```bash
python3.7 -m venv env
. env/bin/activate
pip install -U pip
pip install -r requirements.txt -r dev-requirements.txt -e .
```

Keep env activated for all following instructions.

### Pre-Commit Hooks

Enable pre-commit hooks:

```bash
pre-commit install
```

From time to time (not automated yet) run

```bash
pre-commit autoupdate
```

to update frozen revs.

### Run tests

Run tests and analyze code coverage:

```bash
pytest --cov=pysnapping --cov-report term --cov-fail-under=85 pysnapping
```
