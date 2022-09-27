from enum import Enum

import pyproj


EPSG4326 = pyproj.CRS.from_epsg(4326)

# Earth centered, earth fixed right handed metric 3d cartesian coordinate system
# based on WGS84 ellipsoid.
#
# Good for getting cartesian metric distances when distances are not too large
# (say a few hundred kilometers) and for projecting points to linestrings without distortion.
# In our case, typical distances refer to segment lengths of linestrings
# (NOT total linestring lengths) and distances of points to linestrings.
# So this is perfectly fine for public transport trajectories and stations anywhere on earth
# as long as the discretization of linestrings is reasonably fine and stations
# are reasonably close by.
# Maybe not suitable for aircraft trajectories without any intermediate vertices
# or overly long beelines. But we can live with that since it makes things so much easier when
# dealing with meter-based parameters. And more vertices can be easily added when needed.
EPSG4978 = pyproj.CRS.from_epsg(4978)


class SnappingError(ValueError):
    pass


class NoSolution(ValueError):
    pass


class ExtrapolationError(ValueError):
    pass


class SnappingMethod(Enum):
    trusted = "trusted"
    projected = "projected"
    iterative = "iterative"
    fallback = "fallback"
