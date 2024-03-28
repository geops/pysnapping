import typing
from enum import Enum

import pyproj
import numpy as np

if typing.TYPE_CHECKING:
    from numpy.typing import ArrayLike


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


class BadShortestDistances(SnappingError):
    bad_indices: np.ndarray
    bad_distances: np.ndarray
    max_allowed_distance: float

    def __init__(
        self,
        bad_indices: "ArrayLike",
        distances: "ArrayLike",
        max_allowed_distance: float,
    ) -> None:
        self.bad_indices = np.array(bad_indices, dtype=int)
        self.bad_distances = np.asarray(distances, dtype=float)[self.bad_indices]
        self.max_allowed_distance = max_allowed_distance
        message = (
            f"The shortest distances {self.bad_distances} between points at indices "
            f"{self.bad_indices} and the trajectory are greater than "
            f"{max_allowed_distance:g} meters."
        )
        super().__init__(message)


class ExtrapolationError(ValueError):
    pass


class SnappingMethod(Enum):
    trusted = "trusted"
    routed = "routed"
    fallback = "fallback"
