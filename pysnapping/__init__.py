import pyproj

WGS84_GEOD = pyproj.Geod(ellps="WGS84")
EPSG4326 = pyproj.CRS.from_epsg(4326)


class SnappingError(ValueError):
    pass


class NoSolution(ValueError):
    pass
