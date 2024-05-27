from shapely.geometry import LineString, MultiLineString
import geojson
from shapely.geometry import LineString
from geoalchemy2.shape import from_shape
import pandas as pd

# We found antimeridian problem when trajectory cross 180-degree line and cause
# geosever display trajectories error by using geom. But I think I should leave
# this problme to client app, which can retrieve trajectory geojson from web API
# and don't use Geoserver WMS to get trajectory. The following code is not tested at all.

def split_line_at_antimeridian(line):
    """Split a LineString at the antimeridian and return a MultiLineString."""
    coords = list(line.coords)
    parts = []
    current_part = []
    
    def needs_split(lon1, lon2):
        return abs(lon1 - lon2) > 180
    
    for i in range(1, len(coords)):
        current_part.append(coords[i - 1])
        if needs_split(coords[i - 1][0], coords[i][0]):
            current_part.append(coords[i])
            parts.append(LineString(current_part))
            current_part = [coords[i]]
    current_part.append(coords[-1])
    parts.append(LineString(current_part))
    
    return MultiLineString(parts)

def create_geojson_feature(multilinestring):
    """Create a GeoJSON feature from a MultiLineString."""
    # Convert MultiLineString to GeoJSON format
    geometry = geojson.MultiLineString([list(map(list, line.coords)) for line in multilinestring.geoms])
    feature = geojson.Feature(geometry=geometry)
    return geojson.dumps(feature)


# Example data processing function
def process_data(group):
    line = LineString(list(zip(group['longitude'], group['latitude'])))
    split_multilinestring = split_line_at_antimeridian(line)
    
    # Create GeoJSON and geom from the split MultiLineString
    geojson_string = create_geojson_feature(split_multilinestring)
    geom = from_shape(split_multilinestring, srid=4326)  # Ensure SRID matches your CRS
    
    # Create and return the data object (example)
    return {
        'wmo': group['wmo'].iloc[0],
        'geojson': geojson_string,
        'geom': geom,
        # Add other necessary properties
    }
    
""" ---- should test in argo_writedb01.ipynb ----

# Assuming df is your DataFrame and session is your SQLAlchemy session
for wmo, group in df.groupby('wmo'):
    processed_data = process_data(group)
    # Create a new trajectory entry or update an existing one
    trajectory = ArgoTrajectory(
        wmo=wmo,
        geojson=processed_data['geojson'],
        geom=processed_data['geom'],
        # Set other fields...
    )
    session.add(trajectory)

session.commit()
session.close()
"""