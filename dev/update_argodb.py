from argopy import ArgoIndex, ArgoNVSReferenceTables
from datetime import datetime, timezone
from sqlalchemy import create_engine, func, update, Index, Column, Integer, Float, TEXT, TIMESTAMP, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session, relationship
from sqlalchemy.pool import NullPool
from sqlalchemy.dialects.postgresql import insert, JSONB
from geoalchemy2 import WKTElement, Geometry
from geoalchemy2.shape import from_shape
from shapely.geometry import LineString
import argopy
import pandas as pd
import numpy as np
import geojson, os
from urllib.parse import quote 
from dotenv import load_dotenv
from contextlib import contextmanager

ocean_full_names = {
    'A': 'Atlantic Ocean Area',
    'I': 'Indian Ocean Area',
    'P': 'Pacific Ocean Area'
}

profilerRef = ArgoNVSReferenceTables().tbl('R08')
profiler_mapping = profilerRef.set_index('altLabel')['prefLabel'].to_dict()
# profiler_mapping = {
#     int(k): v
#     for k, v in profilerRef.set_index('altLabel')['prefLabel'].to_dict().items()
#     if str(k).strip().isdigit()  # Ensure the key is numeric
# }

def expand_parameters(row):
    parameters = row['parameters'].split()
    modes = list(row['parameter_data_mode'])
    rows = {
        'parameter': parameters,
        'data_mode': [modes[i] for i in range(len(parameters)) if parameters[i] in parameters]
    }
    
    for col in row.index.difference(['parameters', 'parameter_data_mode']):
        rows[col] = [row[col] for _ in parameters]
    
    return pd.DataFrame(rows)

# Convert numpy types to Python scalar types
# convert_types is applied to the entire DataFrame before the data is used, ensuring that all rows are converted from NumPy types to Python types.
# convert_types_dict is applied directly to dictionaries of data that are being prepared for individual SQL statements.
def convert_types(row):
    """Convert all numpy types in the row to native Python types."""
    for key in row.index:
        if isinstance(row[key], (np.generic, np.ndarray)):
            row[key] = row[key].item()
    return row

def convert_types_dict(data):
    """Convert all numpy types in the data dictionary to native Python types."""
    for key, value in data.items():
        if isinstance(value, (np.generic, np.ndarray)):
            data[key] = value.item()
    return data


def create_geojson_line(group):
    if len(group) < 2:
        return None
 
    points = [geojson.Point((lon, lat)) for lon, lat in zip(group['longitude'], group['latitude'])]
    properties = {
        'wmo': int(group['wmo'].iloc[0]),
        'timestamps': group['date'].apply(lambda x: x.isoformat()).tolist()
    }
    line = geojson.LineString([(point.coordinates[0], point.coordinates[1]) for point in points])
    feature = geojson.Feature(geometry=line, properties=properties)
    return geojson.dumps(feature)

print("Update started at: ", datetime.now())

current_dir = os.path.abspath(os.path.dirname("__file__"))
cache_dir = os.path.join(current_dir, '..', 'tmp_cache')
os.makedirs(cache_dir, exist_ok=True)
argopy.set_options(cachedir=cache_dir)
print("set cache: ", cache_dir)

load_dotenv()
DBUSER = os.getenv('DBUSER')
DBPASS = os.getenv('DBPASS')
DBHOST = os.getenv('DBHOST')
DBPORT = os.getenv('DBPORT')
DBNAME = os.getenv('DBNAME')
WMOTABLE = os.getenv('WMOTABLE')
ARGOTABLE = os.getenv('ARGOTABLE')
TRAJTABLE = os.getenv('TRAJTABLE')

conn_uri = f'postgresql://{DBUSER}:{quote(DBPASS)}@{DBHOST}:{DBPORT}/{DBNAME}'
engine = create_engine(
    conn_uri,
    connect_args={'connect_timeout': 10},
    pool_pre_ping=True,
    poolclass=NullPool
)

# Database Models
Base = declarative_base()

class ArgoWMO(Base):
    __tablename__ = WMOTABLE
    wmo = Column(Integer, primary_key=True)
    ocean = Column(TEXT)
    profiler = Column(TEXT)
    institution = Column(TEXT)
    # Relationships
    floats = relationship("ArgoFloat", back_populates="wmo_info")
    trajectories = relationship("ArgoTrajectory", back_populates="wmo_info")

class ArgoTrajectory(Base):
    __tablename__ = TRAJTABLE
    wmo = Column(Integer, ForeignKey(WMOTABLE + '.wmo'), primary_key=True)
    start_date = Column(TIMESTAMP(timezone=True))
    end_date = Column(TIMESTAMP(timezone=True))
    lon_max = Column(Float)
    lon_min = Column(Float)
    lat_max = Column(Float)
    lat_min = Column(Float)
    parameters = Column(TEXT)
    data_modes = Column(TEXT)
    geojson = Column(JSONB)
    geom = Column(Geometry(geometry_type='LINESTRING', srid=4326))
    # Relationship
    wmo_info = relationship("ArgoWMO", back_populates="trajectories")

class ArgoFloat(Base):
    __tablename__ = ARGOTABLE
    date = Column(TIMESTAMP(timezone=True), nullable=False, primary_key=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    ocean = Column(TEXT)
    profiler = Column(TEXT)
    #profiler_code = Column(Integer)
    institution = Column(TEXT)
    #institution_code = Column(TEXT)
    date_update = Column(TIMESTAMP(timezone=True))
    wmo = Column(Integer, ForeignKey(WMOTABLE + '.wmo'))
    cyc = Column(Integer, nullable=False, primary_key=True)
    parameter = Column(TEXT, nullable=False, primary_key=True)
    data_mode = Column(TEXT, nullable=False)
    geom = Column(Geometry(geometry_type='POINT', srid=4326))
    update_timestamp = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('argofloats_date_idx', 'date'),
        Index('argofloats_wmo_cyc_parameter_date_key', 'wmo', 'cyc', 'parameter', 'date', unique=True),
        Index('argofloats_wmo_cyc_parameter_date_idx', 'wmo', 'cyc', 'parameter', 'date'),
        Index('argofloats_date_parameter_idx', 'date', 'parameter'),
        Index('argofloats_date_data_mode_idx', 'date', 'data_mode'),
        Index('argofloats_geom_index', 'geom', postgresql_using='gist')
    )
    # Relationship
    wmo_info = relationship("ArgoWMO", back_populates="floats")

# Session Management with Context Manager
Session = scoped_session(sessionmaker(bind=engine))

@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

def update_database(session, dfo, df_w, current_timestamp):
    for index, row in dfo.iterrows():
        data = row.to_dict()
        data['geom'] = WKTElement(f'POINT({row["longitude"]} {row["latitude"]})', srid=4326)
        data['update_timestamp'] = current_timestamp
        # Convert NumPy types to Python types
        data = convert_types_dict(data)

        existing_record = session.query(ArgoFloat).filter(
            ArgoFloat.wmo == row['wmo'],
            ArgoFloat.cyc == row['cyc'],
            ArgoFloat.parameter == row['parameter'],
            ArgoFloat.date == row['date']
        ).first()

        if existing_record:
            stmt = update(ArgoFloat).where(
                ArgoFloat.wmo == row['wmo'],
                ArgoFloat.cyc == row['cyc'],
                ArgoFloat.parameter == row['parameter'],
                ArgoFloat.date == row['date']
            ).values(**data)
        else:
            stmt = insert(ArgoFloat).values(**data)
        session.execute(stmt)

    for wmo, group in df_w.groupby('wmo'):
        ocean_area = ocean_full_names.get(group['ocean'].iloc[0], '')
        existing_wmo = session.query(ArgoWMO).filter(ArgoWMO.wmo == wmo).first()

        if not existing_wmo:
            argo_wmo = ArgoWMO(
                wmo=wmo,
                ocean=ocean_area,
                profiler=group['profiler'].iloc[0],
                institution=group['institution'].iloc[0]
            )
            session.add(argo_wmo)

        geojson_string = create_geojson_line(group)
        line_geom = from_shape(LineString(list(zip(group['longitude'], group['latitude']))), srid=4326) if geojson_string else None

        existing_trajectory = session.query(ArgoTrajectory).filter(ArgoTrajectory.wmo == wmo).first()
        if existing_trajectory:
            stmt = update(ArgoTrajectory).where(
                ArgoTrajectory.wmo == wmo
            ).values(
                start_date=group['date'].min(),
                end_date=group['date'].max(),
                lon_max=float(group['longitude'].max()),
                lon_min=float(group['longitude'].min()),
                lat_max=float(group['latitude'].max()),
                lat_min=float(group['latitude'].min()),
                parameters=' '.join(set(' '.join(group['parameters']).split())),
                data_modes=' '.join(sorted(set(group['parameter_data_mode'].iloc[0]))),
                geojson=geojson_string,
                geom=line_geom
            )
            session.execute(stmt)
        else:
            argo_trajectory = ArgoTrajectory(
                wmo=wmo,
                start_date=group['date'].min(),
                end_date=group['date'].max(),
                lon_max=float(group['longitude'].max()),
                lon_min=float(group['longitude'].min()),
                lat_max=float(group['latitude'].max()),
                lat_min=float(group['latitude'].min()),
                parameters=' '.join(set(' '.join(group['parameters']).split())),
                data_modes=' '.join(sorted(set(group['parameter_data_mode'].iloc[0]))),
                geojson=geojson_string,
                geom=line_geom
            )
            session.add(argo_trajectory)

with session_scope() as session:
    # Load latest Argo data
    sidx = ArgoIndex(index_file='bgc-s').load()
    dfs = sidx.to_dataframe()

    # replace Unknown profiler with R08 reference table before argopy v1.1.0 (fix at v1.1.0)
    # dfs['profiler'] = dfs['profiler_code'].astype(str).map(profiler_mapping).fillna('Unknown') 
    print("Try to replace Unknown profiler: ",dfs[["profiler_code", "profiler"]].drop_duplicates())

    dfs = dfs.drop(columns=['file', 'profiler_code', 'institution_code', 'dac'])

    dfs = dfs.dropna(subset=['longitude', 'latitude', 'date', 'parameters', 'parameter_data_mode'])
    print("Get data from bgc-s: ", len(dfs))

    dfs_sorted = dfs.sort_values(by='date_update', ascending=False)
    dfs_unique = dfs_sorted.drop_duplicates(subset=['wmo', 'cyc', 'parameters', 'date'], keep='first')
    assert not dfs_unique.duplicated(subset=['wmo', 'cyc', 'parameters', 'date']).any(), "Duplicates remain after processing!"

    dfo = pd.concat([expand_parameters(row) for index, row in dfs_unique.iterrows()]).reset_index(drop=True)
    print(f"Date range in expanded_df: {dfo['date'].min()} to {dfo['date'].max()}")
    print("number of WMO: ", dfo['wmo'].nunique())

    #dfo = expanded_df.drop('file', axis=1)
    df_w = dfs_unique.copy() #drop('file', axis=1)
    df_w.sort_values(by='date', inplace=True)
    # Convert dataframe types before using them in the session
    df_w = df_w.apply(convert_types, axis=1)

    current_timestamp = datetime.now(timezone.utc)
    print("Current Timestamp: ", current_timestamp)

    update_database(session, dfo, df_w, current_timestamp)

    session.query(ArgoFloat).filter(ArgoFloat.update_timestamp != current_timestamp).delete(synchronize_session=False)

print("Update completed at: ", datetime.now())
