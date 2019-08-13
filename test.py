# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""
import plotly
import pandas as pd

from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import plotly.express as px
import matplotlib.pyplot as py

database_path = 'data/DisasterResponse.db'
engine = create_engine('sqlite:///'+ database_path)
database_name = database_path.split('/')[-1].split('.')[0] # split given path to get just the name
df = pd.read_sql_table(database_name, engine)

genre_counts = df.groupby('genre').count()['message']
genre_names = list(genre_counts.index)
cat_col = list(df.columns)
del cat_col[:4]  # category column names
cat_col_genre = df.groupby('genre').mean()[a]

py.bar(list(cat_col_genre.columns),cat_col_genre.iloc[0])