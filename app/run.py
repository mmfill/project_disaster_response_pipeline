import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
#import plotly.graph_objects as go
from sklearn.externals import joblib
from sqlalchemy import create_engine
import plotly.express as px


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
#engine = create_engine('sqlite:///../data/DisasterResponse.db')
database_path = 'data/DisasterResponse.db'
engine = create_engine('sqlite:///'+ database_path)
database_name = database_path.split('/')[-1].split('.')[0] # split given path to get just the name
df = pd.read_sql_table(database_name, engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    cat_col = list(df.columns)
    del cat_col[:4]  # category column names
    cat_col_genre = df.groupby('genre').mean()[cat_col]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
			'data': [
				Bar(
					x=list(cat_col_genre.columns),
					y=cat_col_genre.iloc[0],
					name='direct'
				),
				Bar(
					x=list(cat_col_genre.columns),
					y=cat_col_genre.iloc[1],
					name='news'
				),
				Bar(
					x=list(cat_col_genre.columns),
					y=cat_col_genre.iloc[2],
					name='social'
				),
			],

            'layout': {
                'title': 'Distribution of Categories by genres',
                'yaxis': {
                    'title': "Mean"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
			'data': [
				Pie(labels=genre_names, values=genre_counts,hole=.3)
			],

            'layout': {
                'title': 'Distribution of Messages'
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
