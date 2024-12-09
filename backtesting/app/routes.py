from flask import Blueprint, render_template, jsonify
import pandas as pd
import os
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

# Create a blueprint for routes
routes_blueprint = Blueprint('routes', __name__)

# Route for serving the main visualization page
@routes_blueprint.route('/')
def index():
    return render_template('index.html')

# Route for serving JSON data
# @routes_blueprint.route('/data/<filename>')
# def get_data(filename):
#     file_path = os.path.join('data', filename)
#     try:
#         df = pd.read_csv(file_path)
#
#         df['scaled_predictions'] = df['prediction'] / (0.1 / df['current_price'])
#
#         # Map the normalized values to colors using the colormap
#         cmap = cm.get_cmap('RdYlGn')
#         colors = [cmap(value) for value in df['prediction']]
#         colors = [f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {a})' for r, g, b, a in colors]
#
#         # Add the colors to the dataframe
#         df['color'] = colors
#
#         return jsonify(df.to_dict(orient='records'))
#     except Exception as e:
#         return jsonify({'error': str(e)})

@routes_blueprint.route('/data/<filename>')
def get_data(filename):
    file_path = os.path.join('data', filename)
    try:
        df = pd.read_csv(file_path)

        conf_bandwidth = df['current_price'].max() * 0.05
        df['upper_bound'] = df['current_price'] - conf_bandwidth
        df['lower_bound'] = df['current_price'] - conf_bandwidth

        # Map the normalized values to colors using the colormap
        cmap = cm.get_cmap('RdYlGn')
        colors = [cmap(value) for value in df['prediction']]
        colors = [f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})' for r, g, b, a in colors]


        # Add the colors to the dataframe
        df['color'] = colors

        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)})

@routes_blueprint.route('/files')
def list_files():
    try:
        # List CSV files in the /data directory
        files = [f for f in os.listdir('data') if f.endswith('.csv')]
        return jsonify(files)
    except Exception as e:
        return jsonify({'error': str(e)})