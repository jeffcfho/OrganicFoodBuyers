from flask import Flask, render_template, request
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
import requests

app = Flask(__name__)
api = Api(app)

filename = 'saved_models/model_3predictors.pkl'
model = pickle.load(open(filename, 'rb'))

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('dept_alcohol')
parser.add_argument('days_since_prior_order')
parser.add_argument('dept_bulk')

@app.route('/', methods=['GET', 'POST'])
def PredictLikelihood():
    if request.method == 'POST':
        # use parser and find the user's query
        args = parser.parse_args()
        days_since_prev = float(args['days_since_prior_order'])
        prev_dept_alcohol = float(args['dept_alcohol'])
        prev_dept_bulk = float(args['dept_bulk'])

        #order matters! check original model for schema
        input_vector = np.array([days_since_prev,prev_dept_alcohol,prev_dept_bulk]).reshape(1,-1)
        
        # make a prediction from model
    #         prediction = model.predict(input_vector)
        pred_proba = model.predict_proba(input_vector)

        # round the predict proba value and set to new variable
        prob_of_organic = round(pred_proba[0][1], 3)

        # create JSON object
        # output = {'prob_next_organic': float(prob_of_organic)}
        
        return render_template('index.html',prob_next_organic=prob_of_organic,
                                days_since_prior_order=days_since_prev,
                                dept_alcohol=prev_dept_alcohol,
                                dept_bulk=prev_dept_bulk)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)