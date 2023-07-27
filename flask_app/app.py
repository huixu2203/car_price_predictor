

from flask import Flask, render_template, request
import pickle

import pandas as pd
import mysql.connector as sql


from sklearn.base import BaseEstimator
from xgboost import XGBRegressor


connection =  sql.connect(
 host="localhost",
 user ="*****",
 password= "*****",
 database = "database"
 )

cursor = connection.cursor(prepared=True)


# Define a custom ensemble model that clusters the data based on the mm column and trains an XGBoost model for each cluster

class MmEnsembleModel(BaseEstimator):
    def __init__(self):
        self.models = {}

    def fit(self, X, y):
        # Train an XGBoost model for each mm
        for mm in X['mm'].unique():
            # Filter the training data for the current mm
            X_train_mm = X.loc[X['mm'] == mm, ['year', 'listing_mileage']]

            y_train_mm = y[X['mm'] == mm]

            # Create and train the XGBoost model
            model = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100)
            model.fit(X_train_mm, y_train_mm)

            # Store the trained model in the dictionary
            self.models[mm] = model

        return self

    def predict(self, X):
        # Make predictions using the trained models
        y_pred = []
        for i, row in X.iterrows():
            # Get the mm for the current row
            mm = row['mm']

            # Use the corresponding model to make a prediction
            pred = self.models[mm].predict(row[['year', 'listing_mileage']].values.reshape(1, -1))

            # Append the prediction to the list of predictions
            y_pred.append(pred[0])

        return y_pred


def get_similar_rows(year, mm, cluster_value):
    # Create a cursor
    cursor = connection.cursor(prepared=True)
    n = 100
    if cluster_value == 2618:
        query = "SELECT * FROM data2 WHERE mm = %s AND cluster = %s"
        cursor.execute(query, (mm, cluster_value))
        results = cursor.fetchall()

        # Create a DataFrame from the query results
        df = pd.DataFrame(results, columns=[i[0] for i in cursor.description])
        print(f"All data for mm '{mm}' with cluster value '{cluster_value}':")
        print(df)
    else:
        # Query to get rows with the same mm and years in the specified range
        min_year = 1980
        max_year = 2025
        query = """
            SELECT year, mm, listing_mileage, listing_price FROM data2
            WHERE mm = %s AND year BETWEEN %s AND %s
            ORDER BY ABS(year - %s)
            LIMIT %s
        """
        cursor.execute(query, (mm, min_year, max_year, year, n))
        results = cursor.fetchall()

        # Create a DataFrame from the query results
        df = pd.DataFrame(results, columns=[i[0] for i in cursor.description])

        df['listing_mileage'] = df['listing_mileage'].astype(int)
        df['listing_price'] = df['listing_price'].astype(int)

        df['listing_mileage'] = df['listing_mileage'].astype(str)
        df['listing_price'] = df['listing_price'].astype(str)

        df['listing_mileage'] =  df['listing_mileage'] + ' miles'
        df['listing_price'] = '$' + df['listing_price']
        return df

    # Close the cursor
    cursor.close()


model_non_rare = pickle.load(open('pipe_non_rare.pkl','rb'))
model_rare = pickle.load(open('pipe_rare.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict_price():
    year = (request.form.get('year'))
    make_model = (request.form.get('make_model'))
    mileage = (request.form.get('mileage'))

    #prediction


    # Create a dictionary with the column names as keys and the variable values as values
    if mileage:
        data = {'year': year, 'listing_mileage': mileage, 'mm': make_model}
        index = ['Row 1']
    else :
        data = {'year': year, 'listing_mileage': 0, 'mm': make_model}
        index = ['Row 1']

    dft = pd.DataFrame(data, index=index)
    mm_value = make_model
    year_value = year
    # query to get the cluster

    query = "SELECT cluster FROM data2 WHERE mm = %s"
    cursor.execute(query, (mm_value,))

    result = cursor.fetchall()

    cluster_value = result[0][0]

    if cluster_value == 2618:

        prd_price = model_rare.predict(dft)
    else:
        prd_price = model_non_rare.predict(dft)



    result = 'Estimated price is: $' + str(int(prd_price[0]))

    dt =get_similar_rows(year_value, mm_value,cluster_value)
    table_html = dt.to_html(classes='mytable')
    # Concatenate the response string and the HTML table into a single HTML response
    # response_html = f'{result}<br><br>{table_html}'
    return render_template('index.html', result=result, table=table_html)



if __name__ == '__main__':
    app.run(debug = True)
