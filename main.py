
import pandas as pd
import pickle
import json
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder

from flask import Flask, jsonify, request
app = Flask(__name__)
cors = CORS(app)

with open('tree.pkl', 'rb') as tree_file:
    model = pickle.load(tree_file)

with open('encoder.pkl', 'rb') as encoder_file:
    LE = pickle.load(encoder_file)


@app.route('/')
def index():
    return '<h1>Arbol de prediccion</h1>'


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    print('JSON---------')
    print(request.get_json())
    data = request.get_json()
    print('proceso iniciado')

    datos = pd.DataFrame(data['datos'])

    ID = datos['ID']
    datos = datos.drop(columns=['ID'])

    columns_object = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

    for column in columns_object:
        datos[column] = LE.transform(datos[column])


    y_predict = model.predict(datos)


    y_predict = LE.inverse_transform(y_predict)

    list_of_tuples = list(zip(ID, y_predict))

    # response = { item[0] : item[1] for item in list_of_tuples }

    response =  {"datos": [ { "status" : item[1], "id": item[0]  } for item in list_of_tuples] }

    print(response)
    # json.dumps(response)
    return jsonify(response)


@app.route('/prediction2')
def prediction2():
    with open('datos1.json') as json_file:
        dicc = json.load(json_file)

    datos = pd.DataFrame(dicc['datos'])


    ID = datos['ID']
    datos = datos.drop(columns=['ID'])

    columns_object = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

    for column in columns_object:
        datos[column] = LE.transform(datos[column])


    y_predict = model.predict(datos)


    y_predict = LE.inverse_transform(y_predict)

    list_of_tuples = list(zip(ID, y_predict))

    response = { item[0] : item[1] for item in list_of_tuples }

    print(response)
    # json.dumps(response)
    return jsonify(response)




if __name__ == '__main__':
    app.run() # debug = True  # host='0.0.0.0', port=80








