from flask import Flask, request, jsonify
import json
import sqlite3
import nn, save_ann

app = Flask(__name__)


# for doing the main work; running the input through the ann
@app.route('/process', methods=['GET', 'POST'])
def process():
    if (request.method == 'GET'):
        return jsonify({'result': "it works!"})
    
    data = request.json
    img = data.get('input_img')
    
    mlp = nn.MLP(nn.MLP_LAYOUT)
    save_ann.load_biases(mlp)
    save_ann.load_weights(mlp)
    mlp.forward(img)
    result = mlp.result()
    
    print("got the image!!!")
    return jsonify({ 'result': result })


# for saving the user input to sqlite database
@app.route('/save', methods=['POST'])
def save():
    data = request.json
    img = data.get('input_img')
    result = data.get('mlp_result')
    
    conn = get_db_connection()
    try:
        conn.execute('INSERT INTO sample_data_2 (img_arr, mlp_result) VALUES(?, ?)', (json.dumps(img), result))
        conn.commit()
        conn.close()
        return jsonify({'message': 'data sample saved!'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'message': 'data not saved.'}), 400
    

# sets up sqlite db connection
def get_db_connection():
    conn = sqlite3.connect('mydata.db')
    return conn
    

if __name__ == '__main__':
    #app.run(debug=True, host='0.0.0.0', port=PORT)
    app.run(debug=True)