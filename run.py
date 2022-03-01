
import json
from flask import Flask, current_app, jsonify, send_file, request
from flask_cors import CORS
import numpy as np
from gau import evaluate
from nst import nst_apply
from pipo import convert


app = Flask(__name__, static_url_path='')
CORS(app)


@app.route('/generate', methods = ['POST'])
def generate():
    labelmap = np.asarray(request.json)
    image = evaluate(labelmap)
    return {'url':image}

@app.route('/')
def index():
    return current_app.send_static_file('index.html')

@app.route('/style', methods = ['POST'])
def style():
    url = request.form['url']
    print(url)
    trans_style = nst_apply(url)
    return {'result':trans_style}

@app.route('/pipo', methods = ['POST'])
def pipo():
    type = request.headers['Content-Type']
    if type == 'application/x-www-form-urlencoded; charset=UTF-8':
        job = 'sketch'
        url = request.form.get('url')
    elif type == 'application/json':
        job = 'pipo'
        url = request.json['url']

    result, img = convert(job, url)
    if result == 'sketch':
        return jsonify(blur = img)
    elif result == 'pipo':
        return jsonify(img=img[0], label_img=img[1])
    
    return jsonify(msg='error')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
