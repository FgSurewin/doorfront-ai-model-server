from flask import Flask, jsonify, request, render_template, redirect,url_for
from detect import detect
from datetime import datetime
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def index():
    return render_template("home.html", empty=True, label_list=[], time=None)


@app.route('/home_detect', methods=["POST"])
def home_detect():
    # labels = [dict(name="label1", data="label-one"), dict(name="label2", data="label-two")]
    start = datetime.now()

    
    type_dict = {1: 'door', 2:"knob", 3:"stairs"}
    nms = request.form.get("nms", None)
    url = request.form.get("url", None)
    print("home_detect: request.form -> ",request.form)
    if nms != None and url != None:
        label_list = list(detect(min(abs(float(nms)), 1), url))
        result = list(map(lambda x: dict(bbox=x[0], type=type_dict.get(x[1], "Error"), score=x[2]), label_list ))
        end = datetime.now()
        difference = (end - start).total_seconds()
        return render_template("home.html", empty=len(result) == 0, label_list=result, time=difference)
    else:
        return redirect(url_for("index"))
    


@app.route('/model/detect', methods=["POST"])
def detect_route():
    type_dict = {1: 'door', 2:"knob", 3:"stairs"}
    nms = request.form.get("nms", None)
    url = request.form.get("url", None)
    print("detect: request.form -> ",request.form)
    if url != None:
        label_list = list(detect(min(abs(float(nms)), 1), url))
        result = list(map(lambda x: dict(bbox=x[0], type=type_dict.get(x[1], "Error"), score=x[2]), label_list ))
        response = jsonify(result)

         # Enable Access-Control-Allow-Origin
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response
    else:
        return jsonify(code=404, msg="Please input correct parameters!")


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0' , port=int(os.environ.get("PORT", 8080)))

