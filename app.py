"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import os
from PIL import Image
from flask_cors import CORS
import numpy as np
import cv2
from base64 import b64encode
import json
import base64
import torch
from flask import Flask, render_template, request, redirect,jsonify
from subprocess import STDOUT, check_call
#from w3lib.url import parse_data_uri

import os
app = Flask(__name__)
CORS(app)
check_call(['apt-get', 'update'], stdout=open(os.devnull,'wb'), stderr=STDOUT)


check_call(['apt-get', 'install', '-y', 'libgl1'], stdout=open(os.devnull,'wb'), stderr=STDOUT)

check_call(['apt-get', 'install', '-y', 'libglib2.0-0'], stdout=open(os.devnull,'wb'), stderr=STDOUT)
check_call(['apt-get', 'update'], stdout=open(os.devnull,'wb'), stderr=STDOUT)

check_call(['apt-get', 'install', '-y', 'python3-opencv'], stdout=open(os.devnull,'wb'), stderr=STDOUT)
github='ultralytics/yolov5'
torch.hub.list(github, trust_repo=True)
model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True) # no cases weight
cam_url="region.mp4"
model.classes=[0]
cap = cv2.VideoCapture(cam_url)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

@app.route("/", methods=["GET", "POST"])
def predict():
    # if request.method == "POST":
    while cap.isOpened():
        
        _, frame = cap.read()
        results = model(frame, size=640)
        tensor = results.xyxy[0]
        img = np.squeeze(results.render())
   
        if len(tensor)>0:
            print("Alert")
            cv2.putText(img,'Unauthorized Entry Detected:',(100,100), font, 1,(0,255,255),2,cv2.LINE_4)
            data = Image.fromarray(img)
            file_object = io.BytesIO()
            data.save(file_object, 'JPEG')
            base64img = "data:image/png;base64,"+b64encode(file_object.getvalue()).decode('ascii')
            defect_df=  {"Camera":"Camera1", "Alert":"Unauthorized Entry Detected","Location":"OWS", "image":base64img}
            defect_df = json.loads(json.dumps(defect_df))
            print()
            # self.thread_lock.release()

            return jsonify(defect_df)

    return "please upload an image"
#render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=8000, type=int, help="port number")
    args = parser.parse_args()
    
   # detect.run(weights='yolov5s.pt', save_txt= True)
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
