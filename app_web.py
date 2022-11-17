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

from base64 import b64encode

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
# check_call(['apt-get', 'update'], stdout=open(os.devnull,'wb'), stderr=STDOUT)

# check_call(['apt-get', 'install', '-y', 'python3-opencv'], stdout=open(os.devnull,'wb'), stderr=STDOUT)
github='ultralytics/yolov5'
torch.hub.list(github, trust_repo=True)
model = torch.hub.load("ultralytics/yolov5", "custom", path = "./ppe_weight.pt", force_reload=True) # no cases weight
model.classes=[6,7,8,9,10,11]


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        
        #print("FD=========",request.form.get("file"))
        #print("FILE========",request.form)
        #print("FILE========",request.get_data())
        
#         if "file" not in request.files:
#             return redirect(request.url)
#         file = request.files["file"]
#         if not file:
#             return

        #img_bytes = file.read()
        
        
        header, encoded = request.get_data().decode("utf-8"). split(",", 1)
        data = base64.b64decode(encoded)

# Python 3.4+
# from urllib import request
# with request.urlopen(data_uri) as response:
#     data = response.read()

        with open("image.png", "wb") as f:
            f.write(data)   
        #data=parse_data_uri(data)
#ParseDataURIResult(media_type='image/png', media_type_parameters={}, data=b'\x89PNG\r\n\x1a')
#         data = data.replace('data:image/png;base64,', '')

# # Convert to bytes
#         data = data.encode()

 

# # The data is encoded as base64, so we decode it.
#         data = base64.b64decode(data)

#         #img = Image.open(BytesIO(data))

        img = Image.open(io.BytesIO(data))
        results = model(img, size=640)
        img = np.squeeze(results.render())
        # datatoexcel = pd.ExcelWriter('results.xlsx')
        # results.to_excel(datatoexcel)
        # datatoexcel.save()

        print()
        print()
        print("RESULT=======",results)
        file_object = io.BytesIO()
        
        data = Image.fromarray(img)
        data.save(file_object, 'JPEG')
        base64img = "data:image/png;base64,"+b64encode(file_object.getvalue()).decode('ascii')
    #return render_template('/result.html',uri=uri)
        return jsonify(image=base64img)
    #render_template("index1.html", base64img=base64img)
        
         # updates results.imgs with boxes and labels
        # for img in results.imgs:
     
        
        #img_base64.save("static/image0.jpg", format="JPEG")
#         img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #BGR
#         frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()

#         return render_template("index1.html", img=frame)
#         return Response(gen(),
#                         mimetype='multipart/x-mixed-replace; boundary=frame')
#         #return redirect("static/image0.jpg")

    return "please upload an image"
#render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=8000, type=int, help="port number")
    args = parser.parse_args()
    
   # detect.run(weights='yolov5s.pt', save_txt= True)
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
