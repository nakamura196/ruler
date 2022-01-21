"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import os
from PIL import Image
import torch
from flask import Flask, render_template, request, redirect
from flask_cors import CORS
import requests
from io import BytesIO
import json
import urllib.parse
import warnings
from lib import detect, isHorizontal, otsu, skelton, hlsd, arg_r, check, output, resize
warnings.simplefilter('ignore')
import cv2

app = Flask(__name__)
CORS(app)

model = None

@app.route('/')
def index():
   return 'Hello World!!!'

@app.route("/predict", methods=["GET"])
def predict():
    print("start")
    if request.method == "GET":
        url = request.args.get('url', '')

    type = request.args.get('type', 'url')

    filename = "tmp.jpg"
    # !wget -O $filename $url
    
    try:
        # 超シンプルな画像保存
        r = requests.get(url)
        if r.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(r.content)
    except Exception as e:
        return {
            "success" : 0,
            "msg" : str(e)
        }

    print("image downloaded")

    org = cv2.imread("tmp.jpg")
    org_h, org_w, org_z = org.shape

    showFlag = False

    size = detect(model, showFlag)

    print("ruler detected")

    if size == 0:
        # print("定規が検出されませんでした。")
        return {
            "success" : 0,
            "msg" : "size 0"
        }
    else:
        horizontal = isHorizontal()
        otsu()

        print("otsu ended")

        # 細線化
        skelton(showFlag)

        print("skelton ended")

        # 直線の検出
        x, y = hlsd(horizontal, showFlag)

        print("さいせんか ended")

        # 極大値の表示
        arg_r_max = arg_r(x, y, False)
        # check(x, horizontal, arg_r_max, showFlag)
        value = output(x, arg_r_max, horizontal)

        print("base ended")

        if showFlag:
            from IPython.display import Image,display_jpeg
            display_jpeg(Image("output.jpg"))
        # print("1ピクセルあたりのサイズ: {} mm".format(value))
        
        ratio = value["pixelPerMM"]
        # ruler_w = int(value["width"] / ratio)
        # ruler_h = int(value["height"] / ratio)
        input_image_w = int(org_w / ratio)
        input_image_h = int(org_h / ratio)
        '''
        '''
        # print("1mmあたりのピクセル数: {} pixes".format(ratio))
        # print("検出した定規画像の実サイズ: {} mm x {} mm".format(ruler_w, ruler_h))
        # print("入力画像の実サイズ: {} mm x {} mm".format(input_image_w, input_image_h))

    result = {
        "success" : 1,
        "pixelPerMM": value["pixelPerMM"],
        "input": [input_image_w, input_image_h]
    }

    if type == "iiif":
        shp = resize(url, input_image_w, input_image_h)
        result["input_"] = result["input"]
        result["input"] = shp

    return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=80, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt') # .autoshape()  # force_reload = recache latest code
    # model.eval()
    
    # app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
    app.run(host="0.0.0.0", port=args.port, debug=True)  # debug=True causes Restarting with stat
