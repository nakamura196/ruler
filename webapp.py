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
import shutil
import hashlib

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
        base = request.args.get('url', '')
        url = base

    type = request.args.get('type', 'url')

    partial = False

    x0 = 0
    y0 = 0
    ratio = 1 # 入力画像のサイズとダウンロード画像のサイズ（1024）との比率
    api = None

    if type == "iiif":
        spl = url.split("/")
        ln = spl[-4] + "/" + spl[-3] + "/" + spl[-2] + "/" + spl[-1]
        api = url.replace(ln, "")
        info = api + "info.json"

        try:
            df = requests.get(info).json()
        except Exception as e:
            return {
                "success" : 0,
                "msg" : str(e)
            }
        
        full_image_width = df["width"]
        full_image_height = df["height"]

        base_xywh = spl[-4]

        isWidthLarge = True

        bsize = 1024

        if base_xywh != "full":
            partial = True

            base_spl = base_xywh.split(",")
            x0 = int(base_spl[0])
            y0 = int(base_spl[1])
            base_w = int(base_spl[2])
            base_h = int(base_spl[3])

            if base_w < base_h:
                isWidthLarge = False

            ratio = base_w / bsize if isWidthLarge else base_h / bsize
        else:
            if full_image_width < full_image_height:
                isWidthLarge = False
            ratio = full_image_width / bsize if isWidthLarge else full_image_height / bsize
        url = api + base_xywh + "/" + ("{},".format(bsize) if isWidthLarge else ",{}".format(bsize)) + "/0/default.jpg"

    filename = "tmp.jpg"
    # !wget -O $filename $url
    
    try:
        # 超シンプルな画像保存
        print("url", url)
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

    org_h = org_h * ratio
    org_w = org_w * ratio

    showFlag = False

    size, xywh = detect(model, ratio, type, api, partial, x0, y0, showFlag)

    print("ruler detected")

    if size == 0:
        # print("定規が検出されませんでした。")
        return {
            "success" : 1,
            "msg" : "No rulers detected."
        }
    else:
        horizontal = isHorizontal()
        otsu()

        # print("otsu ended")

        # 細線化
        skelton(showFlag)

        # print("skelton ended")

        # 直線の検出
        x, y = hlsd(horizontal, showFlag)

        # print("さいせんか ended")

        # 極大値の表示
        arg_r_max = arg_r(x, y, False)
        # check(x, horizontal, arg_r_max, showFlag)
        value = output(x, arg_r_max, horizontal)

        # print("base ended")

        if showFlag:
            from IPython.display import Image,display_jpeg
            display_jpeg(Image("output.jpg"))
        # print("1ピクセルあたりのサイズ: {} mm".format(value))
        
        ratio = value["pixelPerMM"]
        ruler_w = int(value["width"] / ratio)
        ruler_h = int(value["height"] / ratio)
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
        "ruler" : [ruler_w, ruler_h],
        "image": [input_image_w, input_image_h]
    }

    '''
    if type == "iiif":
        shp = resize(url, input_image_w, input_image_h)
        result["input_"] = result["input"]
        result["input"] = shp
    '''

    partial = request.args.get('partial', '0')
    if partial == "1":
        shp = resize(url, input_image_w, input_image_h)
        # result["input"] = result["input"]
        result["full"] = shp

    hs = hashlib.md5(base.encode()).hexdigest()

    static_path = "static/{}.jpg".format(hs)

    os.makedirs(os.path.dirname(static_path), exist_ok=True)

    shutil.copy("output.jpg", static_path)

    result["hash"] = hs

    # result["r"] = ratio

    result["xywh"] = xywh

    return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=80, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt') # .autoshape()  # force_reload = recache latest code
    # model.eval()
    
    # app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
    app.run(host="0.0.0.0", port=args.port, debug=True)  # debug=True causes Restarting with stat
