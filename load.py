from IPython.display import Image,display_jpeg
import cv2
import requests

import argparse    # 1. argparseをインポート

parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')    # 2. パーサを作る

# 3. parser.add_argumentで受け取る引数を追加していく
parser.add_argument('url', help='この引数の説明（なくてもよい）')
parser.add_argument('-t', '--type', default="url", help='この引数の説明（なくてもよい）')
parser.add_argument('-s', '--show', default=False, type=bool, help='この引数の説明（なくてもよい）')


args = parser.parse_args()    # 4. 引数を解析

url = args.url
type = args.type
showFlag = args.show

####

'''

key = "kyoto"

####

collection = {
    "sat": {
        "url" : "https://candra.dhii.jp/iipsrv/iipsrv.fcgi?IIIF=/kakouzou_pub/001_1/0001s.tif/full/full/0/default.jpg"
    },
    "nijl": {
        "url" : "https://kotenseki.nijl.ac.jp/api/iiif/100302677/v4/KANS/KANS-00281/KANS-00281-00002.tif/full/full/0/default.jpg"
    },
    "kyoto": {
        "iiif" : "https://rmda.kulib.kyoto-u.ac.jp/iiif/RB00020027/RB00020027_00001_0.ptif/16512,327,1872,1200/full/0/default.jpg"
    },
    "kyushu": {
        "url" : "https://catalog.lib.kyushu-u.ac.jp/image/iiif/820/1467768/467234.tiff/full/full/0/default.jpg"
    }
}

#####

item = collection[key]

if "iiif" in item:
  url = item["iiif"]
  type = "iiif"
else:
  url = item["url"]
  type = "url"

'''

filename = "tmp.jpg"
# !wget -O $filename $url

# 超シンプルな画像保存
r = requests.get(url)
if r.status_code == 200:
    with open(filename, 'wb') as f:
        f.write(r.content)

if showFlag:
    display_jpeg(Image(filename))

img = cv2.imread(filename)
org_w = img.shape[1]
org_h = img.shape[0]
print("入力画像サイズ: {} px x {} px".format(org_w, org_h))