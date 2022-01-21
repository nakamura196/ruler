from lib import detect, isHorizontal, otsu, skelton, hlsd, arg_r, check, output, resize
import cv2
import argparse    # 1. argparseをインポート
import torch

parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')    # 2. パーサを作る

# 3. parser.add_argumentで受け取る引数を追加していく
parser.add_argument('url', help='この引数の説明（なくてもよい）')
parser.add_argument('-t', '--type', default="url", help='この引数の説明（なくてもよい）')
parser.add_argument('-s', '--show', default=False, type=bool, help='この引数の説明（なくてもよい）')

args = parser.parse_args()    # 4. 引数を解析

url = args.url
type = args.type

org = cv2.imread("tmp.jpg")
org_h, org_w, org_z = org.shape

#####

showFlag = args.show

# ロード
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt') # .autoshape()  # force_reload = recache latest code

size = detect(model, showFlag)

if size == 0:
  print("定規が検出されませんでした。")
else:
  horizontal = isHorizontal()
  otsu()

  # 細線化
  skelton(showFlag)

  # 直線の検出
  x, y = hlsd(horizontal, showFlag)

  # 極大値の表示
  arg_r_max = arg_r(x, y, False)
  if showFlag:
    check(x, horizontal, arg_r_max, showFlag)
  value = output(x, arg_r_max, horizontal)

  if showFlag:
    from IPython.display import Image,display_jpeg
    display_jpeg(Image("output.jpg"))
  # print("1ピクセルあたりのサイズ: {} mm".format(value))
  ratio = value["pixelPerMM"]
  ruler_w = int(value["width"] / ratio)
  ruler_h = int(value["height"] / ratio)
  input_image_w = int(org_w / ratio)
  input_image_h = int(org_h / ratio)
  print("1mmあたりのピクセル数: {} pixes".format(ratio))
  print("検出した定規画像の実サイズ: {} mm x {} mm".format(ruler_w, ruler_h))
  print("入力画像の実サイズ: {} mm x {} mm".format(input_image_w, input_image_h))

  if type == "iiif":
    resize(url, input_image_w, input_image_h)
