# Yolov5 object detection model deployment using flask

## Run & Develop locally

Run locally and dev:

- `python3 -m venv venv`
- `source venv/bin/activate`
- `(venv) $ pip install -r requirements.txt`
- `(venv) $ python3 webapp.py --port 5000`

## Docker

The example dockerfile shows how to expose the rest API:

```
# Build
docker build -t ruler .
# Run
docker run -p 5000:80 ruler
```

## reference

- https://github.com/ultralytics/yolov5
- https://github.com/jzhang533/yolov5-flask (this repo was forked from here)
- https://github.com/avinassh/pytorch-flask-api-heroku
