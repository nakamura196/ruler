FROM python:3.8-slim-buster

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /app
ADD . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "webapp.py", "--port=5000"]