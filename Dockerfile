FROM python:3.8

COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U numpy
RUN pip install --index-url https://google-coral.github.io/py-repo/ tflite-runtime

EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app

LABEL authors="kausik"