FROM python:latest

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENV NAME World

CMD ["python", "app.py"]
