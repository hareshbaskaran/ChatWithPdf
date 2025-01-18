FROM python:3.10-slim
LABEL authors="haresh"


COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app/main.py"]