FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install -U pip && pip install -r requirements.txt

COPY . /app
ENV APP_CONFIG=configs/app.yaml
EXPOSE 8000

CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
