FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

WORKDIR /app/src

EXPOSE 18792

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "18792"]
