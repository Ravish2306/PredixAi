FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install fastapi uvicorn numpy tensorflow psycopg2-binary

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]