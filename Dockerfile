FROM python:3.9-slim


WORKDIR /app

COPY src ./src

RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8001

CMD ["python","src/app.py"]
