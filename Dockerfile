# syntax=docker/dockerfile:1.2
FROM python:3.10-slim
# put you docker configuration here

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1

ENV PYTHONUNBUFFERED 1

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
    
RUN mkdir challenge 

#COPY challenge/{__init__.py,api.py,model.py} challenge/

COPY challenge/__init__.py challenge/
COPY challenge/api.py challenge/
COPY challenge/model.py challenge/

EXPOSE 8080

CMD ["uvicorn", "challenge.api:app", "--host=0.0.0.0", "--port=8080"]



