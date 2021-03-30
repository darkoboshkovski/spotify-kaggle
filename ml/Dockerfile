FROM python:3.8-slim

WORKDIR /app

ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m venv $VIRTUAL_ENV \
    && pip install --upgrade pip \
    && pip install --upgrade setuptools wheel

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
CMD ["uvicorn","app:app","--workers","1","--host","0.0.0.0"]
