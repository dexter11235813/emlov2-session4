FROM python:3.9.7-slim-buster


RUN apt-get update &&  apt-get install -y -qq  \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists

RUN pip install --upgrade pip

WORKDIR /opt/src

COPY "./requirements.txt" .

# RUN apt-get install libgtk2.0-dev pkg-config -yqq

RUN pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt

# COPY . .

# ENTRYPOINT [ "python", "main.py" ]
