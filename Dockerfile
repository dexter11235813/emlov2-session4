
FROM python:3.9-slim-buster

WORKDIR /opt

COPY "./requirements.txt" .

RUN export DEBIAN_FRONTEND=noninteractive \
  && echo "LC_ALL=en_US.UTF-8" >> /etc/environment \
  && echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
  && echo "LANG=en_US.UTF-8" > /etc/locale.conf \
  && apt update && apt install -y locales \
  && locale-gen en_US.UTF-8 \
  && rm -rf /var/lib/apt/lists/* \
  &&  pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt && rm -rf /root/.cache/pip

# COPY src/ /opt/src 

# COPY /logs/train/runs/2022-10-04_15-52-16/model.script.pt /opt/model.script.pt

# COPY configs/ /opt/configs

# COPY pyproject.toml /opt/

COPY docker_folder . 

EXPOSE 8080

ENTRYPOINT [ "python", "src/demo_scripted.py", "ckpt_path=model.script.pt", "experiment=cifar10"]
