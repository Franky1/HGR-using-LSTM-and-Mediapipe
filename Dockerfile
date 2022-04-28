# this Dockerfile mimics the streamlit cloud runtime
FROM python:3.7-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# build tools
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    python3-dev

WORKDIR /app

# if we have a packages.txt, install it
COPY packages.txt packages.txt
RUN xargs -a packages.txt apt-get install --yes

# update python tools
RUN python -m pip install --upgrade pip
RUN pip install --upgrade setuptools wheel

# install python packages
COPY requirements.txt requirements.txt
RUN pip install --upgrade -r requirements.txt

EXPOSE 8501

COPY . .

CMD ["streamlit", "run", "camera.py"]

# docker build --progress=plain --tag hgr:latest .
# docker run -ti -p 8501:8501 --rm hgr:latest /bin/bash
# docker run -ti -p 8501:8501 --rm hgr:latest
# docker run -ti -p 8501:8501 -v ${pwd}:/app --rm hgr:latest
# docker run -ti -p 8501:8501 -v ${pwd}:/app --rm hgr:latest /bin/bash
