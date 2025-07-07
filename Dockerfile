FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libhdf5-dev \
    libxml2-dev \
    libxslt-dev \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# setting up the working directory
WORKDIR /app

# Copy the requirements file
COPY requirement.txt .

# install the dependencies using pip
RUN  pip install  --prefer-binary --no-cache-dir -r requirement.txt

# copy the rest of the application code
COPY . .

CMD ["python", "test.py"]