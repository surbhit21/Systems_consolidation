FROM python:3.9-slim

# setting up the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .
# install the dependencies using pip
RUN  pip install -r requirements.txt

# copy the rest of the application code
COPY . .

CMD ["python", "app.py"]