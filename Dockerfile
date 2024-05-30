# Use the official optimized image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

# Set the working directory in the container
WORKDIR /app

# RUN adduser --disabled-password --gecos '' myuser
# USER myuser

# Copy the requirements file first, for better cache on rebuilds
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code
RUN mkdir -p /app/argo_app

COPY ./argo_app/*.py /app/argo_app/

COPY ./argo_app/src /app/argo_app/src

# RUN pip install -e .

# Make port 8090 available outside this container
EXPOSE 8090

# Use fastapi run to serve your application in the single file
# CMD ["fastapi", "run", "argo_app.py", "--port", "8090", "--timeout", "600"]

ENV MODULE_NAME=argo_app.app
ENV VARIABLE_NAME=app
ENV GUNICORN_CMD_ARGS="--timeout 600 --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8090"
