# Use a base image with Python
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script into the container
COPY src/train.py ./  # Copying train.py from the src directory

# Copy the data directory into the container
COPY data ./data  # This will copy the entire data directory

# Copy the application code into the container (app.py is in the same folder as Dockerfile)
COPY app.py ./  # Make sure the path to app.py is correct

# Run the training script to generate the model
RUN python train.py

# Set the command to run the application
CMD ["python", "app.py"]
