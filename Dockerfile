# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for numpy/matplotlib/prophet if necessary
# build-essential and python3-dev are often needed for compiling some python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Create CSV directory if it doesn't exist (for data persistence if using volumes, or just to prevent errors)
RUN mkdir -p CSV

# Make port 8000 available to the world outside this container
# Render will override this with the PORT env var, but good for documentation
EXPOSE 8000

# Define environment variable
ENV PORT=8000

# Run app.py when the container launches
# Using shell form to allow variable expansion for PORT
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT}
