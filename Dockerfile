# # Use Python 3.9 as the base image
# FROM python:3.9-slim

# # Set the working directory
# WORKDIR /app

# # Copy only the requirements file
# COPY requirements.txt .

# # Upgrade pip and install dependencies
# RUN pip install --upgrade pip
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application
# COPY . .

# # Expose the port for Cloud Run
# ENV PORT=8080
# EXPOSE 8080

# # Start the application
# CMD ["gunicorn", "app:app", "-b", "0.0.0.0:8080", "--workers=2", "--threads=2", "--timeout=120"]

# Use Python 3.9 as the base image
FROM python:3.9-slim

# Install git and git-lfs
RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install

# Set the working directory
WORKDIR /app

# Copy only the requirements file
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the git files
COPY .git* ./
COPY .gitattributes ./

# Pull LFS files
RUN git lfs pull

# Copy the rest of the application
COPY . .

# Expose the port for Cloud Run
ENV PORT=8080
EXPOSE 8080

# Start the application with increased timeout and workers
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:8080", "--workers=2", "--threads=2", "--timeout=120"]
