FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye
# Install necessary build dependencies
RUN apt-get update
RUN python3.12 -m pip install --no-cache-dir ipykernel python-dotenv pdf2image azure-ai-documentintelligence openai docx2pdf python-pptx azure-identity scikit-learn
RUN python3.12 -m pip cache purge
RUN apt install poppler-utils -y
RUN pip install --upgrade pip

