# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9.20-slim


WORKDIR /workspace
COPY train.py train.py
COPY requirements.txt requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "train.py"]




# #WORKDIR /app
# WORKDIR /workspace 

# #COPY . /app
# COPY train.py /workspace
# COPY requirements.txt /workspace 


# RUN python -m pip install -r requirements.txt
# #size of images small-delete cache
# RUN pip install --no-cache-dir -r requirements.txt

# # # Creates a non-root user with an explicit UID and adds permission to access the /app folder
# # # For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# # RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# # USER appuser

# # During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
# CMD ["python", "train.py"]
