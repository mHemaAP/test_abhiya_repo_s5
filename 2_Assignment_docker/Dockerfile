
FROM python:3.12-slim
# create dockerfile here

# Set the working directory in the container

WORKDIR /workspace
# Copy the necessary Python scripts into the container
COPY train.py eval.py infer.py model.py /workspace/

RUN pip install \
	torch==2.3.1+cpu \
	torchvision==0.18.1+cpu \
	-f https://download.pytorch.org/whl/torch_stable.html \
	&& rm -rf /root/.cache/pip



# Set a default command to run train.py (this will be overridden by docker-compose.yml for other services)
CMD ["python", "train.py"]

