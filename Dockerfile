FROM python:3.9.11
WORKDIR /app
ADD . /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt
EXPOSE 5001
ENV NAME OpentoAll
CMD ["echo", "Feedback Dapp Docker container has started..."]
CMD ["python", "app.py"]