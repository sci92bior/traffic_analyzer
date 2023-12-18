# Traffic Analyzer
## Description
This is a traffic analyzer for the SDN network. It is based on the [Ryu](https://osrg.github.io/ryu/) SDN controller framework.
The controller is able to detect and mitigate attacks on the network. It is also able to detect and mitigate attacks on the controller itself.

Its written in Python 3.11 and uses FastAPI as a web framework.

## Installation

Before running the controller, make sure that the following environment variables are set:
- KAFKA_URL - URL of the Kafka broker
- SECURITY_CONTROLLER_URL - URL of the security controller

### Requirements
- Python 3.11
- FastAPI
- PostgreSQL

### Docker
The easiest way to run the controller is to use the provided Dockerfile. To build the image run:
```bash
docker build -t traffic-analyzer .
```
To run the container:
```bash
docker run -p 8080:8080 traffic-analyzer
```

### Manual

To run the controller manually, first install the requirements:
```bash
pip install -r requirements.txt
```
Run FastAPI app:
```bash
uvicorn main:app --reload
```
The traffic analyzer will be available at http://localhost:8080
