import asyncio
import csv
import json
import logging
import os
from typing import List
from random import shuffle

import requests
from aiokafka import AIOKafkaConsumer
from fastapi import FastAPI

from app.test_script import process

logger = logging.getLogger(__name__)
KAFKA_URL = os.environ.get('KAFKA_URL')
SECURITY_CONTROLLER_URL = os.environ.get('SECURITY_CONTROLLER_URL')

app = FastAPI()
loop = asyncio.get_event_loop()


def send_post_request(data):
    # Convert the dictionary to a JSON string
    json_data = json.dumps(data)
    # Set the headers to indicate that you're sending JSON data
    headers = {'Content-Type': 'application/json'}

    # Send the POST request with JSON data
    response = requests.post(SECURITY_CONTROLLER_URL, headers=headers, data=json_data)

    # Check the response status and content
    if response.status_code == 200:
        print("POST request successful!")
        print("Response content:", response.json())
    else:
        print("POST request failed with status code:", response.status_code)
        print("Response content:", response.text)


async def consume():
    consumer = AIOKafkaConsumer(
        'flows',
        loop=loop,
        bootstrap_servers=KAFKA_URL,
    )

    try:
        await consumer.start()

    except Exception as e:
        print(e)
        return

    batch_size = 5
    messages = []

    try:
        async for message in consumer:
            if message is not None:
                logger.warning(f"Received message: {message.value}")
                value = message.value.decode('utf-8')  # Assuming messages are strings
                messages.append(value)
                logger.warning(f"Received message: {messages}")

                if len(messages) >= batch_size:
                    await process(messages)
                    send_post_request({"alert_type": "string",
                                       "device_id": "string",
                                       "port": "string",
                                       "src_ip": "string",
                                       "dst_ip": "string"
                                       })
                    messages = []  # Reset messages list after sending batch
    finally:
        await consumer.stop()


asyncio.create_task(consume())
