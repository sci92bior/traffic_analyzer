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
KAFKA_URL = '127.0.0.1:9092'
SECURITY_CONTROLLER_URL = 'http://127.0.0.1:9000/alerts/'

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
            logger.warning(f"Received message: {message.value}")
            value = message.value.decode('utf-8')  # Assuming messages are strings
            messages.append(value)
            logger.warning(f"Received message: {messages}")

            if len(messages) >= batch_size:
                data_array = []
                for i in messages:
                    data_array.append(json.loads(i))

                # calculate mean time difference between messages
                time_diff = 0
                for i in range(len(data_array) - 1):
                    time_diff += data_array[i + 1]['time_received_ns'] - data_array[i]['time_received_ns']
                time_diff /= len(data_array) - 1
                logger.warning(f"Time difference: {time_diff}")

                # if time_diff > 1000000000:
                #     scr_addr = data_array[0]["src_addr"]
                #     dst_addr = data_array[0]["dst_addr"]
                #     src_port = data_array[0]["src_port"]
                #     dst_port = data_array[0]["dst_port"]
                #     source_ip = data_array[0]["sampler_address"]
                #     send_post_request({"alert_type": "ddos","device_ip": source_ip,"src_ip": scr_addr, "dst_ip": dst_addr, "port": src_port, })
                messages = []  # Reset messages list after sending batch
    finally:
        await consumer.stop()


asyncio.create_task(consume())
