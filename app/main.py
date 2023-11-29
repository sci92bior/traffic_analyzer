

import asyncio
import csv
import json
import os
from typing import List
from random import shuffle

from aiokafka import AIOKafkaConsumer
from fastapi import FastAPI

from test_script import process

app = FastAPI()
loop = asyncio.get_event_loop()

async def consume():
    consumer = AIOKafkaConsumer(
        'flows',
        loop=loop,
        bootstrap_servers='kafka:9092',
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
            value = message.value.decode('utf-8')  # Assuming messages are strings
            messages.append(value)

            if len(messages) >= batch_size:
                for item in messages:
                    # await process(item)
                    print(item)
                messages = []  # Reset messages list after sending batch
    finally:
        await consumer.stop()

asyncio.create_task(consume())
