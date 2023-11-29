

import asyncio
import csv
import json
import logging
import os
from typing import List
from random import shuffle

from aiokafka import AIOKafkaConsumer
from fastapi import FastAPI
logger = logging.getLogger(__name__)


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
            logger.warning(f"Received message: {message.value}")
            value = message.value.decode('utf-8')  # Assuming messages are strings
            messages.append(value)

            if len(messages) >= batch_size:
                for item in messages:
                    # await process(item)
                    logger.warning(f"Received message: {item.value}")
                messages = []  # Reset messages list after sending batch
    finally:
        await consumer.stop()

asyncio.create_task(consume())
