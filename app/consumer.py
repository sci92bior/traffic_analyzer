import os
import json
from confluent_kafka import Consumer, KafkaException
import sys
import csv


def consume_messages():
    bootstrap_servers = os.getenv('KAFKA_BROKER', 'localhost:9092')
    group_id = os.getenv('KAFKA_GROUP_ID', 'postgres-inserter')
    topic = os.getenv('KAFKA_TOPIC', 'flows')

    conf = {
        'bootstrap.servers': bootstrap_servers,
        'group.id': group_id,
        'auto.offset.reset': 'earliest'
    }

    consumer = Consumer(conf)
    consumer.subscribe([topic])

    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaException._PARTITION_EOF:
                    sys.stderr.write('%% Reached end of partition %d\n' % msg.partition())
                else:
                    sys.stderr.write('%% Kafka error: %s\n' % msg.error().str())
                    break

            try:

                print('Received message:', msg.value())
                save_to_csv(msg.value())
            except json.JSONDecodeError as e:
                print('Received message could not be decoded as JSON:', msg.value().decode('utf-8'))

    except KeyboardInterrupt:
        sys.stderr.write('%% Aborted by user\n')

    finally:
        consumer.close()


def save_to_csv(message):
    # Parsing the JSON message
    data = json.loads(message.decode('utf-8'))

    # CSV file to which we'll append data
    csv_file = 'kafka_data.csv'  # Adjust the path as needed

    # Extract column names from JSON data
    columns = list(data.keys())

    # Append data to the CSV file
    with open(csv_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)

        # Write the data to the CSV file
        writer.writerow(data)


if __name__ == "__main__":
    consume_messages()
