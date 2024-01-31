#!/usr/bin/env python

import sys
from argparse import ArgumentParser, FileType
from configparser import ConfigParser
from confluent_kafka import Consumer, OFFSET_BEGINNING, KafkaError
import cv2
from ultralytics import YOLO
import numpy as np


class ConsumerThread:
    def __init__(self, config, topic, model):
        self.consumer = Consumer(config)
        self.consumer.subscribe([topic])
        self.model = model

    def run(self):
        try:
            while True:
                msg = self.consumer.poll(0.5)
                if msg == None:
                    print("Waiting...")
                elif msg.error():
                    print("ERROR: %s".format(msg.error()))
                else:

                    # convert image bytes data to numpy array of dtype uint8
                    nparr = np.frombuffer(msg.value(), np.uint8)

                    # decode image
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    # img = cv2.resize(img, (224, 224))

                    results = self.model.track(img, persist=True)
                    annotated_frame = results[0].plot()
                    cv2.imshow('Image', annotated_frame)
                    cv2.waitKey(1)

        except KeyboardInterrupt:
            print("Detected Keyboard Interrupt. Quitting...")
            pass

        finally:
            self.consumer.close()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    # Parse the command line.
    parser = ArgumentParser()
    parser.add_argument('config_file', type=FileType('r'))
    parser.add_argument('--reset', action='store_true')
    args = parser.parse_args()

    # Parse the configuration.
    # See https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
    config_parser = ConfigParser()
    config_parser.read_file(args.config_file)
    config = dict(config_parser['default'])
    config.update(config_parser['consumer'])

    topic = "action_detection"
    model = YOLO('yolov8n.pt')
    consumer = ConsumerThread(config, topic, model)
    # Subscribe to topic
    consumer.run()
  