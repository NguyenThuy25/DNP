#!/usr/bin/env python

import json
import sys
from argparse import ArgumentParser, FileType
from configparser import ConfigParser
from confluent_kafka import Consumer, OFFSET_BEGINNING, KafkaError, TopicPartition
import cv2
from ultralytics import YOLO
import numpy as np
from confluent_kafka import Producer

from video_producer import delivery_report, serializeImg


class Detector:
    def __init__(self, producer_config, detector_config, topic, model):
        self.producer = Producer(producer_config)
        self.consumer = Consumer(detector_config)
        self.consumer.subscribe([topic])
        self.topic = topic
        self.model = model
        self.last_offset = 0
    
    def process_frame(self, frame, offset, visualize):
        results = self.model.track(frame, persist=True)
        bbox_data = {
            "offset": offset,
            "bbox": []
        }
        for result in results:
            boxes = result.boxes
            for box in boxes:
                bbox_data["bbox"].append(box.xyxy[0])
        print(f"Detection for frame at offset {offset}")
        if visualize:
            annotated_frame = results[0].plot()
            cv2.imshow('Image', annotated_frame)
            cv2.waitKey(1)
        return bbox_data
    
    def receive_and_process_frames(self, visualize=False):
        try:
            while True:
                msg = self.consumer.poll(0.5)
                if msg == None:
                    print("Waiting...")
                elif msg.error():
                    print("ERROR: %s".format(msg.error()))
                else:
                    partitions = self.consumer.assignment()
                    for partition in partitions:
                        first, last = self.consumer.get_watermark_offsets(partition)
                        if (last-1) != self.last_offset:
                            print(first, last)
                            tp = TopicPartition(self.topic, partition=0, offset=last-1)
                            self.consumer.seek(tp)
                            self.last_offset = last-1
                    # convert image bytes data to numpy array of dtype uint8
                    nparr = np.frombuffer(msg.value(), np.uint8)

                    # decode image
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    # results = self.model.track(img, persist=True)
                    # bbox_data = {
                    #     "offset": msg.offset(),
                    #     "bbox": []
                    # }
                    # for result in results:
                    #     # print(result.boxes)   
                    #     boxes = result.boxes
                    #     for box in boxes:
                    #         bbox_data["bbox"].append(box.xyxy[0])
                    bbox_data = self.process_frame(img, msg.offset(), visualize=visualize)
                    # print(bbox_data)
                    # send data to producer
                    self.producer.produce(value=json.dumps(bbox_data, default=lambda x: x.tolist()).encode('utf-8'), topic="bbox", on_delivery=delivery_report)
                    self.producer.poll(0)
                    
        except KeyboardInterrupt:
            print("Detected Keyboard Interrupt. Quitting...")
            pass

        finally:
            self.consumer.close()
            if visualize:
                cv2.destroyAllWindows()


    # def run(self):


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
    producer_config = dict(config_parser['producer'])
    detector_config = dict(config_parser['detector'])
    topic = dict(config_parser['detector_topic'])['topic']
    model = YOLO('ckpt/yolov8n.pt')
    consumer = Detector(producer_config, detector_config, topic, model)
    # Subscribe to topic
    consumer.receive_and_process_frames(visualize=True)
  