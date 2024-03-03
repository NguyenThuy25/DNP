#!/usr/bin/env python

import json
from pathlib import Path
import sys
from argparse import ArgumentParser, FileType
from configparser import ConfigParser
from boxmot import DeepOCSORT
from confluent_kafka import Consumer, OFFSET_BEGINNING, KafkaError, TopicPartition
import cv2
from ultralytics import YOLO
import numpy as np
from confluent_kafka import Producer

from video_producer import delivery_report, serializeImg


class Detector:
    def __init__(self, producer_config, detector_config, topic, detection_model, tracking_model):
        self.producer = Producer(producer_config)
        self.consumer = Consumer(detector_config)
        self.consumer.subscribe([topic])
        self.topic = topic
        self.detection_model = detection_model
        self.tracking_model = tracking_model
        self.last_offset = 0
    
    def visualize(self, frame, tracks):
        color = (0, 0, 255)  # BGR
        thickness = 2
        fontscale = 0.5
        if tracks.shape[0] != 0:
            xyxys = tracks[:, 0:4].astype('int') # float64 to int
            ids = tracks[:, 4].astype('int') # float64 to int
            confs = tracks[:, 5].round(decimals=2)
            clss = tracks[:, 6].astype('int') # float64 to int
            # inds = tracks[:, 7].astype('int') # float64 to int

            # print bboxes with their associated id, cls and conf
            for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
                frame = cv2.rectangle(
                    frame,
                    (xyxy[0], xyxy[1]),
                    (xyxy[2], xyxy[3]),
                    color,
                    thickness
                )
                cv2.putText(
                    frame,
                    f'id: {id}, conf: {conf}, c: {cls}',
                    (xyxy[0], xyxy[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontscale,
                    color,
                    thickness
                )

        # show image with bboxes, ids, classes and confidences
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    def process_frame(self, frame, offset, visualize):
        results = self.detection_model.track(frame, persist=True)
        for result in results:
            boxes = result.boxes
            dets = []
            for box in boxes:
                det = []
                det.append(float(box.xyxy[0][0]))
                det.append(float(box.xyxy[0][1]))
                det.append(float(box.xyxy[0][2]))
                det.append(float(box.xyxy[0][3]))
                # det.append(int(box.id))
                det.append(float(box.conf))
                det.append(int(box.cls))
            # N X (x, y, x, y, conf, cls)
                dets.append(det)
            dets = np.array(dets)
        print("dets", dets)
        # (x, y, x, y, id, conf, cls, ind)
        tracks = self.tracking_model.update(dets, frame)

        # SENDING DATA
        bbox_id_data = {
            "offset": offset,
            "bbox": [],
        }
        for track in tracks:
            # (x, y, x, y, id, conf, cls)
            bbox_id_data["bbox"].append(track[0:7].tolist())
        if visualize:
            self.visualize(frame, tracks)
            # self.visualize(frame, dets)
        print("bbox_id_data", bbox_id_data)
        return bbox_id_data
    
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

                    bbox_id_data = self.process_frame(img, msg.offset(), visualize=visualize)
                    # send data to producer
                    self.producer.produce(value=json.dumps(bbox_id_data, default=lambda x: x.tolist()).encode('utf-8'), topic="bbox", on_delivery=delivery_report)
                    # self.producer.poll(0)
                    
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
    detection_model = YOLO('ckpt/yolov8n.pt')
    tracking_model = DeepOCSORT(
        model_weights=Path('ckpt/osnet_x0_25_msmt17.pt'), # which ReID model to use
        device='cpu',
        fp16=False,
    )
    consumer = Detector(producer_config, detector_config, topic, detection_model, tracking_model)
    # Subscribe to topic
    consumer.receive_and_process_frames(visualize=False)
  