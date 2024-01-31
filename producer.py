#!/usr/bin/env python

import logging
import sys
from random import choice
import concurrent.futures
from argparse import ArgumentParser, FileType
from configparser import ConfigParser
import time
from confluent_kafka import Producer
import os

import cv2

def serializeImg(img):
    _, img_buffer_arr = cv2.imencode(".jpg", img)
    img_bytes = img_buffer_arr.tobytes()
    return img_bytes

def delivery_report(err, msg):
    if err:
        logging.error("Failed to deliver message: {0}: {1}"
              .format(msg.value(), err.str()))
    else:
        logging.info(f"msg produced. \n"
                    f"Topic: {msg.topic()} \n" +
                    f"Partition: {msg.partition()} \n" +
                    f"Offset: {msg.offset()} \n" +
                    f"Timestamp: {msg.timestamp()} \n")
        
class ProducerThread:
    def __init__(self, config):
        self.producer = Producer(config)
    
    def publishFrame(self, video_path):
        video = cv2.VideoCapture(video_path)
        video_name = os.path.basename(video_path).split(".")[0]
        frame_no = 1
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                frame_bytes = serializeImg(frame)
                self.producer.produce(
                    topic="action_detection", 
                    value=frame_bytes, 
                    on_delivery=delivery_report,
                    timestamp=frame_no,
                    headers={
                        "video_name": str.encode(video_name)
                    }
                )
                print("frame", frame_no)
                self.producer.poll(0)
                
                frame_no += 1
            else:
                break
        video.release()
        self.producer.flush()
        print("done")
        return
    
    def start(self, vid_paths):
        # runs until the processes in all the threads are finished
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self.publishFrame, vid_paths)

        self.producer.flush() # push all the remaining messages in the queue
        print("Finished...")


if __name__ == '__main__':
    # Parse the command line.
    parser = ArgumentParser()
    parser.add_argument('config_file', type=FileType('r'))
    args = parser.parse_args()

    # Parse the configuration.
    # See https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
    config_parser = ConfigParser()
    config_parser.read_file(args.config_file)
    config = dict(config_parser['default'])
   
    # Create Producer instance
    video_path = "/Users/ctpanh/Documents/code/dnp/video/moving.mp4"
    
    producer = ProducerThread(config)
    producer.publishFrame(video_path)