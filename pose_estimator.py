from argparse import ArgumentParser, FileType
from configparser import ConfigParser
import json
from confluent_kafka import Consumer
import cv2
import numpy as np
from pose import PoseEstimation

class PoseEstimator:
    def __init__(self, pose_config):
        self.received_frame = []
        self.pose_estimation = PoseEstimation(model_type='rtmpose | body')

        self.consumer = Consumer(pose_config)
        self.consumer.subscribe([detection_topic, frame_topic])
        self.data_list = []
    
    def process_frames(self, detection_data, bbox_data):
        detection_offset = detection_data['offset']
        bbox_offset = bbox_data['offset']

        # Check if offsets match
        if detection_offset == bbox_offset:
            # Perform pose estimation with the input frame and bbox information
            detection_frame = np.array(detection_data['image'], dtype=np.uint8)
            bbox_info_list = bbox_data['bbox']

            x, y, w, h = bbox_info_list
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(detection_frame, (x, y), (w, h), (0, 255, 0), 2)
            cropped_image = detection_frame[y:h, x:w]

            # Perform pose estimation on the cropped image
            pose_result = self.pose_estimation.predict(cropped_image)

            # Apply the pose estimation result to the original image
            detection_frame[y:h, x:w] = pose_result

            print(f"Pose estimation for frame at offset {detection_offset}")
            return detection_frame

    def receive_bbox(self):
        try:
            while True:
                msg = self.consumer.poll(0.5)
                if msg == None:
                    print("Waiting...")
                elif msg.error():
                    print("ERROR: %s".format(msg.error()))
                else:
                    offset = msg.offset()
                    # convert image bytes data to numpy array of dtype uint8
                    if msg.topic() == "action_detection":
                        nparr = np.frombuffer(msg.value(), np.uint8)
                        # decode image
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        frame_data = {'offset': offset, 'image': img}
                        self.received_frame.append(frame_data)
                    elif msg.topic() == "bbox":
                        decoded_data = json.loads(msg.value().decode('utf-8'))
                        bbox_data = {'offset': decoded_data['offset'], 'bbox': decoded_data['bbox'][0]}
                        for data in self.received_frame:
                            if data['offset'] == bbox_data['offset']:
                                res_frame = self.process_frames(data, bbox_data)
                                cv2.imshow('Image', res_frame)
                                cv2.waitKey(1)
                        
                        self.received_frame = [d for d in self.received_frame if d['offset'] > bbox_data['offset']]

                        
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
    config = dict(config_parser['pose_estimator'])

    frame_topic = "action_detection"
    detection_topic = "bbox"

    # model = YOLO('ckpt/yolov8n.pt')
    pose = PoseEstimator(pose_config=config)
    # Subscribe to topic
    pose.receive_bbox()