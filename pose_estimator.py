from argparse import ArgumentParser, FileType
from configparser import ConfigParser
import json
from pathlib import Path
from confluent_kafka import Consumer, Producer
import cv2
import numpy as np
from pose import PoseEstimation
from boxmot import OCSORT

from video_producer import delivery_report

class PoseEstimator:
    def __init__(self, pose_config):
        self.received_frame = []
        self.pose_model = PoseEstimation(model_type='rtmpose | body')
        self.consumer = Consumer(pose_config)
        self.consumer.subscribe([detection_topic, frame_topic])
        self.output_data = []
        self.output_frame = []
        self.producer = Producer(pose_config)
    
    def process_frames(self, detection_data, bbox_data):
        detection_offset = detection_data['offset']
        bbox_offset = bbox_data['offset']
        # SENDING DATA
        out_data = {
            'offset': bbox_offset,
            'bbox': [],
            'kp': [],
        }
        # Check if offsets match
        if detection_offset == bbox_offset:

            # Perform pose estimation with the input frame and bbox information
            detection_frame = np.array(detection_data['image'], dtype=np.uint8)
            bbox_info_list = bbox_data['bbox']
            # bbox_info_list = np.array(bbox_info_list)
            
            # x, y, w, h = bbox_info_list[0:4]
            xs = [box[0] for box in bbox_info_list]  # Extract x values
            ys = [box[1] for box in bbox_info_list]  # Extract y values
            ws = [box[2] for box in bbox_info_list]  # Extract width values
            hs = [box[3] for box in bbox_info_list]  # Extract height values
            ids = [box[4] for box in bbox_info_list]  # Extract id values
            confs = [box[5] for box in bbox_info_list]  # Extract confidence values
            cls = [box[6] for box in bbox_info_list]  # Extract class values

            color = (0, 0, 255)  # BGR
            thickness = 2
            fontscale = 0.6
            for x, y, w, h, id, conf, cls in zip(xs, ys, ws, hs, ids, confs, cls):
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.rectangle(detection_frame, (x, y), (w, h), (0, 255, 0), 2)
                cv2.putText(
                    detection_frame,
                    f'id: {int(id)}, conf: {conf: .2f}, c: {int(cls)}',
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontscale,
                    color,
                    thickness
                )
                
                cropped_image = detection_frame[y:h, x:w]

                # Perform pose estimation on the cropped image
                pose_result, res = self.pose_model.predict(cropped_image)
                kps = res['predictions'][0]
                out_data['bbox'].append([x, y, w, h, id, conf, cls])
                out_data['kp'].append(kps[0]['keypoints'])
                # Apply the pose estimation result to the original image
                detection_frame[y:h, x:w] = pose_result
            
            print(f"Pose estimation for frame at offset {detection_offset}")
            return detection_frame, out_data
    def save_output_video(self, output_video_path):
        # Save the processed frames to a video file
        print(len(self.output_frame))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width, _ = self.output_frame[0].shape
        self.video_writer = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))
        for frame in self.output_frame:
            self.video_writer.write(frame)
        self.video_writer.release()
        print(f"Output video saved at {output_video_path}")
        
    def receive_bbox(self, visualize=False):
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
                        # bbox_data = {'offset': decoded_data['offset'], 'bbox': decoded_data['bbox'][0]}
                        bbox_data = {'offset': decoded_data['offset'], 'bbox': decoded_data['bbox']}
                        kps_data_list = []
                        for data in self.received_frame:
                            if data['offset'] == bbox_data['offset']:
                                res_frame, kps_data = self.process_frames(data, bbox_data)
                                self.output_frame.append(res_frame)
                                kps_data_list.append(kps_data)
                                if visualize:
                                    cv2.imshow('Image', res_frame)
                                    cv2.waitKey(1)
                                # print("output_data", self.output_data)
                                print(kps_data['offset'])
                        
                        self.received_frame = [d for d in self.received_frame if d['offset'] > bbox_data['offset']]
                        self.producer.produce(value=json.dumps(kps_data_list, default=lambda x: x.tolist()).encode('utf-8'), topic="kp", on_delivery=delivery_report)

                        
        except KeyboardInterrupt:
            print("Detected Keyboard Interrupt. Quitting...")
            pass

        finally:
            self.consumer.close()
            cv2.destroyAllWindows()
    def save_output_json(self, output_json_path):
        with open(output_json_path, 'w') as json_file:
            json.dump(self.output_data, json_file, default=lambda x: x.tolist())
                

        print(f"Output JSON saved at {output_json_path}")

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
    pose.receive_bbox(visualize=False)
    # pose.save_output_video("output/output.mp4")
    # pose.video_writer.release()
    # pose.save_output_json("output/output.json")
