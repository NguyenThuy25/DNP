from argparse import ArgumentParser, FileType
from configparser import ConfigParser
import json
from pathlib import Path
from confluent_kafka import Consumer, Producer
import cv2
import numpy as np
# from pose import PoseEstimation
from boxmot import OCSORT

from rtmpose.cspnext import CSPNeXt
from rtmpose.data_preprocesser import PoseDataPreprocessor
from rtmpose.rtmcc_head import RTMCCHead
from rtmpose.test import TransformRegistry, inference_topdown, init_model
from rtmpose.top_down import TopdownPoseEstimator
from rtmpose.transforms.common_transforms import GetBBoxCenterScale
from rtmpose.transforms.formatting import PackPoseInputs
from rtmpose.transforms.loading import LoadImage
from rtmpose.transforms.topdown_transforms import TopdownAffine
from utils.visualization.draw import draw_skeleton
from video_producer import delivery_report

class PoseEstimator:
    def __init__(self, pose_config, model):
        self.received_frame = []
        # self.pose_model = PoseEstimation(model_type='rtmpose | body')
        self.pose_model = model
        self.consumer = Consumer(pose_config)
        self.consumer.subscribe([detection_topic, frame_topic])
        self.output_data = []
        self.output_frame = []
        self.producer = Producer(pose_config)
    
    def process_frames(self, detection_data, bbox_data, visualize=False):
        detection_offset = detection_data['offset']
        bbox_offset = bbox_data['offset']
        # SENDING DATA
        out_data = {
            'offset': bbox_offset,
            'bbox': [],
            'kp': [],
            'kp_score': [],
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


            pose_result = inference_topdown(self.pose_model, detection_frame, [item[:4] for item in bbox_info_list], 'xyxy')
            for pose in pose_result:
                kps = pose.pred_instances.keypoints
                bboxes = pose.pred_instances.bboxes
                score = pose.pred_instances.keypoint_scores
                out_data['kp'].append(kps)
                out_data['kp_score'].append(score)
                # import IPython; IPython.embed()
                # out_data['bbox'].append(bboxes)
                if visualize:
                    x1, y1, x2, y2 = int(bboxes[0][0]), int(bboxes[0][1]), int(bboxes[0][2]), int(bboxes[0][3])
                    cv2.rectangle(detection_data['image'], (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # for single_kp in kps[0]:
                    #     cv2.circle(detection_data['image'], (int(single_kp[0]), int(single_kp[1])), radius = 1, color = (0, 255, 0), thickness = -1)
                    img = draw_skeleton(img=detection_data['image'], keypoints=kps,
                                scores=pose.pred_instances.keypoint_scores, kpt_thr=0.2)
                    # cv2.imshow('Image', detection_data['image'])
                    cv2.imshow('Image', img)
                    cv2.waitKey(1)
            out_data['bbox'].append([box for box in bbox_info_list])
            
            print(f"Pose estimation for frame at offset {detection_offset}")
            return out_data
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
                                kps_data = self.process_frames(data, bbox_data)
                                # self.output_frame.append(res_frame)
                                kps_data_list.append(kps_data)
                        
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
    pose_config = dict(config_parser['pose_estimator'])

    frame_topic = "action_detection"
    detection_topic = "bbox"

    # POSE MODEL
    TransformRegistry.register('LoadImage')(LoadImage)
    TransformRegistry.register('GetBBoxCenterScale')(GetBBoxCenterScale)
    TransformRegistry.register('TopdownAffine')(TopdownAffine)
    TransformRegistry.register('PackPoseInputs')(PackPoseInputs)

    config = '/Users/nttthuy/Documents/Project/DNP/rtmpose/config/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py'
    checkpoint = "./ckpt/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth"
    model = init_model(config=config, checkpoint=checkpoint, device='cpu')


    # model = YOLO('ckpt/yolov8n.pt')
    pose = PoseEstimator(pose_config=pose_config, model=model)
    # Subscribe to topic
    pose.receive_bbox(visualize=True)
    # pose.save_output_video("output/output.mp4")
    # pose.video_writer.release()
    # pose.save_output_json("output/output.json")
