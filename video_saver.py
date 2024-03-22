from argparse import ArgumentParser, FileType
from configparser import ConfigParser
import json
from queue import Queue
from confluent_kafka import Consumer
import cv2
import numpy as np

class VideoSaver:
    def __init__(self, saver_config, json_path, video_path):
        self.frame_consumer = Consumer(saver_config)
        self.kp_consumer = Consumer(saver_config)
        self.frame_consumer.subscribe(['action_detection'])
        self.kp_consumer.subscribe(['kp'])
        self.frame = Queue()
        self.kp = Queue()
        self.json_path = json_path
        self.video_path = video_path
        # self.video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (1080, 1920))  # Change width and height as per your frame size
        self.video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (1920, 1080))  # Change width and height as per your frame size

    def receive_and_procress_frames(self):
        try:
            while True:
                frame_msg = self.frame_consumer.poll(0)
                kp_msg = self.kp_consumer.poll(0)
    
                if frame_msg is not None:
                    nparr = np.frombuffer(frame_msg.value(), np.uint8)
                    # decode image
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    frame_data = {'offset': frame_msg.offset(), 'image': img}
                    self.frame.put(frame_data)

                if kp_msg is not None:
                    decoded_kp_data = json.loads(kp_msg.value().decode('utf-8'))
                    kp_data = {'offset': decoded_kp_data[0]['offset'], 'bbox':decoded_kp_data[0]['bbox'], 'kp': decoded_kp_data[0]['kp']}
                    self.kp.put(kp_data)
                    
                    frame = self.frame.get()
                    while frame['offset'] < kp_data['offset']:
                        frame = self.frame.get()
                        pass
                    
                    # Write to video
                    image = frame['image']
                    color = (0, 0, 255)  # BGR
                    thickness = 2
                    fontscale = 0.6
                    xs = [box[0] for box in kp_data['bbox']]  # Extract x values
                    ys = [box[1] for box in kp_data['bbox']]  # Extract y values
                    ws = [box[2] for box in kp_data['bbox']]  # Extract width values
                    hs = [box[3] for box in kp_data['bbox']]  # Extract height values
                    ids = [box[4] for box in kp_data['bbox']]  # Extract id values
                    confs = [box[5] for box in kp_data['bbox']]  # Extract confidence values
                    cls = [box[6] for box in kp_data['bbox']]  # Extract class values
                    kps = [kp for kp in kp_data['kp']]
                    for x, y, w, h, id, conf, cls, kp in zip(xs, ys, ws, hs, ids, confs, cls, kps):
                        x, y, w, h = int(x), int(y), int(w), int(h)
                        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
                        cv2.putText(
                            image,
                            f'id: {int(id)}, conf: {conf: .2f}, c: {int(cls)}',
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontscale,
                            color,
                            thickness
                        )
                        for single_kp in kp:
                            cv2.circle(image, (int(x + single_kp[0]), int(y + single_kp[1])), radius = 2, color = (0, 255, 0), thickness = -1)
                    self.video_writer.write(image)
                    print("video_writer is open", self.video_writer.isOpened())
                    print("image shape", image.shape)

                    # Write to JSON
                    print("offset", frame['offset'], kp_data['offset'])
                    print("len", len(self.frame.queue), len(self.kp.queue))
                    if frame['offset'] == kp_data['offset']:
                        with open(self.json_path, 'a') as f:
                            json.dump(kp_data, f)
                            f.write('\n')

                    cv2.imshow('Image', image)
                    cv2.waitKey(1)

        except KeyboardInterrupt:
            print("Detected Keyboard Interrupt. Quitting...")
            pass

        finally:
            self.frame_consumer.close()
            self.kp_consumer.close()
            self.video_writer.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    # Parse the command line.
    parser = ArgumentParser()
    parser.add_argument('config_file', type=FileType('r'))
    parser.add_argument('--reset', action='store_true')
    args = parser.parse_args()

    config_parser = ConfigParser()
    config_parser.read_file(args.config_file)
    config = dict(config_parser['video_saver'])

    video_saver = VideoSaver(config, json_path="output/output.json", video_path="output/output.mp4")
    video_saver.receive_and_procress_frames()
    # video_saver.video_writer.release()