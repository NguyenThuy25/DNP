# DNP

Week 1: Init Kafka
 - Read and send frame to Kafka
 - Receive frame from Kafka and run detection

Week 2: Init DNP
 - Node A: read and send frame to node B, C
 - Node B: receive frame, run detection, send frame to node C
 - Node C: receive frame (from node A) and bbox (from node B) to run pose estimation

Week 3: Tracking
 - Run tracking in node B

Week 4: Complete
 - Run Kafka docker
 - Drop frame using OFFSET_END
 - Node D: save file output video, json

Week 5: Implement RTMPOSE
 - Implemnt RTMPOSE (from mmpose)
