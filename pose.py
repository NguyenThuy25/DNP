from mmpose.apis import MMPoseInferencer

class PoseEstimation:
    def __init__(self, model_type='rtmpose | body'):
        self.model_type = model_type
        self.model = None

        self.setup_models()

    def setup_models(self):
        if self.model_type == 'rtmpose | body':
            pose2d_model = 'rtmpose-l'
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.model = MMPoseInferencer(pose2d=pose2d_model, device='cpu')

    def predict(self, input_image, draw_heatmap=False):
        result = next(self.model(input_image, return_vis=True, draw_heatmap=draw_heatmap))
        img = result['visualization'][0][..., ::-1]
        return img

# if __name__ == "__main__":
#     pose_estimation = PoseEstimation(model_type='rtmpose | body')
#     pose_estimation.run_demo()