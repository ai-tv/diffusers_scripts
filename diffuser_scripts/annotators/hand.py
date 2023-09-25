from sbp.nn.app.mediapipe.hand_annotator import HandAnnotator


class Hand:

    def __init__(self):
        self.hand_model = HandAnnotator('/mnt/2T/zwshi/model_zoo/hand_landmarker.task')
    
    def __call__(self, image):
        image = image[..., ::-1]
        print(image.shape)
        annotated, hand_result = self.hand_model.draw_annotation(image, None)
        return annotated