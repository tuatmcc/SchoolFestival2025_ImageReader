import cv2
from cv2.typing import MatLike


class Camera:
    def __init__(self, camera_idx: int = 0):
        self.__camera_idx = camera_idx
        self.__cap = cv2.VideoCapture(self.__camera_idx)

    def readFrame(self) -> MatLike | None:
        ret, frame = self.__cap.read()

        # Unable to read frame
        if not ret:
            return None

        return frame
