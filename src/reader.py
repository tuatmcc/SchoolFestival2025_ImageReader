import cv2
from cv2.typing import MatLike
from io import BytesIO
import numpy
import requests

from aruco import ArucoFinder
from camera import Camera
from image import ImageProcessor


FRAME_WINDOW_NAME = "Frame"
TRIMMED_WINDOW_NAME = "Trimmed"


class ImageReader:
    def __init__(
        self, config: dict, camera: Camera, aruco_finder: ArucoFinder, debug: bool
    ) -> None:
        self.__debug = debug
        self.__camera = camera
        self.__finder = aruco_finder
        self.__BLACK_IMAGE = numpy.zeros((600, 800, 3), dtype=numpy.uint8)
        self.__trimmed_image: MatLike | None = None
        self.__db = config.get("endpoints", {}).get("db", "http://localhost:8000")

    def __send_image(self, image: MatLike) -> None:
        if self.__debug:
            print("Debug mode: Skipping image upload")
            return

        print("Uploading image to database...")
        _, img_encoded = cv2.imencode(".jpg", image)
        img_bytes = BytesIO(img_encoded.tobytes())
        _ = requests.post(
            f"{self.__db}/request",
            files={"file": ("trimmed.jpg", img_bytes, "image/jpeg")},
        )
        _ = requests.post(
            f"{self.__db}/upload/raw-image",
            files={"file": ("raw.jpg", img_bytes, "image/jpeg")},
        )

    def run(self) -> None:
        cv2.imshow(TRIMMED_WINDOW_NAME, self.__BLACK_IMAGE)
        while True:
            frame = self.__camera.readFrame()
            if frame is None:  # No frame captured
                continue

            pressing_key = cv2.waitKey(1)

            if pressing_key & 0xFF == ord("q"):
                break
            elif pressing_key & 0xFF == ord("r"):
                cv2.imshow(TRIMMED_WINDOW_NAME, self.__BLACK_IMAGE)
                self.__trimmed_image = None
            elif pressing_key & 0xFF == ord("s") and self.__trimmed_image is not None:
                self.__send_image(self.__trimmed_image)
                cv2.imshow(TRIMMED_WINDOW_NAME, self.__BLACK_IMAGE)
                self.__trimmed_image = None
                continue

            marker_positions = self.__finder.getMarkerPositions(frame)
            if marker_positions is None:  # No markers found
                cv2.imshow(FRAME_WINDOW_NAME, frame)
                continue

            corners = self.__finder.getInnerRect(marker_positions)
            if corners is None:  # Unable to determine inner rectangle
                cv2.imshow(FRAME_WINDOW_NAME, frame)
                continue

            marked_frame = ImageProcessor.drawMarkers(frame.copy(), corners)

            width, height = self.__finder.getSize(corners)
            processed_frame = ImageProcessor.trim(
                frame,
                marker_positions=corners,
                width=width * 2,
                height=height * 2,
            )

            marked_frame = cv2.putText(
                marked_frame,
                "Press 'c' to capture / 'r' to retake",
                (0, 30),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                (0, 128, 0),
                2,
            )

            if pressing_key & 0xFF == ord("c"):
                processed_frame = cv2.putText(
                    processed_frame,
                    "Press 's' to send",
                    (0, 30),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0,
                    (0, 128, 0),
                    2,
                )
                self.__trimmed_image = processed_frame.copy()

            cv2.imshow(FRAME_WINDOW_NAME, marked_frame)

            if self.__trimmed_image is not None:
                cv2.imshow(TRIMMED_WINDOW_NAME, self.__trimmed_image)
            else:
                cv2.imshow(TRIMMED_WINDOW_NAME, processed_frame)
