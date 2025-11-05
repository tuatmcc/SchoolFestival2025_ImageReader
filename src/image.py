import cv2
from cv2.typing import MatLike
import numpy


class ImageProcessor:
    @staticmethod
    def trim(
        frame: MatLike,
        marker_positions: list[tuple[int, int]],
        width: int,
        height: int,
    ) -> MatLike:
        marker_coords = [
            marker_positions[0],
            marker_positions[1],
            marker_positions[3],
            marker_positions[2],
        ]
        marker_coords = numpy.array(marker_coords, dtype=numpy.float32)
        target_coords = numpy.array(
            [[0, 0], [width, 0], [0, height], [width, height]], dtype=numpy.float32
        )
        matrix = cv2.getPerspectiveTransform(marker_coords, target_coords)
        trimmed = cv2.warpPerspective(frame, matrix, (width, height))
        return trimmed

    @staticmethod
    def filter(frame: MatLike, gain: float, bias: float) -> MatLike:
        return numpy.clip(frame * gain + bias, 0, 255).astype(numpy.uint8)

    @staticmethod
    def drawMarkers(frame: MatLike, corner: list[tuple[int, int]]) -> MatLike:
        for i in range(len(corner)):
            cv2.line(
                frame,
                corner[i],
                corner[(i + 1) % len(corner)],
                (0, 255, 0),
                2,
            )
        return frame
