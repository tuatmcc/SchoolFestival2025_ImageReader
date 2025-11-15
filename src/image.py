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
    def filter(
        frame: MatLike,
        gain: float,
        bias: float,
        th: float = 0.1,
        gray_th: float = 0.2,
        saturation_boost: float = 2.0,
        value_boost: float = 1.5,
    ) -> MatLike:
        image_array = frame.copy().astype(numpy.float32) / 255.0
        min_filter = numpy.min(image_array, axis=2, keepdims=True)
        diff = image_array - min_filter  # Remove color cast
        converted = numpy.clip((diff * gain) + bias, 0, 1)

        red = (converted[:, :, 0] < th).astype(numpy.bool)
        green = (converted[:, :, 1] < th).astype(numpy.bool)
        blue = (converted[:, :, 2] < th).astype(numpy.bool)
        black = numpy.logical_not(
            (numpy.mean(image_array, axis=2) < gray_th).astype(numpy.bool)
        )
        mask = red & green & blue  # Gray(White) area
        mask = mask & black
        boosted = numpy.clip(image_array + mask[..., None], 0, 1)
        hsv = cv2.cvtColor(
            (boosted * 255).astype(numpy.uint8), cv2.COLOR_RGB2HSV
        ).astype(numpy.float32)
        hsv[:, :, 1] *= saturation_boost
        hsv[:, :, 1] = numpy.clip(hsv[:, :, 1], 0, 255)
        hsv[:, :, 2] *= value_boost
        hsv[:, :, 2] = numpy.clip(hsv[:, :, 2], 0, 255)
        filtered = cv2.cvtColor(hsv.astype(numpy.uint8), cv2.COLOR_HSV2RGB)
        return filtered

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
