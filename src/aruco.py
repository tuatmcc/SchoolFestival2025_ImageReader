import cv2
import numpy
from cv2 import aruco
from cv2.typing import MatLike


Marker = tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]


class ArucoFinder:
    def __init__(self, marker_ids: list[int] = [0, 1, 2, 3]) -> None:
        self.__marker_ids = marker_ids
        self.__detector = aruco.ArucoDetector(
            aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        )

    def getMarkerPositions(self, frame: MatLike) -> list[Marker] | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        markers, ids, _ = self.__detector.detectMarkers(gray)

        # Non existent or incomplete marker set
        if ids is None:
            return None

        # Incomplete marker set
        if len(ids) != len(self.__marker_ids):
            return None

        # Incorrect marker set
        # if sorted(ids.flatten()) != sorted(self.__marker_ids):
        #     return None

        corners: list[Marker] = []
        for marker in markers:
            corner = (
                (int(marker[0][0][0]), int(marker[0][0][1])),
                (int(marker[0][1][0]), int(marker[0][1][1])),
                (int(marker[0][2][0]), int(marker[0][2][1])),
                (int(marker[0][3][0]), int(marker[0][3][1])),
            )
            corners.append(corner)

        return corners

    def orderMarkers(self, corners: numpy.ndarray) -> list[tuple[int, int]]:
        # Order: top-left, top-right, bottom-right, bottom-left
        pts = numpy.array(corners, dtype=numpy.int32)
        s = pts.sum(axis=1)
        diff = numpy.diff(pts, axis=1)

        tl = pts[numpy.argmin(s)]
        br = pts[numpy.argmax(s)]
        tr = pts[numpy.argmin(diff)]
        bl = pts[numpy.argmax(diff)]

        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))

        return [tl, tr, br, bl]

    def getInnerRect(self, corners: list[Marker]) -> list[tuple[int, int]] | None:
        if len(corners) != 4:
            return None

        points = [numpy.array(marker, dtype=numpy.int32) for marker in corners]
        marker_centers = numpy.array(
            [numpy.mean(marker, axis=0) for marker in points], dtype=numpy.float32
        )
        global_center = numpy.mean(marker_centers, axis=0)

        inner_points = []
        for marker in points:
            distances = numpy.linalg.norm(marker - global_center, axis=1)
            inner_point = marker[numpy.argmin(distances)]
            inner_points.append(inner_point)
        inner_points = numpy.array(inner_points, dtype=numpy.float32)

        return self.orderMarkers(inner_points)

    def getSize(self, corners: list[tuple[int, int]]) -> tuple[int, int]:
        if len(corners) != 4:
            return (0, 0)

        tl, tr, br, bl = corners

        width_a = numpy.linalg.norm(numpy.array(br) - numpy.array(bl))
        width_b = numpy.linalg.norm(numpy.array(tr) - numpy.array(tl))
        max_width = int(max(width_a, width_b))

        height_a = numpy.linalg.norm(numpy.array(tr) - numpy.array(br))
        height_b = numpy.linalg.norm(numpy.array(tl) - numpy.array(bl))
        max_height = int(max(height_a, height_b))

        return (max_width, max_height)
