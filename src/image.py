import cv2
from cv2.typing import MatLike
import numpy

GRAY_IMG_WINDOW_NAME="GRAY_IMG_WINDOW_NAME"
MASK_IMG_WINDOW_NAME="MASK_IMG_WINDOW_NAME"
SAT_GRAY_IMG_WINDOW_NAME="SAT_GRAY_IMG_WINDOW_NAME"
SAT_MASK_IMG_WINDOW_NAME="SAT_MASK_IMG_WINDOW_NAME"
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
        th: float = 0.8,
        gray_th: float = 0.5,
        saturation_boost: float = 2.0,
        saturation_th: float = 0.2,
        value_boost: float = 1.5,
    ) -> MatLike:

        # --- 1. ヒストグラム（0-255 → 0-63 に縮小） ---
        size = int(256/4)
        b_hist = [0] * size
        g_hist = [0] * size
        r_hist = [0] * size

        for row in frame:
            for pixel in row:
                b_hist[pixel[0] // 4] += 1
                g_hist[pixel[1] // 4] += 1
                r_hist[pixel[2] // 4] += 1

        # --- 2. 最大頻度の bin を取得 ---
        mb = b_hist.index(max(b_hist))
        mg = g_hist.index(max(g_hist))
        mr = r_hist.index(max(r_hist))

        # --- 3. max±10 bin の範囲 ---
        rng = 8
        min_b, max_b = mb - rng, mb + rng
        min_g, max_g = mg - rng, mg + rng
        min_r, max_r = mr - rng, mr + rng

        # --- 4. RGB を bin 化（0〜63） ---
        b_bin = frame[:, :, 0] // 4
        g_bin = frame[:, :, 1] // 4
        r_bin = frame[:, :, 2] // 4

        # --- 5. 3つすべてが範囲内なら mask=1、1つでも外なら mask=0 ---
        mask = (
            (b_bin >= min_b) & (b_bin <= max_b) &
            (g_bin >= min_g) & (g_bin <= max_g) &
            (r_bin >= min_r) & (r_bin <= max_r)
        ).astype("uint8")

        # --- 6. 出力画像を作成（mask=1 の部分だけ元画像表示） ---
        out = numpy.ones_like(frame) * 255  # 全部白で初期化
        out[mask == 0] = frame[mask == 0]   # 範囲内だけ元の色にする

        # 可視化用
        cv2.imshow(MASK_IMG_WINDOW_NAME, mask * 255)

        return out

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

# import cv2
# from cv2.typing import MatLike
# import numpy


# class ImageProcessor:
#     @staticmethod
#     def trim(
#         frame: MatLike,
#         marker_positions: list[tuple[int, int]],
#         width: int,
#         height: int,
#     ) -> MatLike:
#         marker_coords = [
#             marker_positions[0],
#             marker_positions[1],
#             marker_positions[3],
#             marker_positions[2],
#         ]
#         marker_coords = numpy.array(marker_coords, dtype=numpy.float32)
#         target_coords = numpy.array(
#             [[0, 0], [width, 0], [0, height], [width, height]], dtype=numpy.float32
#         )
#         matrix = cv2.getPerspectiveTransform(marker_coords, target_coords)
#         trimmed = cv2.warpPerspective(frame, matrix, (width, height))
#         return trimmed

#     @staticmethod
#     def filter(
#         frame: MatLike,
#         gain: float,
#         bias: float,
#         th: float = 0.1,
#         th_b: float = 0.3,
#         gray_th: float = 0.2,
#         saturation_boost: float = 2.0,
#         value_boost: float = 1.5,
#     ) -> MatLike:
#         image_array = frame.copy().astype(numpy.float32) / 255.0
#         min_filter = numpy.min(image_array, axis=2, keepdims=True)
#         diff = image_array - min_filter  # Remove color cast
#         converted = numpy.clip((diff * gain) + bias, 0, 1)

#         red = (converted[:, :, 0] < th).astype(numpy.bool)
#         green = (converted[:, :, 1] < th).astype(numpy.bool)
#         blue = (converted[:, :, 2] < th_b).astype(numpy.bool)
#         black = numpy.logical_not(
#             (numpy.mean(image_array, axis=2) < gray_th).astype(numpy.bool)
#         )
#         mask = red & green & blue  # Gray(White) area
#         mask = mask & black
#         boosted = numpy.clip(image_array + mask[..., None], 0, 1)
#         hsv = cv2.cvtColor(
#             (boosted * 255).astype(numpy.uint8), cv2.COLOR_RGB2HSV
#         ).astype(numpy.float32)
#         hsv[:, :, 1] *= saturation_boost
#         hsv[:, :, 1] = numpy.clip(hsv[:, :, 1], 0, 255)
#         hsv[:, :, 2] *= value_boost
#         hsv[:, :, 2] = numpy.clip(hsv[:, :, 2], 0, 255)
#         filtered = cv2.cvtColor(hsv.astype(numpy.uint8), cv2.COLOR_HSV2RGB)
#         return filtered

#     @staticmethod
#     def drawMarkers(frame: MatLike, corner: list[tuple[int, int]]) -> MatLike:
#         for i in range(len(corner)):
#             cv2.line(
#                 frame,
#                 corner[i],
#                 corner[(i + 1) % len(corner)],
#                 (0, 255, 0),
#                 2,
#             )
#         return frame
