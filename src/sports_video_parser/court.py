"""Hoop detection, court analysis, and team classification."""

from collections import deque

import cv2
import numpy as np

from sports_video_parser.config import (
    HOOP_ACCUMULATOR_WINDOW,
    HOOP_CIRCULARITY_MIN,
    HOOP_CLUSTER_DISTANCE,
    HOOP_HSV_HIGH,
    HOOP_HSV_LOW,
    HOOP_SEARCH_REGION,
    THREE_POINT_DISTANCE_RATIO,
)
from sports_video_parser.models import HoopPosition


class CourtAnalyzer:
    """Detects hoops, estimates court geometry, and classifies teams."""

    def __init__(self) -> None:
        self._confirmed_hoop: HoopPosition | None = None
        self._candidates: deque[tuple[int, int]] = deque(
            maxlen=HOOP_ACCUMULATOR_WINDOW
        )  # (center_x, center_y) of each candidate detection

    def detect_hoop(self, frame: np.ndarray, frame_index: int = 0) -> HoopPosition | None:
        """Detect a basketball hoop in a frame using orange color filtering.

        Uses a temporal accumulator: collects candidate positions across many
        frames, then clusters them spatially. Once 30+ candidates form a
        dominant cluster, the hoop position is confirmed and locked permanently.
        """
        # If hoop is confirmed, return it immediately (hoops don't move)
        if self._confirmed_hoop is not None:
            return self._confirmed_hoop

        h, w = frame.shape[:2]
        search_region = frame[: int(h * HOOP_SEARCH_REGION), :]

        hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array(HOOP_HSV_LOW, dtype=np.uint8),
            np.array(HOOP_HSV_HIGH, dtype=np.uint8),
        )

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = _find_best_hoop_contour(contours, w)
        if best is None:
            return None

        x, y, cw, ch = cv2.boundingRect(best)
        center_x = x + cw // 2
        center_y = y + ch // 2
        self._candidates.append((center_x, center_y))

        # Try to confirm hoop from accumulated candidates
        if len(self._candidates) >= 30:
            cluster_center = _find_dominant_cluster(
                list(self._candidates), HOOP_CLUSTER_DISTANCE
            )
            if cluster_center is not None:
                cx, cy = cluster_center
                side = "left" if cx < w // 2 else "right"
                self._confirmed_hoop = HoopPosition(
                    x=cx - cw // 2, y=cy - ch // 2,
                    width=cw, height=ch, side=side,
                )
                return self._confirmed_hoop

        return None

    def estimate_three_point_distance(self, hoop_pos: HoopPosition, frame_width: int) -> float:
        """Estimate the pixel distance threshold for the 3-point line."""
        return frame_width * THREE_POINT_DISTANCE_RATIO

    def classify_teams(self, player_crops: list[tuple[int, np.ndarray]]) -> dict[int, int]:
        """Assign players to teams based on jersey color clustering."""
        if len(player_crops) < 2:
            return {tid: 0 for tid, _ in player_crops}

        features = []
        track_ids = []
        for track_id, crop in player_crops:
            color = _dominant_jersey_color(crop)
            if color is not None:
                features.append(color)
                track_ids.append(track_id)

        if len(features) < 2:
            return {tid: 0 for tid in track_ids}

        features_array = np.array(features, dtype=np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, _ = cv2.kmeans(
            features_array, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )

        return {tid: int(labels[i][0]) for i, tid in enumerate(track_ids)}


def _find_dominant_cluster(
    points: list[tuple[int, int]], max_distance: float
) -> tuple[int, int] | None:
    """Find the center of the largest spatial cluster of points.

    Simple greedy clustering: for each point, count how many other points
    are within max_distance. The point with the most neighbors is the
    cluster center. Returns None if no cluster has enough members.
    """
    if not points:
        return None

    best_center = None
    best_count = 0

    for px, py in points:
        count = 0
        sx, sy = 0, 0
        for qx, qy in points:
            dist = ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5
            if dist <= max_distance:
                count += 1
                sx += qx
                sy += qy
        if count > best_count:
            best_count = count
            best_center = (sx // count, sy // count)

    # Require a strict majority of candidates to be in the dominant cluster
    if best_count <= len(points) // 2:
        return None

    return best_center


def _find_best_hoop_contour(contours: list, frame_width: int):
    """Find the contour most likely to be a basketball hoop.

    Filters by area, aspect ratio, and circularity.
    """
    best = None
    best_score = 0

    min_area = (frame_width * 0.01) ** 2
    max_area = (frame_width * 0.15) ** 2

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            continue

        aspect_ratio = w / h
        if not (0.5 <= aspect_ratio <= 4.0):
            continue

        # Circularity check: 4*pi*area / perimeter^2
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < HOOP_CIRCULARITY_MIN:
            continue

        # Score: prefer larger contours with aspect ratio close to 2:1
        aspect_score = 1.0 - abs(aspect_ratio - 2.0) / 2.0
        score = area * max(0.1, aspect_score)

        if score > best_score:
            best_score = score
            best = contour

    return best


def _dominant_jersey_color(crop: np.ndarray) -> np.ndarray | None:
    """Extract the dominant color from a player crop, excluding skin tones."""
    if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
        return None

    h, w = crop.shape[:2]
    torso = crop[h // 4 : h * 3 // 4, w // 4 : w * 3 // 4]
    if torso.size == 0:
        return None

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    skin_low = np.array([0, 20, 70], dtype=np.uint8)
    skin_high = np.array([20, 150, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, skin_low, skin_high)
    non_skin_mask = cv2.bitwise_not(skin_mask)

    pixels = torso[non_skin_mask > 0]
    if len(pixels) < 10:
        pixels = torso.reshape(-1, 3)

    if len(pixels) == 0:
        return None

    return np.mean(pixels, axis=0).astype(np.float32)
