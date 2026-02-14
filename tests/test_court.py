"""Tests for court analysis module."""

import cv2
import numpy as np

from sports_video_parser.court import (
    CourtAnalyzer,
    _dominant_jersey_color,
    _find_best_hoop_contour,
    _find_dominant_cluster,
)
from sports_video_parser.models import HoopPosition


class TestHoopDetection:
    def setup_method(self):
        self.analyzer = CourtAnalyzer()

    def _make_orange_frame(self, x1=900, y1=200, x2=960, y2=230):
        """Create a frame with an orange rectangle (hoop-like)."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 255), -1)
        return frame

    def test_no_hoop_in_black_frame(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = self.analyzer.detect_hoop(frame)
        assert result is None

    def test_accumulator_needs_30_frames_to_confirm(self):
        """Hoop should NOT be confirmed after just 1 frame — needs 30+ candidates."""
        frame = self._make_orange_frame()
        result = self.analyzer.detect_hoop(frame, frame_index=0)
        # Only 1 candidate — not enough to confirm
        assert result is None
        assert self.analyzer._confirmed_hoop is None

    def test_accumulator_confirms_after_30_frames(self):
        """After 30+ consistent frames, hoop should be confirmed."""
        frame = self._make_orange_frame()
        result = None
        for i in range(35):
            result = self.analyzer.detect_hoop(frame, frame_index=i)
        assert result is not None
        assert isinstance(result, HoopPosition)
        assert self.analyzer._confirmed_hoop is not None

    def test_confirmed_hoop_is_permanent(self):
        """Once confirmed, hoop should be returned even for black frames."""
        frame = self._make_orange_frame()
        for i in range(35):
            self.analyzer.detect_hoop(frame, frame_index=i)
        assert self.analyzer._confirmed_hoop is not None

        black = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = self.analyzer.detect_hoop(black, frame_index=100)
        assert result is not None
        assert result == self.analyzer._confirmed_hoop

    def test_scattered_candidates_dont_confirm(self):
        """Candidates at wildly different positions should not form a cluster."""
        analyzer = CourtAnalyzer()
        for i in range(35):
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            # Place orange blob at random positions each frame
            x = 100 + (i * 200) % 1800
            y = 50 + (i * 100) % 400
            cv2.rectangle(frame, (x, y), (x + 60, y + 30), (0, 128, 255), -1)
            analyzer.detect_hoop(frame, frame_index=i)
        # Scattered — should not confirm
        assert analyzer._confirmed_hoop is None


class TestThreePointDistance:
    def test_estimation(self):
        analyzer = CourtAnalyzer()
        hoop = HoopPosition(x=100, y=200, width=50, height=30, side="right")
        distance = analyzer.estimate_three_point_distance(hoop, frame_width=1920)
        assert distance == 1920 * 0.45
        assert distance > 0


class TestTeamClassification:
    def test_two_distinct_teams(self):
        """Players in red vs blue jerseys should be classified into two groups."""
        analyzer = CourtAnalyzer()

        red1 = np.full((200, 100, 3), (0, 0, 200), dtype=np.uint8)
        red2 = np.full((200, 100, 3), (0, 0, 210), dtype=np.uint8)
        blue1 = np.full((200, 100, 3), (200, 0, 0), dtype=np.uint8)
        blue2 = np.full((200, 100, 3), (210, 0, 0), dtype=np.uint8)

        crops = [(1, red1), (2, red2), (3, blue1), (4, blue2)]
        teams = analyzer.classify_teams(crops)

        assert len(teams) == 4
        assert teams[1] == teams[2]
        assert teams[3] == teams[4]
        assert teams[1] != teams[3]

    def test_single_player(self):
        analyzer = CourtAnalyzer()
        crop = np.full((200, 100, 3), (0, 0, 200), dtype=np.uint8)
        teams = analyzer.classify_teams([(1, crop)])
        assert teams == {1: 0}

    def test_empty_crops(self):
        analyzer = CourtAnalyzer()
        teams = analyzer.classify_teams([])
        assert teams == {}


class TestDominantJerseyColor:
    def test_solid_color(self):
        crop = np.full((200, 100, 3), (255, 0, 0), dtype=np.uint8)
        color = _dominant_jersey_color(crop)
        assert color is not None
        assert len(color) == 3

    def test_tiny_crop(self):
        crop = np.zeros((1, 1, 3), dtype=np.uint8)
        color = _dominant_jersey_color(crop)
        assert color is None

    def test_empty_crop(self):
        crop = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        color = _dominant_jersey_color(crop)
        assert color is None


class TestFindBestHoopContour:
    def test_picks_best_contour(self):
        img = np.zeros((500, 500), dtype=np.uint8)
        cv2.rectangle(img, (200, 200), (260, 225), 255, -1)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = _find_best_hoop_contour(contours, frame_width=500)
        # May or may not pass circularity depending on rectangle shape
        # Rectangles have circularity ~0.78 so should pass 0.3 threshold
        assert best is not None

    def test_rejects_tiny_contour(self):
        img = np.zeros((500, 500), dtype=np.uint8)
        cv2.rectangle(img, (200, 200), (202, 202), 255, -1)  # tiny 2x2
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = _find_best_hoop_contour(contours, frame_width=500)
        assert best is None

    def test_no_contours(self):
        best = _find_best_hoop_contour([], frame_width=500)
        assert best is None

    def test_rejects_low_circularity(self):
        """A very thin L-shaped contour should be rejected by circularity check."""
        img = np.zeros((500, 500), dtype=np.uint8)
        # Draw a thin L shape — low circularity
        cv2.line(img, (200, 200), (200, 280), 255, 2)
        cv2.line(img, (200, 280), (260, 280), 255, 2)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = _find_best_hoop_contour(contours, frame_width=500)
        # Thin L-shape should have low circularity and be rejected
        assert best is None


class TestFindDominantCluster:
    def test_tight_cluster(self):
        points = [(100, 200)] * 20 + [(500, 500)] * 5
        center = _find_dominant_cluster(points, max_distance=50.0)
        assert center is not None
        assert abs(center[0] - 100) < 10
        assert abs(center[1] - 200) < 10

    def test_no_dominant_cluster(self):
        """Evenly split points should not confirm."""
        points = [(100, 100)] * 10 + [(500, 500)] * 10
        center = _find_dominant_cluster(points, max_distance=50.0)
        assert center is None

    def test_empty_points(self):
        assert _find_dominant_cluster([], max_distance=50.0) is None
