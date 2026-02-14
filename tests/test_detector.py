"""Tests for YOLO detector module."""

from unittest.mock import MagicMock, patch

from sports_video_parser.detector import Detector, _parse_result, _CLASS_NAMES
from sports_video_parser.models import FrameDetections


class TestParseResult:
    """Test _parse_result with mock YOLO results."""

    def _make_mock_result(self, boxes_data: list[dict] | None = None):
        """Build a mock YOLO result object."""
        result = MagicMock()
        result.orig_shape = (1080, 1920)
        result.orig_fps = 30.0

        if boxes_data is None or len(boxes_data) == 0:
            result.boxes = MagicMock()
            result.boxes.__len__ = lambda self: 0
            return result

        boxes = MagicMock()
        boxes.__len__ = lambda self: len(boxes_data)

        # Build tensor-like mocks for cls, id, xyxy, conf
        cls_items = []
        id_items = []
        xyxy_items = []
        conf_items = []
        for b in boxes_data:
            cls_mock = MagicMock()
            cls_mock.item.return_value = b["cls"]
            cls_items.append(cls_mock)

            id_mock = MagicMock()
            id_mock.item.return_value = b["track_id"]
            id_items.append(id_mock)

            xyxy_mock = MagicMock()
            xyxy_mock.tolist.return_value = b["bbox"]
            xyxy_items.append(xyxy_mock)

            conf_mock = MagicMock()
            conf_mock.item.return_value = b["conf"]
            conf_items.append(conf_mock)

        boxes.cls = cls_items
        boxes.id = id_items
        boxes.xyxy = xyxy_items
        boxes.conf = conf_items

        result.boxes = boxes
        return result

    def test_empty_frame(self):
        result = self._make_mock_result([])
        fd = _parse_result(result, frame_index=0)
        assert isinstance(fd, FrameDetections)
        assert fd.frame_index == 0
        assert fd.detections == []

    def test_single_player(self):
        result = self._make_mock_result([
            {"cls": 0, "track_id": 1, "bbox": [100, 200, 150, 400], "conf": 0.85},
        ])
        fd = _parse_result(result, frame_index=10)
        assert len(fd.detections) == 1
        det = fd.detections[0]
        assert det.class_name == "player"
        assert det.track_id == 1
        assert det.bbox == (100, 200, 150, 400)
        assert det.confidence == 0.85

    def test_player_and_ball(self):
        result = self._make_mock_result([
            {"cls": 0, "track_id": 1, "bbox": [100, 200, 150, 400], "conf": 0.85},
            {"cls": 32, "track_id": 2, "bbox": [300, 100, 320, 120], "conf": 0.6},
        ])
        fd = _parse_result(result, frame_index=5)
        assert len(fd.detections) == 2
        assert fd.detections[0].class_name == "player"
        assert fd.detections[1].class_name == "ball"

    def test_filters_unknown_classes(self):
        result = self._make_mock_result([
            {"cls": 0, "track_id": 1, "bbox": [100, 200, 150, 400], "conf": 0.85},
            {"cls": 15, "track_id": 3, "bbox": [0, 0, 50, 50], "conf": 0.9},  # bench
        ])
        fd = _parse_result(result, frame_index=0)
        assert len(fd.detections) == 1
        assert fd.detections[0].class_name == "player"

    def test_timestamp_calculation(self):
        result = self._make_mock_result([])
        result.orig_fps = 60.0
        fd = _parse_result(result, frame_index=120)
        assert fd.timestamp_sec == 2.0

    def test_no_track_ids(self):
        result = self._make_mock_result([
            {"cls": 0, "track_id": 1, "bbox": [100, 200, 150, 400], "conf": 0.8},
        ])
        result.boxes.id = None
        fd = _parse_result(result, frame_index=0)
        assert fd.detections[0].track_id == -1

    def test_per_class_confidence_filters_low_player(self):
        """A player below player_confidence should be filtered out."""
        result = self._make_mock_result([
            {"cls": 0, "track_id": 1, "bbox": [100, 200, 150, 400], "conf": 0.2},
        ])
        fd = _parse_result(result, frame_index=0, player_confidence=0.3, ball_confidence=0.15)
        assert len(fd.detections) == 0

    def test_per_class_confidence_keeps_low_ball(self):
        """A ball at 0.2 confidence should pass with ball_confidence=0.15."""
        result = self._make_mock_result([
            {"cls": 32, "track_id": 2, "bbox": [300, 100, 320, 120], "conf": 0.2},
        ])
        fd = _parse_result(result, frame_index=0, player_confidence=0.3, ball_confidence=0.15)
        assert len(fd.detections) == 1
        assert fd.detections[0].class_name == "ball"

    def test_per_class_confidence_filters_very_low_ball(self):
        """A ball below ball_confidence should be filtered out."""
        result = self._make_mock_result([
            {"cls": 32, "track_id": 2, "bbox": [300, 100, 320, 120], "conf": 0.1},
        ])
        fd = _parse_result(result, frame_index=0, player_confidence=0.3, ball_confidence=0.15)
        assert len(fd.detections) == 0


class TestDetectorInit:
    """Test Detector initialization."""

    @patch("sports_video_parser.detector.YOLO")
    def test_default_init(self, mock_yolo_cls):
        detector = Detector()
        mock_yolo_cls.assert_called_once_with("yolo11n.pt")
        assert detector.confidence == 0.15
        assert detector.imgsz == 1280
        assert detector.player_confidence == 0.3
        assert detector.ball_confidence == 0.15

    @patch("sports_video_parser.detector.YOLO")
    def test_custom_init(self, mock_yolo_cls):
        detector = Detector(model_name="yolo11s.pt", confidence=0.5, imgsz=640)
        mock_yolo_cls.assert_called_once_with("yolo11s.pt")
        assert detector.confidence == 0.5
        assert detector.imgsz == 640


class TestClassNames:
    def test_class_mapping(self):
        assert _CLASS_NAMES[0] == "player"
        assert _CLASS_NAMES[32] == "ball"
