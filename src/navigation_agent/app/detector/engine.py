"""YOLO-World 与会话独立的 BoT-SORT 跟踪；不接触机器人接口。"""

import base64
import time
from types import SimpleNamespace
from uuid import uuid4

from navigation_agent.contracts.detection import Detection, FrameResponse


class YoloEngine:
    def __init__(self, weights, device):
        import cv2
        import numpy as np
        from ultralytics import YOLOWorld
        from ultralytics.trackers.bot_sort import BOTSORT

        self.cv2, self.np, self.tracker_type = cv2, np, BOTSORT
        self.model = YOLOWorld(weights)
        self.device = device
        self.labels = None
        # 文本编码器在服务启动时准备，不能在机器人任务中隐式安装依赖。
        self.model.set_classes(['object'])
        self.model.predict(np.zeros((270, 480, 3), dtype=np.uint8),
                           device=device, verbose=False)
        self.model_name = weights.rsplit('/', 1)[-1]

    def new_state(self, confidence=.25):
        settings = SimpleNamespace(
            track_high_thresh=confidence, track_low_thresh=min(.1, confidence / 2), new_track_thresh=confidence,
            track_buffer=10, match_thresh=.8, fuse_score=True,
            gmc_method='sparseOptFlow', proximity_thresh=.5, appearance_thresh=.8,
            with_reid=False, model='auto',
        )
        return {'tracker': self.tracker_type(settings),
                'visible': {}, 'stamp': None}

    def detect(self, session, request):
        started = time.monotonic()
        raw = base64.b64decode(request.jpeg_base64, validate=True)
        image = self.cv2.imdecode(self.np.frombuffer(raw, dtype=self.np.uint8), self.cv2.IMREAD_COLOR)
        if image is None or image.size > 1920 * 1080 * 3:
            raise ValueError('需要有效 JPEG，图像不超过 1920×1080')
        state = session['state']
        gap = (request.camera_stamp_ns - (state['stamp'] or request.camera_stamp_ns)) / 1e9
        if gap < 0 or gap > 1.:
            state = session['state'] = self.new_state(session['confidence'])
        state['stamp'] = request.camera_stamp_ns
        labels = session['labels']
        if self.labels != labels:
            self.model.set_classes(labels)
            self.labels = labels[:]
        prediction = self.model.predict(
            image, device=self.device, conf=session['confidence'], imgsz=640,
            max_det=50, verbose=False,
        )[0]
        tracks = state['tracker'].update(prediction.boxes.cpu().numpy(), image)
        # Tracker 丢失后可能沿用旧编号；连续段身份只能由相邻有效帧延续。
        tracked = {int(row[-1]): int(row[4]) for row in tracks}
        visible = {}
        detections = []
        height, width = image.shape[:2]
        boxes = prediction.boxes.xyxy.cpu().numpy()
        annotated = image.copy()
        for index, box in enumerate(boxes):
            track_id = tracked.get(index)
            # 重叠同类框身份不明确，停止继承抑制状态。
            ambiguous = any(
                index != other and int(prediction.boxes.cls[index]) == int(prediction.boxes.cls[other])
                and self._iou(box, other_box) > .5
                for other, other_box in enumerate(boxes)
            )
            continuity = state['visible'].get(track_id) if track_id is not None and not ambiguous else None
            continuity = continuity or uuid4().hex
            if track_id is not None and not ambiguous:
                visible[track_id] = continuity
            coords = [max(0., min(1., float(value / size)))
                      for value, size in zip(box, (width, height, width, height))]
            if coords[0] >= coords[2] or coords[1] >= coords[3]:
                continue
            label = labels[int(prediction.boxes.cls[index])]
            confidence = float(prediction.boxes.conf[index])
            detections.append(Detection(label=label, confidence=confidence, bbox=tuple(coords),
                                        continuity_id=continuity))
            x1, y1, x2, y2 = map(int, box)
            self.cv2.rectangle(annotated, (x1, y1), (x2, y2), (20, 150, 230), 2)
            self.cv2.putText(annotated, label, (x1, max(12, y1 - 5)),
                             self.cv2.FONT_HERSHEY_SIMPLEX, .45, (20, 150, 230), 1)
        state['visible'] = visible
        encoded = None
        if detections:
            ok, data = self.cv2.imencode('.jpg', annotated)
            if ok:
                encoded = base64.b64encode(data).decode('ascii')
        return FrameResponse(
            request_id=request.request_id, camera_stamp_ns=request.camera_stamp_ns,
            width=width, height=height, detections=detections, boxed_jpeg_base64=encoded,
            inference_ms=(time.monotonic() - started) * 1000,
        )

    @staticmethod
    def _iou(a, b):
        intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
        union = (a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - intersection
        return intersection / union if union > 0 else 0.
