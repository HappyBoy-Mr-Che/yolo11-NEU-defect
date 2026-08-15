"""检测推理工作线程: 生产者-消费者架构, 推理不阻塞 UI。

- 主线程只负责: 读帧 -> put_frame 入队 (队列满时丢最旧帧, 保最新帧)
- 工作线程负责: 取帧 -> YOLO predict -> emit result_ready
- 画图/写视频/UI 更新在 result_ready 槽函数完成 (Qt 自动跨线程排队回主线程)
- conf/iou 通过 params dict 实时读取, 滑块拖动立即生效
"""
import queue
import threading

from PyQt5.QtCore import QThread, pyqtSignal


class DetectionWorker(QThread):
    result_ready = pyqtSignal(object)  # (frame, pred_result, speed_ms)

    def __init__(self, model, params, parent=None):
        super().__init__(parent)
        self.model = model
        self.params = params  # {"conf": float, "iou": float}, 由主线程更新
        self.frame_queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()

    def put_frame(self, frame):
        """最新帧策略: 队列满时丢最旧帧, 保证界面显示的是最新画面。"""
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def stop(self):
        """清空队列并退出线程 (等待至多 3s)。"""
        self._stop.set()
        try:
            while True:
                self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        self.wait(3000)

    def run(self):
        while not self._stop.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            pred = self.model.predict(source=frame, conf=self.params["conf"],
                                      iou=self.params["iou"], verbose=False)
            speed = (pred[0].speed.get("preprocess", 0.0)
                     + pred[0].speed.get("inference", 0.0)
                     + pred[0].speed.get("postprocess", 0.0))
            self.result_ready.emit((frame, pred[0], speed))
