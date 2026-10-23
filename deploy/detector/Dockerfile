ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
WORKDIR /opt/detector
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
# 复用设备兼容的底层库，不让 pip 替换厂商 PyTorch 或拉取 CUDA 依赖。
RUN python -c "import torch, torchvision, cv2, numpy, fastapi, uvicorn, pydantic"
COPY deploy/detector/requirements.txt /opt/detector/requirements.txt
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation -r requirements.txt
COPY src/navigation_agent/__init__.py /opt/detector/src/navigation_agent/__init__.py
COPY src/navigation_agent/app/__init__.py /opt/detector/src/navigation_agent/app/__init__.py
COPY src/navigation_agent/app/detector /opt/detector/src/navigation_agent/app/detector
COPY src/navigation_agent/contracts/__init__.py /opt/detector/src/navigation_agent/contracts/__init__.py
COPY src/navigation_agent/contracts/detection.py /opt/detector/src/navigation_agent/contracts/detection.py
COPY LICENSE NOTICE THIRD_PARTY_NOTICES.md /opt/detector/
ENV PYTHONPATH=/opt/detector/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    YOLO_AUTOINSTALL=false YOLO_CONFIG_DIR=/tmp/Ultralytics HF_HUB_OFFLINE=1
RUN ln -s /models weights && python -c "import clip, lap; from ultralytics import YOLOWorld; from navigation_agent.app.detector.server import create_app"
EXPOSE 8060
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import json, urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8060/capabilities', timeout=3))['ready']"
ENTRYPOINT ["python", "-m", "navigation_agent.app.detector"]
CMD ["--weights", "/models/yolov8l-worldv2.pt", "--device", "0", "--host", "0.0.0.0", "--port", "8060"]
