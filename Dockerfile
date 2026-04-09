FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace/rgbt_uav_detection

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/workspace/rgbt_uav_detection
ENV DEPLOY_CONFIG=/workspace/rgbt_uav_detection/configs/deploy.yaml

EXPOSE 8000

CMD ["uvicorn", "service.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

