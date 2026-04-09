from __future__ import annotations

from typing import Dict

import torch


class TorchInferenceEngine:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    def predict(self, rgb_tensor: torch.Tensor, thermal_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            rgb_batch = [rgb_tensor.to(self.device, non_blocking=self.device.type == "cuda")]
            thermal_batch = [thermal_tensor.to(self.device, non_blocking=self.device.type == "cuda")]
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    result = self.model(rgb_batch, thermal_batch)[0]
            else:
                result = self.model(rgb_batch, thermal_batch)[0]
        return {key: value.detach().cpu() for key, value in result.items()}


class ONNXRuntimeEngine:
    def __init__(self, onnx_path: str):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("onnxruntime is required for ONNX inference.") from exc
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def predict(self, combined_tensor: torch.Tensor):
        inputs = {"images": combined_tensor.unsqueeze(0).cpu().numpy()}
        outputs = self.session.run(None, inputs)
        return outputs
