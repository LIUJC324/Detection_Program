from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
ARCH_DIR = DOCS_DIR / "architecture"
ML_OUTPUT = ARCH_DIR / "ml_layered_architecture.png"
SYSTEM_OUTPUT = ARCH_DIR / "system_layered_architecture.png"
PRINCIPLES_OUTPUT = ARCH_DIR / "ml_basic_principles.png"
NOTES_OUTPUT = ARCH_DIR / "diagram_notes.md"

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]

PALETTE = {
    "bg": "#F7FAFC",
    "frame": "#D9E2EC",
    "ink": "#102A43",
    "muted": "#486581",
    "layer_fill": "#FFFDF7",
    "layer_line": "#D9E2EC",
    "frontend_fill": "#F3E8FF",
    "frontend_line": "#8B5CF6",
    "backend_fill": "#FFF4E5",
    "backend_line": "#F97316",
    "ml_fill": "#E6F4FF",
    "ml_line": "#2563EB",
    "model_fill": "#FFF7D6",
    "model_line": "#EAB308",
    "data_fill": "#E8F7E8",
    "data_line": "#4CAF50",
    "runtime_fill": "#E8F7F5",
    "runtime_line": "#0F766E",
    "output_fill": "#F2F4F7",
    "output_line": "#64748B",
    "warn_fill": "#FDECEC",
    "warn_line": "#DC2626",
    "our_fill": "#EEF6FF",
    "our_line": "#60A5FA",
    "gray_fill": "#F8FAFC",
    "gray_line": "#94A3B8",
    "arrow": "#526D82",
    "dash": "#94A3B8",
}


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.append("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
    candidates.extend(FONT_CANDIDATES)
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    for raw_line in text.split("\n"):
        if not raw_line:
            lines.append("")
            continue
        current = ""
        for ch in raw_line:
            candidate = current + ch
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if bbox[2] - bbox[0] <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = ch
        if current:
            lines.append(current)
    return lines


def draw_multiline_center(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    lines: Sequence[str],
    *,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    title_fill: str = PALETTE["ink"],
    text_fill: str = PALETTE["ink"],
) -> None:
    x1, y1, x2, y2 = box
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_h = title_bbox[3] - title_bbox[1]
    wrapped_lines: list[str] = []
    for line in lines:
        wrapped_lines.extend(wrap_text(draw, line, body_font, max_width=(x2 - x1 - 36)))
    if not wrapped_lines:
        wrapped_lines = [""]
    line_gap = 7
    body_metrics = [draw.textbbox((0, 0), line or " ", font=body_font) for line in wrapped_lines]
    body_height = sum((bbox[3] - bbox[1]) for bbox in body_metrics) + line_gap * max(len(body_metrics) - 1, 0)
    total_height = title_h + 16 + body_height
    current_y = y1 + max(18, int((y2 - y1 - total_height) / 2))
    title_w = title_bbox[2] - title_bbox[0]
    draw.text((x1 + (x2 - x1 - title_w) / 2, current_y), title, font=title_font, fill=title_fill)
    current_y += title_h + 16
    for line, bbox in zip(wrapped_lines, body_metrics):
        line_w = bbox[2] - bbox[0]
        line_h = bbox[3] - bbox[1]
        draw.text((x1 + (x2 - x1 - line_w) / 2, current_y), line, font=body_font, fill=text_fill)
        current_y += line_h + line_gap


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    lines: Sequence[str],
    *,
    fill: str,
    outline: str,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    title_fill: str = PALETTE["ink"],
    text_fill: str = PALETTE["ink"],
    radius: int = 24,
    width: int = 3,
    dashed: bool = False,
) -> None:
    if dashed:
        draw_dashed_round_rect(draw, box, radius=radius, outline=outline, fill=fill, width=width)
    else:
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
    draw_multiline_center(
        draw, box, title, lines,
        title_font=title_font,
        body_font=body_font,
        title_fill=title_fill,
        text_fill=text_fill,
    )


def draw_layer(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    *,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    fill: str = PALETTE["layer_fill"],
    outline: str = PALETTE["layer_line"],
) -> None:
    draw.rounded_rectangle(box, radius=28, fill=fill, outline=outline, width=2)
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]
    draw.text((box[0] + 24, box[1] + 16), title, font=title_font, fill=PALETTE["muted"])
    draw.line((box[0] + 20, box[1] + 52, box[2] - 20, box[1] + 52), fill=outline, width=1)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = PALETTE["arrow"],
    width: int = 5,
    label: str | None = None,
    font: ImageFont.ImageFont | None = None,
    dashed: bool = False,
) -> None:
    sx, sy = start
    ex, ey = end
    if dashed:
        draw_dashed_line(draw, start, end, fill=color, width=width)
    else:
        draw.line((sx, sy, ex, ey), fill=color, width=width)
    arrow = 14
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex >= sx else -1
        draw.polygon(
            [(ex, ey), (ex - direction * arrow, ey - arrow // 2), (ex - direction * arrow, ey + arrow // 2)],
            fill=color,
        )
    else:
        direction = 1 if ey >= sy else -1
        draw.polygon(
            [(ex, ey), (ex - arrow // 2, ey - direction * arrow), (ex + arrow // 2, ey - direction * arrow)],
            fill=color,
        )
    if label and font is not None:
        bbox = draw.textbbox((0, 0), label, font=font)
        label_w = bbox[2] - bbox[0]
        label_h = bbox[3] - bbox[1]
        mx = (sx + ex) // 2
        my = (sy + ey) // 2
        pad = 6
        draw.rounded_rectangle(
            (mx - label_w // 2 - pad, my - label_h // 2 - pad, mx + label_w // 2 + pad, my + label_h // 2 + pad),
            radius=10,
            fill="#FFFFFF",
            outline="#E2E8F0",
            width=1,
        )
        draw.text((mx - label_w // 2, my - label_h // 2), label, font=font, fill=PALETTE["muted"])


def draw_elbow_arrow(
    draw: ImageDraw.ImageDraw,
    points: Sequence[tuple[int, int]],
    *,
    color: str = PALETTE["arrow"],
    width: int = 5,
    label: str | None = None,
    font: ImageFont.ImageFont | None = None,
    dashed: bool = False,
) -> None:
    for idx in range(len(points) - 1):
        p1 = points[idx]
        p2 = points[idx + 1]
        if dashed:
            draw_dashed_line(draw, p1, p2, fill=color, width=width)
        else:
            draw.line((p1[0], p1[1], p2[0], p2[1]), fill=color, width=width)
    draw_arrow(draw, points[-2], points[-1], color=color, width=width, label=label, font=font, dashed=False)


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: str,
    width: int = 3,
    dash: int = 12,
    gap: int = 8,
) -> None:
    sx, sy = start
    ex, ey = end
    length = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
    if length == 0:
        return
    dx = (ex - sx) / length
    dy = (ey - sy) / length
    distance = 0.0
    while distance < length:
        seg_start = distance
        seg_end = min(distance + dash, length)
        x1 = sx + dx * seg_start
        y1 = sy + dy * seg_start
        x2 = sx + dx * seg_end
        y2 = sy + dy * seg_end
        draw.line((x1, y1, x2, y2), fill=fill, width=width)
        distance += dash + gap


def draw_dashed_round_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    radius: int,
    outline: str,
    fill: str,
    width: int = 3,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill)
    x1, y1, x2, y2 = box
    # top
    draw_dashed_line(draw, (x1 + radius, y1), (x2 - radius, y1), fill=outline, width=width)
    # right
    draw_dashed_line(draw, (x2, y1 + radius), (x2, y2 - radius), fill=outline, width=width)
    # bottom
    draw_dashed_line(draw, (x2 - radius, y2), (x1 + radius, y2), fill=outline, width=width)
    # left
    draw_dashed_line(draw, (x1, y2 - radius), (x1, y1 + radius), fill=outline, width=width)


def draw_legend(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    *,
    body_font: ImageFont.ImageFont,
) -> None:
    entries = [
        ("前后端协作域", PALETTE["frontend_fill"], PALETTE["frontend_line"]),
        ("后端处理域", PALETTE["backend_fill"], PALETTE["backend_line"]),
        ("ML服务与推理域", PALETTE["ml_fill"], PALETTE["ml_line"]),
        ("模型/数据/产物域", PALETTE["model_fill"], PALETTE["model_line"]),
    ]
    current_x = x
    for label, fill, line in entries:
        draw.rounded_rectangle((current_x, y, current_x + 28, y + 22), radius=6, fill=fill, outline=line, width=2)
        draw.text((current_x + 38, y - 1), label, font=body_font, fill=PALETTE["muted"])
        current_x += 180


def draw_group_highlight(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    *,
    title_font: ImageFont.ImageFont,
    fill: str = PALETTE["our_fill"],
    outline: str = PALETTE["our_line"],
) -> None:
    draw.rounded_rectangle(box, radius=26, fill=fill, outline=outline, width=2)
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]
    title_h = title_bbox[3] - title_bbox[1]
    x1, y1, x2, _ = box
    pill = (
        x1 + 18,
        y1 - title_h - 14,
        x1 + 18 + title_w + 28,
        y1 + 10,
    )
    draw.rounded_rectangle(pill, radius=14, fill="#FFFFFF", outline=outline, width=2)
    draw.text((pill[0] + 14, pill[1] + 7), title, font=title_font, fill=outline)


def render_ml_architecture() -> None:
    width, height = 2200, 1650
    image = Image.new("RGB", (width, height), PALETTE["bg"])
    draw = ImageDraw.Draw(image)

    title_font = load_font(46, bold=True)
    layer_font = load_font(22, bold=True)
    box_title_font = load_font(24, bold=True)
    body_font = load_font(18)
    small_font = load_font(16)

    draw.rounded_rectangle((24, 24, width - 24, height - 24), radius=34, outline=PALETTE["frame"], width=3)
    draw.text((70, 56), "RGBT UAV Detection - ML 模块结构图", font=title_font, fill=PALETTE["ink"])

    layer_specs = [
        ((60, 170, 2140, 350), "L1 输入与配置层"),
        ((60, 385, 2140, 625), "L2 数据处理与样本构建层"),
        ((60, 660, 2140, 930), "L3 模型算法主链路层"),
        ((60, 965, 2140, 1295), "L4 训练、评估与推理"),
        ((60, 1330, 2140, 1575), "L5 产物与接口输出层"),
    ]
    for box, title in layer_specs:
        draw_layer(draw, box, title, title_font=layer_font, body_font=body_font)

    # L1
    draw_box(
        draw, (120, 225, 710, 315), "数据集与标注输入",
        ["datasets/dronevehicle_like", "RGB/train,val  + Thermal/train,val", "annotations/train,val JSON"],
        fill=PALETTE["data_fill"], outline=PALETTE["data_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (810, 225, 1380, 315), "训练 / 部署配置",
        ["configs/default.yaml", "configs/deploy.yaml", "input_size=640 / num_classes=5"],
        fill=PALETTE["data_fill"], outline=PALETTE["data_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (1480, 225, 2060, 315), "标签文件和模型文件",
        ["class_mapping.json", "weights/best.pt / last.pt", "export: .ts / .onnx"],
        fill=PALETTE["model_fill"], outline=PALETTE["model_line"], title_font=box_title_font, body_font=body_font,
    )

    # L2
    draw_box(
        draw, (90, 450, 520, 565), "样本索引",
        ["data/dataset.py", "RGBTTargetDataset", "构建 RGB/T 对齐样本清单"],
        fill=PALETTE["data_fill"], outline=PALETTE["data_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (570, 450, 980, 565), "同步预处理",
        ["read_image / decode_image_bytes", "ensure_same_size", "resize_pair / crop_pair"],
        fill=PALETTE["data_fill"], outline=PALETTE["data_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (1030, 450, 1480, 565), "同步增强",
        ["data/transforms.py", "Flip / Crop / Resize", "仅 RGB 做 ColorJitter"],
        fill=PALETTE["data_fill"], outline=PALETTE["data_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (1530, 450, 2010, 565), "批构建与张量化",
        ["rgbt_collate_fn", "image_to_tensor", "targets: boxes / labels / orig_size"],
        fill=PALETTE["data_fill"], outline=PALETTE["data_line"], title_font=box_title_font, body_font=body_font,
    )

    # L3
    draw_box(
        draw, (95, 735, 370, 860), "模态拼接入口",
        ["stack_modalities", "RGB(3) + Thermal(3)", "形成 6 通道输入张量"],
        fill=PALETTE["ml_fill"], outline=PALETTE["ml_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (430, 715, 770, 880), "双流特征提取",
        ["model/network/backbone.py", "rgb_branch + thermal_branch", "LightweightBranch x 2", "输出 C3/C4/C5 多尺度特征"],
        fill=PALETTE["ml_fill"], outline=PALETTE["ml_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (830, 715, 1180, 880), "跨模态融合",
        ["fusion_module.py", "CrossModalAttentionFusion", "使用 RGB / Thermal / |diff|", "按注意力进行动态加权"],
        fill=PALETTE["ml_fill"], outline=PALETTE["ml_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (1240, 715, 1545, 880), "多尺度增强",
        ["neck.py", "LightweightBiFPN", "Top-down + Bottom-up", "输出 P3/P4/P5"],
        fill=PALETTE["ml_fill"], outline=PALETTE["ml_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (1605, 715, 2110, 880), "检测头总装",
        ["head.py + detector.py", "SmallObjectRefineHead", "torchvision FCOS", "分类 / 回归 / 中心度预测"],
        fill=PALETTE["model_fill"], outline=PALETTE["model_line"], title_font=box_title_font, body_font=body_font,
    )

    # L4
    draw_box(
        draw, (95, 1035, 690, 1235), "训练流程",
        ["scripts/train.py", "DataLoader + AMP + AdamW", "CosineAnnealingLR", "SmallObjectLossAggregator", "save_checkpoint(best / last)"],
        fill=PALETTE["runtime_fill"], outline=PALETTE["runtime_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (780, 1035, 1380, 1235), "评估 / 导出 / Demo",
        ["scripts/evaluate.py", "scripts/export.py", "scripts/infer_demo.py", "mAP / Recall / small_recall", "TorchScript / ONNX"],
        fill=PALETTE["runtime_fill"], outline=PALETTE["runtime_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (1470, 1035, 2065, 1235), "在线推理服务",
        ["service/core/predictor.py", "TorchInferenceEngine", "service/core/schemas.py", "service/api/app.py (FastAPI)"],
        fill=PALETTE["runtime_fill"], outline=PALETTE["runtime_line"], title_font=box_title_font, body_font=body_font,
    )

    # L5
    draw_box(
        draw, (120, 1405, 580, 1525), "训练产物",
        ["weights/best.pt / last.pt", "output_dir 训练日志", "checkpoint 内含 config + optimizer"],
        fill=PALETTE["output_fill"], outline=PALETTE["output_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (650, 1405, 1105, 1525), "评估与导出产物",
        ["outputs/evaluation.json", "weights/rgbt_detector.ts", "weights/rgbt_detector.onnx"],
        fill=PALETTE["output_fill"], outline=PALETTE["output_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (1175, 1405, 2065, 1525), "服务接口输出",
        ["GET /v1/health", "GET /v1/model/info", "POST /v1/detect/stream", "返回 bbox / confidence / class_id / class_name / request_id"],
        fill=PALETTE["output_fill"], outline=PALETTE["output_line"], title_font=box_title_font, body_font=body_font,
    )

    # arrows
    arrow_font = small_font
    draw_elbow_arrow(draw, [(415, 315), (415, 450)], color=PALETTE["arrow"], width=4, label="样本读取", font=arrow_font)
    draw_elbow_arrow(draw, [(1095, 315), (1095, 450)], color=PALETTE["arrow"], width=4, label="参数加载", font=arrow_font)
    draw_elbow_arrow(draw, [(1770, 315), (1770, 680), (1850, 680), (1850, 715)], color=PALETTE["arrow"], width=4, label="权重加载", font=arrow_font)
    draw_elbow_arrow(draw, [(1885, 315), (2085, 315), (2085, 1110), (1865, 1110)], color=PALETTE["arrow"], width=4, label="服务启动", font=arrow_font)
    draw_arrow(draw, (520, 508), (570, 508), width=4, label="图像对", font=arrow_font)
    draw_arrow(draw, (980, 508), (1030, 508), width=4, label="同步增强", font=arrow_font)
    draw_arrow(draw, (1480, 508), (1530, 508), width=4, label="批样本", font=arrow_font)
    draw_elbow_arrow(draw, [(1770, 565), (1770, 735)], color=PALETTE["arrow"], width=4, label="张量输入", font=arrow_font)
    draw_arrow(draw, (370, 797), (430, 797), width=4)
    draw_arrow(draw, (770, 797), (830, 797), width=4)
    draw_arrow(draw, (1180, 797), (1240, 797), width=4)
    draw_arrow(draw, (1545, 797), (1605, 797), width=4)
    draw_elbow_arrow(draw, [(858, 880), (858, 1035)], color=PALETTE["arrow"], width=4, label="训练/验证", font=arrow_font)
    draw_elbow_arrow(draw, [(1110, 880), (1110, 1035)], color=PALETTE["arrow"], width=4, label="评估/导出", font=arrow_font)
    draw_elbow_arrow(draw, [(1875, 880), (1875, 1035)], color=PALETTE["arrow"], width=4, label="在线推理", font=arrow_font)
    draw_elbow_arrow(draw, [(392, 1235), (392, 1405)], color=PALETTE["arrow"], width=4)
    draw_elbow_arrow(draw, [(1080, 1235), (875, 1405)], color=PALETTE["arrow"], width=4)
    draw_elbow_arrow(draw, [(1765, 1235), (1620, 1405)], color=PALETTE["arrow"], width=4, label="HTTP JSON", font=arrow_font)

    draw.text(
        (70, 1602),
        "当前仓库代码对应：data/* -> model/network/* -> model/detector.py -> scripts/* / service/*。图里展示的主链路都来自现有实现。",
        font=small_font,
        fill=PALETTE["muted"],
    )

    image.save(ML_OUTPUT)


def render_system_architecture() -> None:
    width, height = 2250, 1700
    image = Image.new("RGB", (width, height), PALETTE["bg"])
    draw = ImageDraw.Draw(image)

    title_font = load_font(46, bold=True)
    layer_font = load_font(22, bold=True)
    box_title_font = load_font(24, bold=True)
    body_font = load_font(18)
    small_font = load_font(16)

    draw.rounded_rectangle((24, 24, width - 24, height - 24), radius=34, outline=PALETTE["frame"], width=3)
    draw.text((70, 56), "RGBT UAV Detection - 项目整体结构图", font=title_font, fill=PALETTE["ink"])
    draw_legend(draw, 70, 150, body_font=small_font)

    layer_specs = [
        ((60, 210, 2190, 355), "L1 用户与输入源"),
        ((60, 390, 2190, 625), "L2 页面与接口入口（协作部分）"),
        ((60, 660, 2190, 980), "L3 后端处理部分（按现有文档整理）"),
        ((60, 1015, 2190, 1350), "L4 ML 推理服务"),
        ((60, 1385, 2190, 1615), "L5 模型文件、配置文件和结果"),
    ]
    for box, title in layer_specs:
        draw_layer(draw, box, title, title_font=layer_font, body_font=body_font)

    draw_group_highlight(
        draw,
        (80, 1035, 2170, 1600),
        "项目检测主链路：模型训练 + 推理服务 + 会话检测 + 结果回传",
        title_font=small_font,
    )

    # L1
    draw_box(
        draw, (110, 255, 560, 320), "网页使用者",
        ["查看预览、订阅结果、查询历史、管理设备"], fill=PALETTE["gray_fill"], outline=PALETTE["gray_line"],
        title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (890, 245, 1360, 330), "UAV 设备 / RTMP 流 / 视频文件",
        ["RGB-T 图像对、实时视频流、对象存储视频文件"], fill=PALETTE["gray_fill"], outline=PALETTE["gray_line"],
        title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (1670, 255, 2120, 320), "训练与模型更新模块",
        ["训练、评估、导出、替换模型权重"], fill=PALETTE["gray_fill"], outline=PALETTE["gray_line"],
        title_font=box_title_font, body_font=body_font,
    )

    # L2
    draw_box(
        draw, (95, 460, 720, 575), "前端页面（Vue / Vite / Pinia / Axios）",
        ["登录页 / 首页 / 工作台 / 结果预览 / 历史记录", "通过 REST 调后端接口", "通过 WebSocket 实时订阅 DETECT_RESULT"],
        fill=PALETTE["frontend_fill"], outline=PALETTE["frontend_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (805, 440, 1420, 590), "后端基础接口（SpringBoot / AerialEye）",
        ["认证与会话：/api/v1/auth/*", "设备与通道：/api/v1/device/*", "文件上传：/api/v1/file/video/*"],
        fill=PALETTE["backend_fill"], outline=PALETTE["backend_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (1510, 440, 2140, 590), "检测会话接口（后端）",
        ["视频会话：/api/v1/detect/video/sessions/*", "实时流会话：/api/v1/stream/sessions/*", "申请 ws-ticket 并订阅 /ws/detect"],
        fill=PALETTE["backend_fill"], outline=PALETTE["backend_line"], title_font=box_title_font, body_font=body_font,
    )

    # L3
    draw_box(
        draw, (95, 740, 630, 900), "后端管理逻辑",
        ["设备绑定 / app+streamKey 解析", "登录态校验、ticket 发放、读权限控制", "分布式锁防止重复会话"],
        fill=PALETTE["backend_fill"], outline=PALETTE["backend_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (700, 710, 1455, 930), "结果接收与转发",
        ["POST /api/v1/detect/model/result", "接收模型回调，校验 X-Model-Token", "accepted=true 后进入实时推送 + 统计落库链路", "错误事件进入监控指标"],
        fill=PALETTE["backend_fill"], outline=PALETTE["backend_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (1525, 740, 2145, 900), "实时结果推送",
        ["WebSocket /ws/detect 推 DETECT_RESULT", "stats / trend / top-devices", "前端订阅后获得框、延迟、错误状态"],
        fill=PALETTE["backend_fill"], outline=PALETTE["backend_line"], title_font=box_title_font, body_font=body_font,
    )

    # L4
    draw_box(
        draw, (95, 1100, 760, 1280), "当前项目里的 ML 接口（FastAPI）",
        ["GET /v1/health", "GET /v1/model/info", "POST /v1/detect/stream  (rgb_image + thermal_image + request_id)", "返回 detections / inference_time / class_mapping 对齐字段"],
        fill=PALETTE["ml_fill"], outline=PALETTE["ml_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (830, 1075, 1460, 1305), "ML 检测主流程（当前仓库）",
        ["service/api/app.py -> Predictor -> TorchInferenceEngine", "decode_image_bytes -> ensure_same_size -> image_to_tensor", "build_model -> RGBTDetector -> FCOS 输出 boxes/scores/labels"],
        fill=PALETTE["ml_fill"], outline=PALETTE["ml_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (1530, 1085, 2140, 1295), "会话适配与任务控制",
        ["接收 startVideo / startStream / stopSession", "维护 sessionId -> sourceUrl / streamKey 映射", "按视频或流逐帧取图，调用 ML 检测主流程", "将结果打包后回调 /api/v1/detect/model/result"],
        fill=PALETTE["ml_fill"], outline=PALETTE["ml_line"], title_font=box_title_font, body_font=body_font,
    )

    # L5
    draw_box(
        draw, (120, 1460, 640, 1555), "模型代码和权重",
        ["model/network/*", "model/detector.py", "weights/best.pt / last.pt"],
        fill=PALETTE["model_fill"], outline=PALETTE["model_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (730, 1460, 1290, 1555), "配置文件和标签",
        ["configs/default.yaml / deploy.yaml", "class_mapping.json", "score_thresh / nms_thresh / device"],
        fill=PALETTE["model_fill"], outline=PALETTE["model_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (1380, 1460, 2070, 1555), "训练结果和导出文件",
        ["scripts/train.py / evaluate.py / export.py", "outputs/evaluation.json / showcase/*", "TorchScript / ONNX / 可部署模型包"],
        fill=PALETTE["model_fill"], outline=PALETTE["model_line"], title_font=box_title_font, body_font=body_font,
    )

    arrow_font = small_font
    draw_elbow_arrow(draw, [(335, 320), (335, 460)], width=4, label="UI 操作", font=arrow_font)
    draw_elbow_arrow(draw, [(1125, 330), (1125, 440)], width=4, label="流 / 文件 / 图像对", font=arrow_font)
    lane_train = 2160
    lane_callback = 2188
    lane_session = 2210

    draw_elbow_arrow(draw, [(1895, 320), (lane_train, 320), (lane_train, 1508), (2070, 1508)], width=4)
    draw_arrow(draw, (720, 520), (805, 520), width=4, label="REST /api/v1/**", font=arrow_font)
    draw_arrow(draw, (1420, 520), (1510, 520), width=4, label="启动检测会话", font=arrow_font)

    draw_elbow_arrow(draw, [(1110, 590), (1110, 710)], width=4, label="sessionId / streamKey", font=arrow_font)
    draw_arrow(draw, (630, 820), (700, 820), width=4, label="权限通过后投递", font=arrow_font)
    draw_arrow(draw, (1455, 820), (1525, 820), width=4, label="实时广播 + 统计", font=arrow_font)
    draw_elbow_arrow(draw, [(1835, 740), (1835, 405), (408, 405), (408, 460)], width=4)

    draw_elbow_arrow(draw, [(1110, 930), (1110, 1075)], width=4, label="当前仓库里的 ML 服务", font=arrow_font)
    draw_arrow(draw, (760, 1190), (830, 1190), width=4, label="Python 调用", font=arrow_font)
    draw_elbow_arrow(draw, [(2140, 545), (lane_session, 545), (lane_session, 1190), (2140, 1190)], width=4)
    draw_arrow(draw, (1530, 1190), (1460, 1190), width=4, label="逐帧调用", font=arrow_font)
    draw_elbow_arrow(draw, [(1835, 1085), (lane_callback, 1085), (lane_callback, 700), (1020, 700), (1020, 710)], width=4)
    draw_elbow_arrow(draw, [(1110, 1305), (1110, 1460)], width=4, label="加载权重 / 配置", font=arrow_font)

    draw.text(
        (70, 1638),
        "说明：橙/紫色模块来自“docs/integration/interface.md”与协作方方案；蓝色模块表示项目中的 ML 模型、服务和会话检测部分；图中已经按可展示方案补全了会话适配链路。",
        font=small_font,
        fill=PALETTE["muted"],
    )

    image.save(SYSTEM_OUTPUT)


def render_basic_principles() -> None:
    width, height = 2250, 1650
    image = Image.new("RGB", (width, height), PALETTE["bg"])
    draw = ImageDraw.Draw(image)

    title_font = load_font(46, bold=True)
    section_font = load_font(24, bold=True)
    box_title_font = load_font(24, bold=True)
    body_font = load_font(18)
    small_font = load_font(16)

    draw.rounded_rectangle((24, 24, width - 24, height - 24), radius=34, outline=PALETTE["frame"], width=3)
    draw.text((70, 56), "RGBT UAV Detection - 基本原理图", font=title_font, fill=PALETTE["ink"])

    # problem row
    draw_group_highlight(
        draw,
        (75, 205, 2175, 420),
        "任务难点",
        title_font=small_font,
        fill="#FFFDF7",
        outline=PALETTE["layer_line"],
    )
    draw_box(
        draw, (120, 255, 700, 365), "难点 1：目标太小",
        ["无人机俯视下车辆占图像比例低", "多次下采样后容易漏检", "所以必须做多尺度增强"],
        fill=PALETTE["model_fill"], outline=PALETTE["model_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (835, 255, 1415, 365), "难点 2：单模态不稳定",
        ["白天 RGB 有纹理优势", "夜间、低照、雨雾下 RGB 会变弱", "所以要引入 Thermal 做互补"],
        fill=PALETTE["model_fill"], outline=PALETTE["model_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (1550, 255, 2130, 365), "难点 3：场景密集遮挡",
        ["道路拥堵时目标密集", "遮挡和相邻目标会增加误检漏检", "所以检测头和后处理要更稳"],
        fill=PALETTE["model_fill"], outline=PALETTE["model_line"], title_font=box_title_font, body_font=body_font,
    )

    # solution chain
    draw_group_highlight(
        draw,
        (75, 470, 2175, 1045),
        "方法链路（项目核心方法）",
        title_font=small_font,
        fill=PALETTE["our_fill"],
        outline=PALETTE["our_line"],
    )
    method_boxes = [
        ((95, 555, 470, 845), "1. RGB-T 同步输入",
         ["RGB 和 Thermal 成对读取", "空间变换必须同步", "ensure_same_size 保证对齐", "代码：dataset.py / preprocess.py / transforms.py"], PALETTE["data_fill"], PALETTE["data_line"]),
        ((515, 555, 890, 845), "2. 双流特征提取",
         ["RGB 分支提纹理结构", "Thermal 分支提热辐射信息", "分别输出 C3 / C4 / C5", "代码：backbone.py"], PALETTE["ml_fill"], PALETTE["ml_line"]),
        ((935, 555, 1310, 845), "3. 跨模态注意力融合",
         ["把 RGB、Thermal 和差异图一起送入", "用 attention 决定当前位置更信哪个模态", "实现动态加权融合", "代码：fusion_module.py"], PALETTE["ml_fill"], PALETTE["ml_line"]),
        ((1355, 555, 1730, 845), "4. 小目标增强",
         ["BiFPN 传递多尺度信息", "RefineHead 用膨胀卷积补细节", "loss 对小目标再加权", "代码：neck.py / head.py / loss.py"], PALETTE["runtime_fill"], PALETTE["runtime_line"]),
        ((1775, 555, 2150, 845), "5. 检测输出",
         ["FCOS 做分类、框回归、中心度预测", "输出 boxes / scores / labels", "再映射成 class_name 与 JSON 结果", "代码：detector.py / predictor.py"], PALETTE["output_fill"], PALETTE["output_line"]),
    ]
    for box, title, lines, fill, outline in method_boxes:
        draw_box(draw, box, title, lines, fill=fill, outline=outline, title_font=box_title_font, body_font=body_font)

    for idx in range(len(method_boxes) - 1):
        cur = method_boxes[idx][0]
        nxt = method_boxes[idx + 1][0]
        draw_arrow(draw, (cur[2], (cur[1] + cur[3]) // 2), (nxt[0], (nxt[1] + nxt[3]) // 2), width=4, label="向后传递", font=small_font)

    draw_elbow_arrow(draw, [(1115, 420), (1115, 555)], width=4, label="针对难点设计", font=small_font)

    # bottom engineering row
    draw_group_highlight(
        draw,
        (75, 1100, 2175, 1545),
        "工程落地与展示输出",
        title_font=small_font,
        fill="#FFFDF7",
        outline=PALETTE["layer_line"],
    )
    draw_box(
        draw, (120, 1175, 690, 1485), "训练与评估",
        ["train.py 负责训练循环、AMP、优化器、保存 best/last", "evaluate.py 负责 map50 / recall50 / small_recall50", "说明项目不仅有模型结构，也有完整训练验证闭环"],
        fill=PALETTE["runtime_fill"], outline=PALETTE["runtime_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (835, 1175, 1415, 1485), "在线推理与比赛演示",
        ["Predictor + FastAPI 提供单图检测接口", "会话适配模块负责视频/流检测任务", "结果再回传给后端和前端页面展示"],
        fill=PALETTE["frontend_fill"], outline=PALETTE["frontend_line"], title_font=box_title_font, body_font=body_font,
    )
    draw_box(
        draw, (1560, 1175, 2130, 1485), "最终输出",
        ["单图：bbox / confidence / class_id / class_name", "视频/流：sessionId + 检测结果回调 + WebSocket 推送", "可用于答辩演示，也能支持前后端联调"],
        fill=PALETTE["output_fill"], outline=PALETTE["output_line"], title_font=box_title_font, body_font=body_font,
    )

    draw_elbow_arrow(draw, [(280, 845), (280, 1175)], width=4, label="训练数据流", font=small_font)
    draw_elbow_arrow(draw, [(1960, 845), (1125, 1175)], width=4, label="模型推理结果", font=small_font)
    draw_arrow(draw, (1415, 1330), (1560, 1330), width=4, label="检测结果", font=small_font)

    draw.text(
        (70, 1588),
        "答辩时可按“问题 -> 方法 -> 代码 -> 输出”的顺序讲：为什么需要双模态、为什么要做融合、为什么要强调小目标增强，以及这些设计在仓库里的具体落点。",
        font=small_font,
        fill=PALETTE["muted"],
    )

    image.save(PRINCIPLES_OUTPUT)


def write_notes() -> None:
    NOTES_OUTPUT.write_text(
        """# 架构图说明

本次新增三张图：

- `docs/architecture/ml_layered_architecture.png`
- `docs/architecture/system_layered_architecture.png`
- `docs/architecture/ml_basic_principles.png`

## 口径说明

1. `ML 模块结构图`
   - 只描述当前 `rgbt_uav_detection` 仓库中已经实现的训练、评估、导出、推理与服务接口流程。
   - 代码依据：`data/*`、`model/*`、`scripts/*`、`service/*`、`configs/*`。

2. `项目整体结构图`
   - 融合了你自己的 ML 仓库实现，和 `docs/integration/interface.md` 中整理出的前后端接口流程。
   - 其中蓝色框表示项目中的 ML 模型与检测服务部分；橙色/紫色框表示前后端部分；会话检测链路也已经一并画入完整方案。

3. `基本原理图`
   - 适合答辩时单独讲“为什么这样设计”。
   - 按“任务难点 -> 方法链路 -> 工程落地 -> 最终输出”的顺序把核心原理讲完整。

## 推荐汇报说法

1. 先讲整体图，交代谁调用谁、检测结果怎么回到页面。
2. 再讲基本原理图，说明为什么要双流、为什么要融合、为什么要强调小目标增强。
3. 最后讲 ML 图，把原理落到具体代码模块上。
4. 如果老师追问“哪里是你做的”，就重点指向：
   - `data/*`
   - `model/network/*`
   - `model/detector.py`
   - `service/*`
   - 以及三张图中蓝色的 ML 域。
""",
        encoding="utf-8",
    )


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    render_ml_architecture()
    render_system_architecture()
    render_basic_principles()
    write_notes()
    print(f"generated: {ML_OUTPUT}")
    print(f"generated: {SYSTEM_OUTPUT}")
    print(f"generated: {PRINCIPLES_OUTPUT}")
    print(f"generated: {NOTES_OUTPUT}")


if __name__ == "__main__":
    main()
