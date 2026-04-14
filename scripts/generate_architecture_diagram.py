from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "docs" / "architecture" / "project_architecture.png"

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.append("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
    candidates.extend(FONT_CANDIDATES)
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    lines: Iterable[str],
    *,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    fill: str,
    outline: str,
    title_fill: str = "#102A43",
    text_fill: str = "#243B53",
) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=28, fill=fill, outline=outline, width=3)

    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]
    title_h = title_bbox[3] - title_bbox[1]
    lines = list(lines)
    line_metrics = [draw.textbbox((0, 0), line, font=body_font) for line in lines]
    line_heights = [bbox[3] - bbox[1] for bbox in line_metrics]
    line_gap = 8
    title_gap = 16
    body_height = sum(line_heights) + line_gap * max(len(lines) - 1, 0)
    total_height = title_h + title_gap + body_height
    start_y = y1 + max(16, int((y2 - y1 - total_height) / 2))

    draw.text((x1 + (x2 - x1 - title_w) / 2, start_y), title, font=title_font, fill=title_fill)

    current_y = start_y + title_h + title_gap
    for line, line_bbox in zip(lines, line_metrics):
        line_bbox = draw.textbbox((0, 0), line, font=body_font)
        line_w = line_bbox[2] - line_bbox[0]
        draw.text((x1 + (x2 - x1 - line_w) / 2, current_y), line, font=body_font, fill=text_fill)
        current_y += (line_bbox[3] - line_bbox[1]) + line_gap


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = "#486581",
    width: int = 5,
) -> None:
    sx, sy = start
    ex, ey = end
    draw.line((sx, sy, ex, ey), fill=color, width=width)
    arrow = 14
    if sy == ey:
        direction = 1 if ex >= sx else -1
        draw.polygon(
            [
                (ex, ey),
                (ex - direction * arrow, ey - arrow // 2),
                (ex - direction * arrow, ey + arrow // 2),
            ],
            fill=color,
        )
    else:
        direction = 1 if ey >= sy else -1
        draw.polygon(
            [
                (ex, ey),
                (ex - arrow // 2, ey - direction * arrow),
                (ex + arrow // 2, ey - direction * arrow),
            ],
            fill=color,
        )


def main() -> None:
    width, height = 1800, 1260
    image = Image.new("RGB", (width, height), "#F7F9FC")
    draw = ImageDraw.Draw(image)

    title_font = load_font(42, bold=True)
    subtitle_font = load_font(22)
    box_title_font = load_font(26, bold=True)
    box_body_font = load_font(18)
    side_title_font = load_font(26, bold=True)
    side_body_font = load_font(19)

    draw.rounded_rectangle((24, 24, width - 24, height - 24), radius=36, outline="#D9E2EC", width=3)
    draw.text((80, 56), "RGBT UAV Detection 项目架构图", font=title_font, fill="#102A43")
    draw.text(
        (82, 112),
        "初始阶段指导版：按当前仓库结构组织，覆盖数据、模型、训练、部署与联调链路",
        font=subtitle_font,
        fill="#486581",
    )

    draw_box(
        draw,
        (520, 165, 1280, 285),
        "项目目标",
        [
            "全天候 UAV 视角 RGB-T 小目标检测",
            "先跑通基线，再补指标、部署与业务联调闭环",
        ],
        title_font=box_title_font,
        body_font=box_body_font,
        fill="#D9EAFD",
        outline="#7FB3D5",
    )

    row_y1 = 360
    row_y2 = 580
    data_box = (80, row_y1, 520, row_y2)
    model_box = (570, row_y1, 1010, row_y2)
    train_box = (1060, row_y1, 1500, row_y2)

    draw_box(
        draw,
        data_box,
        "数据与配置层",
        [
            "configs/default.yaml",
            "class_mapping.json",
            "RGB / Thermal / Annotation",
            "data/dataset.py",
            "data/preprocess.py + transforms.py",
        ],
        title_font=box_title_font,
        body_font=box_body_font,
        fill="#E8F7E8",
        outline="#6FBF73",
    )
    draw_box(
        draw,
        model_box,
        "模型主干层",
        [
            "backbone.py",
            "fusion_module.py",
            "neck.py",
            "head.py",
            "detector.py (FCOS)",
        ],
        title_font=box_title_font,
        body_font=box_body_font,
        fill="#FFF3D6",
        outline="#F0B429",
    )
    draw_box(
        draw,
        train_box,
        "训练与评估层",
        [
            "scripts/train.py",
            "scripts/evaluate.py",
            "scripts/eval_utils.py",
            "outputs/ + weights/",
            "baseline metrics / checkpoints",
        ],
        title_font=box_title_font,
        body_font=box_body_font,
        fill="#FDE8E8",
        outline="#D64545",
    )

    deploy_box = (300, 690, 860, 925)
    service_box = (940, 690, 1500, 925)
    output_box = (520, 1010, 1280, 1175)

    draw_box(
        draw,
        deploy_box,
        "推理与导出层",
        [
            "scripts/infer_demo.py",
            "scripts/export.py",
            "TorchScript / ONNX",
            "部署前性能与结果验证",
        ],
        title_font=box_title_font,
        body_font=box_body_font,
        fill="#E7E3FF",
        outline="#7C5CE6",
    )
    draw_box(
        draw,
        service_box,
        "服务与集成层",
        [
            "service/core/predictor.py",
            "service/utils/inference_engine.py",
            "service/api/app.py",
            "POST /v1/detect/stream",
            "GET /v1/health /v1/model/info",
        ],
        title_font=box_title_font,
        body_font=box_body_font,
        fill="#E3F8F5",
        outline="#2CB1BC",
    )
    draw_box(
        draw,
        output_box,
        "交付目标",
        [
            "可训练、可评估、可导出、可部署",
            "可与 SpringBoot / Vue 做联调",
            "形成后续优化与重训闭环",
        ],
        title_font=box_title_font,
        body_font=box_body_font,
        fill="#F0F4F8",
        outline="#829AB1",
    )

    side_box = (1540, 220, 1730, 980)
    draw.rounded_rectangle(side_box, radius=28, fill="#FFFFFF", outline="#BCCCDC", width=3)
    side_title = "初始阶段重点"
    side_title_bbox = draw.textbbox((0, 0), side_title, font=side_title_font)
    draw.text(
        (side_box[0] + (side_box[2] - side_box[0] - (side_title_bbox[2] - side_title_bbox[0])) / 2, side_box[1] + 24),
        side_title,
        font=side_title_font,
        fill="#102A43",
    )
    focus_lines = [
        "1. 先接入真实数据",
        "2. 做配对和标注质检",
        "3. 跑出第一版 best.pt",
        "4. 建立统一评估口径",
        "5. 再做模块优化",
        "6. 最后服务化联调",
        "",
        "当前仓库现状",
        "- 训练脚本已具备",
        "- 服务接口已具备",
        "- weights/ 为空",
        "- outputs/ 为空",
        "- datasets/ 未接入",
    ]
    current_y = side_box[1] + 88
    for line in focus_lines:
        draw.text((side_box[0] + 20, current_y), line, font=side_body_font, fill="#334E68")
        current_y += 32

    draw_arrow(draw, (900, 285), (900, 340))
    draw_arrow(draw, (520, 470), (550, 470))
    draw_arrow(draw, (1010, 470), (1040, 470))
    draw_arrow(draw, (1280, 580), (1280, 660))
    draw_arrow(draw, (580, 580), (580, 660))
    draw_arrow(draw, (860, 807), (920, 807))
    draw_arrow(draw, (900, 925), (900, 995))

    note_font = load_font(18)
    footer = "Source: configs/, data/, model/, scripts/, service/ in /home/liujuncheng/rgbt_uav_detection"
    footer_bbox = draw.textbbox((0, 0), footer, font=note_font)
    draw.text(
        ((width - (footer_bbox[2] - footer_bbox[0])) / 2, height - 70),
        footer,
        font=note_font,
        fill="#627D98",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_PATH)
    print(f"saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
