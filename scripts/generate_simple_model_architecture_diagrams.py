from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCH_DIR = PROJECT_ROOT / "docs" / "architecture"
LAYERED_OUTPUT = ARCH_DIR / "model_layered_architecture_modular_v16_20260416.png"
FLOW_OUTPUT = ARCH_DIR / "model_runtime_flow_simple_v15_20260416.png"

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]

PALETTE = {
    "bg": "#F7FAFC",
    "frame": "#D8E1EB",
    "ink": "#153047",
    "muted": "#5B7083",
    "arrow": "#5E7288",
    "input_fill": "#EEF6FF",
    "input_line": "#5B8DEF",
    "app_fill": "#F5F7FF",
    "app_line": "#7C8CE5",
    "prep_fill": "#F4FBF6",
    "prep_line": "#4EAF72",
    "model_fill": "#FFF8E8",
    "model_line": "#E3A93B",
    "result_fill": "#F7FBFC",
    "result_line": "#7A94A8",
    "output_fill": "#FDF3F8",
    "output_line": "#D66A9B",
    "support_fill": "#F8F9FB",
    "support_line": "#A8B6C5",
    "pill_fill": "#FFFFFF",
    "pill_line": "#D6DEE8",
    "note_fill": "#FFFDF6",
    "note_line": "#E6D9A9",
}


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.append("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
    candidates.extend(FONT_CANDIDATES)
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    for raw in text.split("\n"):
        if not raw:
            lines.append("")
            continue
        current = ""
        for ch in raw:
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


def draw_center_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x1, y1, x2, y2 = box
    draw.text((x1 + (x2 - x1 - w) / 2, y1 + (y2 - y1 - h) / 2), text, font=font, fill=fill)


def draw_box_with_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    *,
    fill: str,
    outline: str,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=30, fill=fill, outline=outline, width=3)
    if not subtitle.strip():
        draw_center_text(draw, box, title, title_font, PALETTE["ink"])
        return
    draw.text((x1 + 28, y1 + 16), title, font=title_font, fill=PALETTE["ink"])
    draw.line((x1 + 24, y1 + 48, x2 - 24, y1 + 48), fill=outline, width=1)
    lines = wrap_text(draw, subtitle, body_font, x2 - x1 - 56)
    line_gap = 8
    heights = [draw.textbbox((0, 0), line or " ", font=body_font)[3] for line in lines]
    total_h = sum(heights) + line_gap * max(0, len(lines) - 1)
    cy = y1 + 56 + max(0, (y2 - (y1 + 56) - total_h) // 2)
    for line in lines:
        bbox = draw.textbbox((0, 0), line or " ", font=body_font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        draw.text((x1 + (x2 - x1 - w) / 2, cy), line, font=body_font, fill=PALETTE["ink"])
        cy += h + line_gap


def draw_pill(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
) -> None:
    draw.rounded_rectangle(box, radius=18, fill=PALETTE["pill_fill"], outline=PALETTE["pill_line"], width=2)
    draw_center_text(draw, box, text, font, PALETTE["ink"])


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    width: int = 6,
) -> None:
    sx, sy = start
    ex, ey = end
    draw.line((sx, sy, ex, ey), fill=PALETTE["arrow"], width=width)
    head = 16
    draw.polygon(
        [
            (ex, ey),
            (ex - head, ey - head),
            (ex + head, ey - head),
        ],
        fill=PALETTE["arrow"],
    )


def draw_right_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = PALETTE["arrow"],
    width: int = 4,
) -> None:
    sx, sy = start
    ex, ey = end
    draw.line((sx, sy, ex, ey), fill=color, width=width)
    head = 12
    draw.polygon(
        [
            (ex, ey),
            (ex - head, ey - head // 2),
            (ex - head, ey + head // 2),
        ],
        fill=color,
    )


def draw_polyline_arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    *,
    color: str = PALETTE["arrow"],
    width: int = 4,
) -> None:
    for p1, p2 in zip(points, points[1:]):
        draw.line((p1[0], p1[1], p2[0], p2[1]), fill=color, width=width)
    if len(points) < 2:
        return
    sx, sy = points[-2]
    ex, ey = points[-1]
    head = 11
    if abs(ex - sx) >= abs(ey - sy):
        if ex >= sx:
            draw.polygon([(ex, ey), (ex - head, ey - head // 2), (ex - head, ey + head // 2)], fill=color)
        else:
            draw.polygon([(ex, ey), (ex + head, ey - head // 2), (ex + head, ey + head // 2)], fill=color)
    else:
        if ey >= sy:
            draw.polygon([(ex, ey), (ex - head // 2, ey - head), (ex + head // 2, ey - head)], fill=color)
        else:
            draw.polygon([(ex, ey), (ex - head // 2, ey + head), (ex + head // 2, ey + head)], fill=color)


def draw_vertical_band(
    draw: ImageDraw.ImageDraw,
    x: int,
    y1: int,
    y2: int,
    *,
    color: str = PALETTE["arrow"],
    width: int = 12,
) -> None:
    draw.line((x, y1, x, y2), fill=color, width=width)
    r = width // 2
    draw.ellipse((x - r, y1 - r, x + r, y1 + r), fill=color, outline=color)
    head = max(12, width + 4)
    draw.polygon(
        [
            (x, y2 + head // 2),
            (x - head, y2 - head),
            (x + head, y2 - head),
        ],
        fill=color,
        outline=color,
    )


def draw_layer(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    modules: Iterable[str],
    *,
    fill: str,
    outline: str,
    title_font: ImageFont.ImageFont,
    pill_font: ImageFont.ImageFont,
    seq_font: ImageFont.ImageFont,
) -> list[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=34, fill=fill, outline=outline, width=3)
    draw.text((x1 + 28, y1 + 20), title, font=title_font, fill=PALETTE["ink"])
    draw.line((x1 + 24, y1 + 62, x2 - 24, y1 + 62), fill=outline, width=1)

    modules = list(modules)
    count = len(modules)
    gap = 18
    margin = 28
    pill_h = 54
    available = (x2 - x1) - margin * 2 - gap * (count - 1)
    pill_w = max(180, available // max(count, 1))
    pill_y = y1 + 86
    pill_boxes: list[tuple[int, int, int, int]] = []
    for idx, module in enumerate(modules):
        px1 = x1 + margin + idx * (pill_w + gap)
        px2 = px1 + pill_w
        pill_box = (px1, pill_y, px2, pill_y + pill_h)
        pill_boxes.append(pill_box)
        draw_pill(draw, pill_box, module, font=pill_font)

        seq_box = (px1 + 8, pill_y - 18, px1 + 40, pill_y + 8)
        draw.rounded_rectangle(seq_box, radius=10, fill="#FFFFFF", outline=outline, width=2)
        draw_center_text(draw, seq_box, str(idx + 1), seq_font, PALETTE["muted"])

    for left_box, right_box in zip(pill_boxes, pill_boxes[1:]):
        sy = (left_box[1] + left_box[3]) // 2
        sx = left_box[2] + 8
        ex = right_box[0] - 10
        draw_right_arrow(draw, (sx, sy), (ex, sy), color=outline, width=3)
    return pill_boxes


def build_layered_diagram() -> None:
    width, height = 2200, 1550
    image = Image.new("RGB", (width, height), PALETTE["bg"])
    draw = ImageDraw.Draw(image)

    title_font = load_font(42, bold=True)
    box_title_font = load_font(22, bold=True)
    box_body_font = load_font(16)
    section_font = load_font(24, bold=True)
    note_title_font = load_font(22, bold=True)

    draw.rounded_rectangle((24, 24, width - 24, height - 24), radius=40, outline=PALETTE["frame"], width=3)
    draw.text((84, 62), "模型端架构图", font=title_font, fill=PALETTE["ink"])

    def module_box(x: int, y: int, w: int, h: int, title: str, body: str, fill: str, outline: str) -> tuple[int, int, int, int]:
        box = (x, y, x + w, y + h)
        draw_box_with_text(
            draw,
            box,
            title,
            body,
            fill=fill,
            outline=outline,
            title_font=box_title_font,
            body_font=box_body_font,
        )
        return box

    # 调用方
    frontend_box = module_box(150, 170, 360, 90, "前端页面", "单图 / 视频 / 实时流调用", PALETTE["input_fill"], PALETTE["input_line"])
    backend_box = module_box(520, 170, 360, 90, "后端服务", "会话控制 / 健康检查 / 结果接收", PALETTE["input_fill"], PALETTE["input_line"])
    service_box = module_box(1040, 145, 520, 140, "模型服务入口", "统一处理请求、加载模型、组织推理流程", PALETTE["app_fill"], PALETTE["app_line"])

    draw_right_arrow(draw, (frontend_box[2] + 18, (frontend_box[1] + frontend_box[3]) // 2), (service_box[0] - 18, service_box[1] + 35), color=PALETTE["arrow"], width=5)
    draw_right_arrow(draw, (backend_box[2] + 18, (backend_box[1] + backend_box[3]) // 2), (service_box[0] - 18, service_box[3] - 35), color=PALETTE["arrow"], width=5)

    # 核心模块行
    entry_group = module_box(120, 420, 420, 300, "调用入口模块", "单图检测\n视频会话\n实时流会话\n模型状态", "#FFFFFF", PALETTE["input_line"])
    prep_group = module_box(620, 420, 420, 300, "数据准备模块", "图像解码\n模态对齐\n暗光处理\nResize / Letterbox\n6通道张量组织", "#FFFFFF", PALETTE["prep_line"])
    model_group = module_box(1120, 370, 500, 400, "检测主链模块", "RGB 分支\n热红外分支\n融合模块\nBiFPN\n小目标加强\n尺度对齐\n检测头", "#FFFFFF", PALETTE["model_line"])
    result_group = module_box(1700, 420, 360, 300, "结果处理模块", "分数筛选\n无效框过滤\n重复框处理\n坐标还原\n结果整理", "#FFFFFF", PALETTE["result_line"])

    draw_arrow(draw, ((service_box[0] + service_box[2]) // 2, service_box[3] + 20), ((entry_group[0] + entry_group[2]) // 2, entry_group[1] - 18), width=8)
    draw_right_arrow(draw, (entry_group[2] + 18, (entry_group[1] + entry_group[3]) // 2), (prep_group[0] - 18, (prep_group[1] + prep_group[3]) // 2), color=PALETTE["arrow"], width=5)
    draw_right_arrow(draw, (prep_group[2] + 18, (prep_group[1] + prep_group[3]) // 2), (model_group[0] - 18, (model_group[1] + model_group[3]) // 2), color=PALETTE["arrow"], width=5)
    draw_right_arrow(draw, (model_group[2] + 18, (model_group[1] + model_group[3]) // 2), (result_group[0] - 18, (result_group[1] + result_group[3]) // 2), color=PALETTE["arrow"], width=5)

    # 模型主链细分
    rgb_box = module_box(1160, 460, 170, 70, "RGB 分支", "纹理 / 外观", "#FFFFFF", PALETTE["model_line"])
    thermal_box = module_box(1160, 560, 170, 70, "热红外分支", "热目标信息", "#FFFFFF", PALETTE["model_line"])
    fusion_box = module_box(1380, 510, 170, 80, "融合模块", "注意力融合", "#FFFFFF", PALETTE["model_line"])
    bifpn_box = module_box(1600, 510, 160, 80, "多尺度增强", "BiFPN", "#FFFFFF", PALETTE["model_line"])
    refine_box = module_box(1800, 510, 160, 80, "小目标加强", "精炼头", "#FFFFFF", PALETTE["model_line"])
    head_box = module_box(1650, 640, 190, 80, "检测头", "OBB 输出", "#FFFFFF", PALETTE["model_line"])

    draw_polyline_arrow(draw, [(rgb_box[2] + 10, (rgb_box[1] + rgb_box[3]) // 2), (1360, (rgb_box[1] + rgb_box[3]) // 2), (1360, fusion_box[1] + 20), (fusion_box[0] - 10, fusion_box[1] + 20)], color=PALETTE["model_line"], width=3)
    draw_polyline_arrow(draw, [(thermal_box[2] + 10, (thermal_box[1] + thermal_box[3]) // 2), (1360, (thermal_box[1] + thermal_box[3]) // 2), (1360, fusion_box[3] - 20), (fusion_box[0] - 10, fusion_box[3] - 20)], color=PALETTE["model_line"], width=3)
    draw_right_arrow(draw, (fusion_box[2] + 10, (fusion_box[1] + fusion_box[3]) // 2), (bifpn_box[0] - 10, (bifpn_box[1] + bifpn_box[3]) // 2), color=PALETTE["model_line"], width=3)
    draw_right_arrow(draw, (bifpn_box[2] + 10, (bifpn_box[1] + bifpn_box[3]) // 2), (refine_box[0] - 10, (refine_box[1] + refine_box[3]) // 2), color=PALETTE["model_line"], width=3)
    draw_polyline_arrow(draw, [(refine_box[2] - 10, refine_box[3] + 10), (refine_box[2] - 10, head_box[1] - 10), ((head_box[0] + head_box[2]) // 2, head_box[1] - 10), ((head_box[0] + head_box[2]) // 2, head_box[1] - 4)], color=PALETTE["model_line"], width=3)

    # 输出与支撑
    output_box = module_box(780, 980, 640, 120, "输出能力", "直接返回\n回调推送\n状态更新\n页面展示", PALETTE["output_fill"], PALETTE["output_line"])
    support_box = (320, 1220, 1560, 1320)
    draw.rounded_rectangle(support_box, radius=26, fill=PALETTE["support_fill"], outline=PALETTE["support_line"], width=2)
    draw.text((support_box[0] + 26, support_box[1] + 16), "支撑模块", font=note_title_font, fill=PALETTE["ink"])
    support_items = ["推理引擎", "参数配置", "日志监控", "本地预览", "训练评估导出"]
    sx = support_box[0] + 190
    sy = support_box[1] + 24
    pill_w = 200
    for item in support_items:
        draw_pill(draw, (sx, sy, sx + pill_w, sy + 42), item, font=load_font(18))
        sx += pill_w + 16

    draw_arrow(draw, ((result_group[0] + result_group[2]) // 2, result_group[3] + 20), ((output_box[0] + output_box[2]) // 2, output_box[1] - 18), width=8)
    draw_polyline_arrow(draw, [((support_box[0] + support_box[2]) // 2, support_box[1] - 12), ((support_box[0] + support_box[2]) // 2, 1180), (1300, 1180), (1300, output_box[3] + 12)], color=PALETTE["support_line"], width=3)

    image.save(LAYERED_OUTPUT)


def build_flow_diagram() -> None:
    width, height = 2400, 1020
    image = Image.new("RGB", (width, height), PALETTE["bg"])
    draw = ImageDraw.Draw(image)

    title_font = load_font(40, bold=True)
    step_title_font = load_font(24, bold=True)
    tag_font = load_font(18, bold=True)

    draw.rounded_rectangle((24, 24, width - 24, height - 24), radius=40, outline=PALETTE["frame"], width=3)
    draw.text((84, 60), "模型端统一推理流程图", font=title_font, fill=PALETTE["ink"])

    steps = [
        ("输入组织", "", "input_fill", "input_line"),
        ("数据准备", "", "prep_fill", "prep_line"),
        ("特征提取", "", "app_fill", "app_line"),
        ("检测加强", "", "model_fill", "model_line"),
        ("结果整理", "", "result_fill", "result_line"),
        ("结果输出", "", "output_fill", "output_line"),
    ]

    left = 90
    top = 260
    box_w = 330
    box_h = 280
    gap = 38
    for idx, (title, subtitle, fill_key, line_key) in enumerate(steps):
        x1 = left + idx * (box_w + gap)
        x2 = x1 + box_w
        box = (x1, top, x2, top + box_h)
        draw_box_with_text(
            draw,
            box,
            title,
            subtitle,
            fill=PALETTE[fill_key],
            outline=PALETTE[line_key],
            title_font=step_title_font,
            body_font=load_font(1),
        )
        tag_box = (x1 + 20, top - 46, x1 + 112, top - 6)
        draw.rounded_rectangle(tag_box, radius=14, fill="#FFFFFF", outline=PALETTE[line_key], width=2)
        draw_center_text(draw, tag_box, f"Step {idx + 1}", tag_font, PALETTE["ink"])

        if idx < len(steps) - 1:
            start = (x2 + 8, top + box_h // 2)
            end = (x2 + gap - 8, top + box_h // 2)
            draw.line((start[0], start[1], end[0], end[1]), fill=PALETTE["arrow"], width=6)
            head = 14
            draw.polygon(
                [(end[0], end[1]), (end[0] - head, end[1] - head // 2), (end[0] - head, end[1] + head // 2)],
                fill=PALETTE["arrow"],
            )

    image.save(FLOW_OUTPUT)


def main() -> None:
    ARCH_DIR.mkdir(parents=True, exist_ok=True)
    build_layered_diagram()
    build_flow_diagram()
    print(f"wrote {LAYERED_OUTPUT}")
    print(f"wrote {FLOW_OUTPUT}")


if __name__ == "__main__":
    main()
