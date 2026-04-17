from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCH_DIR = PROJECT_ROOT / "docs" / "architecture"
OUTPUT = ARCH_DIR / "model_layered_architecture_hierarchical_v17_20260416.png"

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]

PALETTE = {
    "bg": "#F7FAFC",
    "frame": "#D8E1EB",
    "ink": "#153047",
    "muted": "#46576B",
    "arrow": "#6B7F93",
    "input_fill": "#EEF6FF",
    "input_line": "#5B8DEF",
    "control_fill": "#F4F5FF",
    "control_line": "#7C8CE5",
    "prep_fill": "#F3FBF5",
    "prep_line": "#52B276",
    "model_fill": "#FFF8EA",
    "model_line": "#E0A12E",
    "result_fill": "#F6FAFC",
    "result_line": "#7A94A8",
    "output_fill": "#FDF3F8",
    "output_line": "#D66A9B",
    "support_fill": "#F8F9FB",
    "support_line": "#A8B6C5",
    "card_fill": "#FFFFFF",
    "card_line": "#D6DEE8",
    "accent_fill": "#FFFDF6",
    "accent_line": "#E6D9A9",
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
    return lines or [""]


def draw_center_text(draw, box, text, font, fill) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x1, y1, x2, y2 = box
    draw.text((x1 + (x2 - x1 - w) / 2, y1 + (y2 - y1 - h) / 2), text, font=font, fill=fill)


def draw_card(draw, box, title, subtitle, *, outline, fill, title_font, body_font) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=24, fill=fill, outline=outline, width=3)
    if not subtitle:
        draw_center_text(draw, box, title, title_font, PALETTE["ink"])
        return
    draw.text((x1 + 18, y1 + 12), title, font=title_font, fill=PALETTE["ink"])
    draw.line((x1 + 16, y1 + 42, x2 - 16, y1 + 42), fill=outline, width=1)
    lines = wrap_text(draw, subtitle, body_font, x2 - x1 - 36)
    y = y1 + 50
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=body_font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        draw.text((x1 + (x2 - x1 - w) / 2, y), line, font=body_font, fill=PALETTE["ink"])
        y += h + 4


def draw_pill(draw, box, text, *, font) -> None:
    draw.rounded_rectangle(box, radius=18, fill=PALETTE["card_fill"], outline=PALETTE["card_line"], width=2)
    draw_center_text(draw, box, text, font, PALETTE["ink"])


def draw_layer(draw, box, title, *, fill, outline, font) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=34, fill=fill, outline=outline, width=3)
    draw.text((x1 + 24, y1 + 16), title, font=font, fill=PALETTE["ink"])
    draw.line((x1 + 20, y1 + 52, x2 - 20, y1 + 52), fill=outline, width=1)


def draw_down_arrow(draw, x, y1, y2, *, color=None, width=8) -> None:
    color = color or PALETTE["arrow"]
    draw.line((x, y1, x, y2), fill=color, width=width)
    head = 16
    draw.polygon([(x, y2 + 8), (x - head, y2 - head), (x + head, y2 - head)], fill=color)


def draw_right_arrow(draw, x1, x2, y, *, color=None, width=4) -> None:
    color = color or PALETTE["arrow"]
    draw.line((x1, y, x2, y), fill=color, width=width)
    head = 12
    draw.polygon([(x2 + 6, y), (x2 - head, y - head // 2), (x2 - head, y + head // 2)], fill=color)


def draw_polyline_arrow(draw, points, *, color=None, width=4) -> None:
    color = color or PALETTE["arrow"]
    for p1, p2 in zip(points, points[1:]):
        draw.line((p1[0], p1[1], p2[0], p2[1]), fill=color, width=width)
    if len(points) < 2:
        return
    sx, sy = points[-2]
    ex, ey = points[-1]
    head = 10
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


def main() -> None:
    ARCH_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 2100, 1720
    image = Image.new("RGB", (width, height), PALETTE["bg"])
    draw = ImageDraw.Draw(image)

    title_font = load_font(42, bold=True)
    layer_font = load_font(28, bold=True)
    card_title = load_font(20, bold=True)
    card_body = load_font(14)
    support_font = load_font(18)
    note_font = load_font(18, bold=True)

    draw.rounded_rectangle((24, 24, width - 24, height - 24), radius=40, outline=PALETTE["frame"], width=3)
    draw.text((84, 60), "模型端架构图", font=title_font, fill=PALETTE["ink"])

    left, right = 120, 1980
    y = 140
    layer_gap = 38
    layer_specs = [
        ("输入层", PALETTE["input_fill"], PALETTE["input_line"], 160),
        ("控制层", PALETTE["control_fill"], PALETTE["control_line"], 160),
        ("准备层", PALETTE["prep_fill"], PALETTE["prep_line"], 190),
        ("模型层", PALETTE["model_fill"], PALETTE["model_line"], 330),
        ("结果层", PALETTE["result_fill"], PALETTE["result_line"], 190),
        ("输出层", PALETTE["output_fill"], PALETTE["output_line"], 160),
    ]
    layers = []
    for title, fill, outline, h in layer_specs:
        box = (left, y, right, y + h)
        draw_layer(draw, box, title, fill=fill, outline=outline, font=layer_font)
        layers.append((box, outline))
        y += h + layer_gap

    # Input layer
    ly = layers[0][0][1] + 68
    input_boxes = [
        draw_card(draw, (150, ly, 450, ly + 72), "单张图片", "", outline=PALETTE["input_line"], fill=PALETTE["card_fill"], title_font=card_title, body_font=card_body),
        draw_card(draw, (520, ly, 820, ly + 72), "视频输入", "", outline=PALETTE["input_line"], fill=PALETTE["card_fill"], title_font=card_title, body_font=card_body),
        draw_card(draw, (890, ly, 1190, ly + 72), "实时流输入", "", outline=PALETTE["input_line"], fill=PALETTE["card_fill"], title_font=card_title, body_font=card_body),
        draw_card(draw, (1320, ly, 1760, ly + 72), "模型状态", "", outline=PALETTE["input_line"], fill=PALETTE["card_fill"], title_font=card_title, body_font=card_body),
    ]

    # Control layer
    ly = layers[1][0][1] + 68
    control_boxes = [
        (180, ly, 500, ly + 78),
        (600, ly, 930, ly + 78),
        (1030, ly, 1450, ly + 92),
        (1540, ly, 1820, ly + 78),
    ]
    draw_card(draw, control_boxes[0], "输入检查", "", outline=PALETTE["control_line"], fill=PALETTE["card_fill"], title_font=card_title, body_font=card_body)
    draw_card(draw, control_boxes[1], "会话管理", "", outline=PALETTE["control_line"], fill=PALETTE["card_fill"], title_font=card_title, body_font=card_body)
    draw_card(draw, control_boxes[2], "推理调度", "把准备、检测、结果整理串起来", outline=PALETTE["control_line"], fill=PALETTE["card_fill"], title_font=card_title, body_font=card_body)
    draw_card(draw, control_boxes[3], "模型准备", "", outline=PALETTE["control_line"], fill=PALETTE["card_fill"], title_font=card_title, body_font=card_body)
    draw_right_arrow(draw, 500, 600, ly + 39, color=PALETTE["control_line"])
    draw_right_arrow(draw, 930, 1030, ly + 39, color=PALETTE["control_line"])

    # Prep layer
    ly = layers[2][0][1] + 68
    prep_specs = [
        ((130, ly, 360, ly + 80), "读入图像", ""),
        ((410, ly, 640, ly + 80), "模态对齐", ""),
        ((690, ly, 950, ly + 104), "暗光处理", "CLAHE + Gamma\n弱光场景更稳"),
        ((1010, ly, 1310, ly + 104), "尺寸统一", "Resize / Letterbox"),
        ((1370, ly, 1680, ly + 104), "转成张量", "统一成 6通道输入"),
    ]
    for box, title, subtitle in prep_specs:
        draw_card(draw, box, title, subtitle, outline=PALETTE["prep_line"], fill=PALETTE["card_fill"], title_font=card_title, body_font=card_body)
    draw_right_arrow(draw, 360, 410, ly + 40, color=PALETTE["prep_line"])
    draw_right_arrow(draw, 640, 690, ly + 40, color=PALETTE["prep_line"])
    draw_right_arrow(draw, 950, 1010, ly + 52, color=PALETTE["prep_line"])
    draw_right_arrow(draw, 1310, 1370, ly + 52, color=PALETTE["prep_line"])

    # Model layer
    ly = layers[3][0][1] + 78
    rgb_box = (170, ly, 420, ly + 84)
    thermal_box = (170, ly + 112, 420, ly + 196)
    fusion_box = (520, ly + 52, 780, ly + 160)
    bifpn_box = (850, ly + 52, 1080, ly + 160)
    refine_box = (1140, ly + 52, 1410, ly + 160)
    align_box = (1480, ly + 52, 1690, ly + 160)
    head_box = (1740, ly + 52, 1910, ly + 160)
    draw_card(draw, rgb_box, "RGB 分支", "纹理 / 外观信息", outline=PALETTE["model_line"], fill=PALETTE["card_fill"], title_font=card_title, body_font=card_body)
    draw_card(draw, thermal_box, "热红外分支", "热目标信息", outline=PALETTE["model_line"], fill=PALETTE["card_fill"], title_font=card_title, body_font=card_body)
    draw_card(draw, fusion_box, "融合模块", "注意力融合\n双路并行后再融合", outline=PALETTE["model_line"], fill=PALETTE["card_fill"], title_font=card_title, body_font=card_body)
    draw_card(draw, bifpn_box, "多尺度增强", "BiFPN\n兼顾大小目标", outline=PALETTE["model_line"], fill=PALETTE["card_fill"], title_font=card_title, body_font=card_body)
    draw_card(draw, refine_box, "小目标加强", "精炼头\n更适合无人机视角", outline=PALETTE["model_line"], fill=PALETTE["card_fill"], title_font=card_title, body_font=card_body)
    draw_card(draw, align_box, "尺度对齐", "适配检测头", outline=PALETTE["model_line"], fill=PALETTE["card_fill"], title_font=card_title, body_font=card_body)
    draw_card(draw, head_box, "检测头", "分类 + 回归", outline=PALETTE["model_line"], fill=PALETTE["card_fill"], title_font=card_title, body_font=card_body)
    draw_polyline_arrow(draw, [(420, ly + 42), (490, ly + 42), (490, ly + 88), (520, ly + 88)], color=PALETTE["model_line"], width=3)
    draw_polyline_arrow(draw, [(420, ly + 154), (490, ly + 154), (490, ly + 124), (520, ly + 124)], color=PALETTE["model_line"], width=3)
    draw_right_arrow(draw, 780, 850, ly + 106, color=PALETTE["model_line"])
    draw_right_arrow(draw, 1080, 1140, ly + 106, color=PALETTE["model_line"])
    draw_right_arrow(draw, 1410, 1480, ly + 106, color=PALETTE["model_line"])
    draw_right_arrow(draw, 1690, 1740, ly + 106, color=PALETTE["model_line"])

    # Result layer
    ly = layers[4][0][1] + 68
    result_specs = [
        ((160, ly, 390, ly + 104), "分数筛选", "先去掉低分结果"),
        ((460, ly, 720, ly + 104), "无效框过滤", "去掉面积异常框"),
        ((800, ly, 1060, ly + 104), "重复框处理", "NMS / 合并"),
        ((1140, ly, 1380, ly + 104), "坐标还原", "映射回原图"),
        ((1460, ly, 1780, ly + 104), "结果整理", "便于比赛展示和联调"),
    ]
    for box, title, subtitle in result_specs:
        draw_card(draw, box, title, subtitle, outline=PALETTE["result_line"], fill=PALETTE["card_fill"], title_font=card_title, body_font=card_body)
    draw_right_arrow(draw, 390, 460, ly + 52, color=PALETTE["result_line"])
    draw_right_arrow(draw, 720, 800, ly + 52, color=PALETTE["result_line"])
    draw_right_arrow(draw, 1060, 1140, ly + 52, color=PALETTE["result_line"])
    draw_right_arrow(draw, 1380, 1460, ly + 52, color=PALETTE["result_line"])

    # Output layer
    ly = layers[5][0][1] + 68
    output_specs = [
        ((180, ly, 450, ly + 78), "直接返回", ""),
        ((570, ly, 840, ly + 78), "回调推送", ""),
        ((960, ly, 1280, ly + 78), "状态更新", ""),
        ((1440, ly, 1750, ly + 78), "页面展示", ""),
    ]
    for box, title, subtitle in output_specs:
        draw_card(draw, box, title, subtitle, outline=PALETTE["output_line"], fill=PALETTE["card_fill"], title_font=card_title, body_font=card_body)

    # Main vertical links
    x = 1070
    draw_down_arrow(draw, x, 286, 400)
    draw_down_arrow(draw, x, 585, 770)
    draw_down_arrow(draw, 1825, 610, 830)
    draw_down_arrow(draw, 1620, 986, 1098)

    support_box = (300, 1250, 1600, 1345)
    draw.rounded_rectangle(support_box, radius=26, fill=PALETTE["support_fill"], outline=PALETTE["support_line"], width=2)
    draw.text((support_box[0] + 26, support_box[1] + 16), "支撑模块", font=note_font, fill=PALETTE["ink"])
    support_items = ["推理引擎", "参数配置", "日志监控", "本地预览", "训练评估导出"]
    sx = support_box[0] + 180
    for item in support_items:
        draw_pill(draw, (sx, 1275, sx + 190, 1316), item, font=support_font)
        sx += 205

    image.save(OUTPUT)


if __name__ == "__main__":
    main()
