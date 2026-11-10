#!/usr/bin/env python3
"""Render a 13-second 16:9 first-person anti-drug film from the gate stills."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
ASSET = ROOT / "assets"
OUT = ROOT / "xintiandi-antidrug.mp4"

W, H = 1920, 1080
FPS = 30
DURATION = 13.0
NFRAMES = int(round(DURATION * FPS))

FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"

# Dolly principal point. Chosen so the aligned street plate expands to full frame.
CX, CY = 960.0, 480.0
Z_FAR = 8.0
Z_WALL = 2.58
CAM_Z_END = Z_FAR / 4.0  # far-plane scale becomes exactly 4/3
STREET_SCALE0 = 0.75
STREET_OX, STREET_OY = 240, 120

HFOV = math.radians(58.0)

QUESTION = (
    "（单选题）1987年6月12日至26日，联合国在维也纳召开由138个国家的3000多名代表参加的"
    "麻醉品滥用和非法贩运问题部长级会议。参会代表一致同意将每年的6月26日定为（      ） ，"
    "即中文语境下的“国际禁毒日”，以引起世界各国对毒品问题的重视，同时号召全球人民共同来解决毒品问题。"
)
OPTIONS = [
    "A.禁止毒品滥用国际日",
    "B.禁止药物滥用和非法贩运国际日",
    "C.禁止药物滥用国际日",
    "D.禁止药物非法贩运国际日",
]
LEFT_TITLE = "巷尾新天地·守护不松懈"
LEFT_BODY = "进入热闹、舒适的公共场所，也要保持自我保护意识"

# Source-pixel polygons (1024x576) of the near gate walls.
LEFT_POLY_SRC = [
    (0, 0), (214, 0), (236, 36), (252, 110), (262, 230), (258, 340),
    (248, 410), (286, 470), (352, 518), (318, 576), (0, 576),
]
RIGHT_POLY_SRC = [
    (792, 0), (1024, 0), (1024, 576), (888, 576), (846, 490),
    (824, 360), (808, 210), (788, 70),
]
# Source-pixel door rectangles (x0, y0, x1, y1), tight to the wood.
DOOR_L_SRC = (125, 24, 204, 568)
DOOR_R_SRC = (944, 2, 1023, 574)


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size)


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if size is not None and im.size != size:
        im = im.resize(size, Image.Resampling.LANCZOS)
    return np.array(im)


def scale_pts(pts) -> np.ndarray:
    s = W / 1024.0
    return np.array([(x * s, y * s) for x, y in pts], np.float32)


def scale_rect(rect) -> tuple[float, float, float, float]:
    s = W / 1024.0
    x0, y0, x1, y1 = rect
    return x0 * s, y0 * s, x1 * s, y1 * s


def ease_in_out(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return 0.5 - 0.5 * math.cos(math.pi * u)


def ease_out(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return 1.0 - (1.0 - u) ** 3


def dolly_progress(t: float) -> float:
    """0 at 2.7s, 1 at 6.5s. Zero start speed, still moving at the end."""
    if t <= 2.7:
        return 0.0
    if t >= 6.5:
        return 1.0
    u = (t - 2.7) / (6.5 - 2.7)
    return u * u * (2.0 - u)


def turn_progress(t: float) -> float:
    """0 at 6.5s, 1 at 9.7s. Speed falls the whole way and ends at rest."""
    if t <= 6.5:
        return 0.0
    if t >= 9.7:
        return 1.0
    u = (t - 6.5) / (9.7 - 6.5)
    return 1.0 - (1.0 - u) ** 1.35


def door_degrees(t: float) -> float:
    if t <= 0.5:
        return 0.0
    if t >= 2.7:
        return 80.0
    return 80.0 * ease_in_out((t - 0.5) / (2.7 - 0.5))


def over(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    sh, sw = src.shape[:2]
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + sw, y0 + sh
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(dst.shape[1], x1), min(dst.shape[0], y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    sx0, sy0 = cx0 - x0, cy0 - y0
    crop = src[sy0 : sy0 + (cy1 - cy0), sx0 : sx0 + (cx1 - cx0)]
    if crop.shape[2] == 3:
        dst[cy0:cy1, cx0:cx1] = crop
        return
    a = crop[:, :, 3:4].astype(np.float32) / 255.0
    rgb = crop[:, :, :3].astype(np.float32)
    base = dst[cy0:cy1, cx0:cx1].astype(np.float32)
    dst[cy0:cy1, cx0:cx1] = np.clip(rgb * a + base * (1.0 - a), 0, 255).astype(np.uint8)


def alpha_composite(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    a = layer[:, :, 3:4].astype(np.float32) / 255.0
    rgb = layer[:, :, :3].astype(np.float32)
    out = rgb * a + base.astype(np.float32) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def warp_plane(img: np.ndarray, z: float, cam_z: float) -> np.ndarray:
    if cam_z <= 1e-4:
        return img.copy()
    scale = z / (z - cam_z)
    mat = np.array(
        [[scale, 0, (1 - scale) * CX], [0, scale, (1 - scale) * CY]], np.float32
    )
    if img.ndim == 3 and img.shape[2] == 4:
        rgb = cv2.warpAffine(
            img[:, :, :3], mat, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        alpha = cv2.warpAffine(
            img[:, :, 3], mat, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        return np.dstack([rgb, alpha])
    return cv2.warpAffine(
        img, mat, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


BLANK = "（      ）"


def atom_width(draw: ImageDraw.ImageDraw, atom: str, face) -> float:
    """The fill-in blank keeps its six spaces, drawn wide enough to read as a gap."""
    if atom == BLANK:
        em = draw.textlength("国", font=face)
        return draw.textlength("（）", font=face) + 6 * em * 0.42
    return draw.textlength(atom, font=face)


def draw_line(draw: ImageDraw.ImageDraw, xy, text: str, face, fill) -> None:
    x, y = xy
    i = 0
    while i < len(text):
        if text.startswith(BLANK, i):
            draw.text((x, y), "（", font=face, fill=fill)
            x += draw.textlength("（", font=face)
            em = draw.textlength("国", font=face)
            step = em * 0.42
            for _ in range(6):
                draw.text((x, y), " ", font=face, fill=fill)
                x += step
            draw.text((x, y), "）", font=face, fill=fill)
            x += draw.textlength("）", font=face)
            i += len(BLANK)
        else:
            draw.text((x, y), text[i], font=face, fill=fill)
            x += draw.textlength(text[i], font=face)
            i += 1


def wrap_text(draw: ImageDraw.ImageDraw, text: str, face, max_w: float) -> list[str]:
    """Character wrap. Keeps the answer blank and leading punctuation intact."""
    atoms: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith(BLANK, i):
            atoms.append(BLANK)
            i += len(BLANK)
        else:
            atoms.append(text[i])
            i += 1
    lines: list[str] = []
    cur: list[str] = []

    def width(seq: list[str]) -> float:
        return sum(atom_width(draw, atom, face) for atom in seq)

    for atom in atoms:
        trial = cur + [atom]
        if width(trial) <= max_w or not cur:
            cur = trial
            continue
        lines.append("".join(cur))
        cur = [atom]
    if cur:
        lines.append("".join(cur))
    return lines


def text_size(draw: ImageDraw.ImageDraw, text: str, face) -> tuple[float, float]:
    box = draw.textbbox((0, 0), text, font=face)
    return box[2] - box[0], box[3] - box[1]


def make_board() -> np.ndarray:
    bw, bh = 860, 930
    im = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    # Soft contact shadow is drawn later on the wall. Frame itself:
    d.rounded_rectangle((0, 0, bw - 1, bh - 1), radius=10, fill=(63, 46, 36, 255))
    d.rounded_rectangle((6, 6, bw - 7, bh - 7), radius=8, fill=(86, 64, 50, 255))
    d.rounded_rectangle((14, 14, bw - 15, bh - 15), radius=4, fill=(243, 238, 228, 255))
    # Title bar
    d.rectangle((14, 14, bw - 15, 108), fill=(122, 36, 40, 255))
    title_font = font(48)
    title = "禁毒宣传栏"
    gap = 10
    widths = [d.textlength(ch, font=title_font) for ch in title]
    total = sum(widths) + gap * (len(title) - 1)
    x = (bw - total) / 2
    for ch, cw in zip(title, widths):
        d.text((x, 32), ch, font=title_font, fill=(247, 241, 230, 255))
        x += cw + gap
    # Slogan
    slogan_font = font(36)
    slogan = "珍爱生命  ·  远离毒品"
    sw, _ = text_size(d, slogan, slogan_font)
    d.text(((bw - sw) / 2, 128), slogan, font=slogan_font, fill=(122, 36, 40, 255))
    d.line((48, 184, bw - 48, 184), fill=(196, 176, 156, 255), width=2)

    head = font(28)
    body = font(26)
    small = font(24)

    def card(x0, y0, x1, y1, heading, lines):
        d.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=(250, 246, 239, 255), outline=(214, 196, 176, 255), width=2)
        d.text((x0 + 22, y0 + 16), heading, font=head, fill=(122, 36, 40, 255))
        y = y0 + 62
        for line in lines:
            d.text((x0 + 22, y), line, font=body, fill=(43, 43, 43, 255))
            y += 40

    card(40, 206, 410, 392, "国际禁毒日", ["每年 6 月 26 日", "提高认识，共同拒毒"])
    card(450, 206, 820, 392, "把住自己", ["绝不尝试", "饮品不离视线", "不明物品不转交"])

    d.text((48, 418), "毒品会造成这些伤害", font=head, fill=(122, 36, 40, 255))
    bullets = [
        "严重损害大脑和身体健康",
        "使人产生依赖，难以自控",
        "破坏家庭关系，诱发违法犯罪",
    ]
    y = 466
    for b in bullets:
        d.ellipse((52, y + 10, 66, y + 24), fill=(122, 36, 40, 255))
        d.text((80, y), b, font=body, fill=(43, 43, 43, 255))
        y += 46

    d.rounded_rectangle((40, 640, 820, 760), radius=8, fill=(250, 246, 239, 255), outline=(214, 196, 176, 255), width=2)
    d.text((62, 658), "公共场所同样要留心", font=head, fill=(122, 36, 40, 255))
    d.text((62, 706), "热闹、舒适的地方，也要守住自己的饮品和随身物品。", font=small, fill=(43, 43, 43, 255))

    d.rectangle((14, 800, bw - 15, bh - 15), fill=(122, 36, 40, 255))
    foot = "发现涉毒线索，立即报警    110"
    fw, _ = text_size(d, foot, head)
    d.text(((bw - fw) / 2, 838), foot, font=head, fill=(247, 241, 230, 255))
    return np.array(im)


def compose_plate(wall: np.ndarray, board: np.ndarray) -> np.ndarray:
    plate = wall.copy()
    bx, by = 448, 58
    bh, bw = board.shape[:2]
    shadow = Image.new("L", (bw + 80, bh + 80), 0)
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle((30, 36, 30 + bw, 36 + bh), radius=12, fill=170)
    shadow = shadow.filter(ImageFilter.GaussianBlur(16))
    sh = np.asarray(shadow).astype(np.float32) / 255.0
    sy, sx = by - 24, bx - 18
    region = plate[sy : sy + sh.shape[0], sx : sx + sh.shape[1]].astype(np.float32)
    region *= (1.0 - 0.45 * sh[..., None])
    plate[sy : sy + sh.shape[0], sx : sx + sh.shape[1]] = region.astype(np.uint8)
    over(plate, board, bx, by)
    return plate


def make_round_panel(width: int, height: int) -> Image.Image:
    im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((8, 10, width - 8, height - 8), radius=16, fill=(70, 48, 32, 70))
    d.rounded_rectangle((0, 0, width - 1, height - 12), radius=14, fill=(251, 246, 238, 255), outline=(140, 48, 52, 255), width=3)
    return im


def make_popups() -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    title_f = font(34)
    body_f = font(28)
    q_f = font(30)
    opt_f = font(28)
    # Left card
    tmp = Image.new("RGB", (10, 10))
    measure = ImageDraw.Draw(tmp)
    left_w = int(round(W * 0.25))
    pad_x, pad_y = 32, 28
    inner = left_w - pad_x * 2
    title_h = 46
    body_lines = wrap_text(measure, LEFT_BODY, body_f, inner)
    body_h = len(body_lines) * 44
    left_h = pad_y + title_h + 12 + body_h + pad_y + 12
    left = make_round_panel(left_w, left_h)
    d = ImageDraw.Draw(left)
    d.text((pad_x, pad_y), LEFT_TITLE, font=title_f, fill=(140, 48, 52, 255))
    y = pad_y + title_h + 8
    for line in body_lines:
        d.text((pad_x, y), line, font=body_f, fill=(46, 46, 46, 255))
        y += 44

    # Right card
    right_w = int(round(W * 0.44))
    pad_x, pad_y = 36, 32
    inner = right_w - pad_x * 2
    q_lines = wrap_text(measure, QUESTION, q_f, inner)
    joined = "".join(q_lines)
    if joined != QUESTION:
        raise SystemExit(f"question wrap lost characters:\n{joined}\n!=\n{QUESTION}")
    q_line_h = 48
    opt_h = 58
    right_h = pad_y + len(q_lines) * q_line_h + 18 + len(OPTIONS) * opt_h + pad_y + 8
    if right_h > H - 48:
        raise SystemExit(f"right panel too tall: {right_h}")
    right = make_round_panel(right_w, right_h)
    d = ImageDraw.Draw(right)
    y = pad_y
    for line in q_lines:
        draw_line(d, (pad_x, y), line, q_f, (46, 46, 46, 255))
        y += q_line_h
    y += 10
    for opt in OPTIONS:
        cy = y + 16
        d.ellipse((pad_x, cy, pad_x + 26, cy + 26), outline=(90, 78, 70, 255), width=2)
        d.text((pad_x + 40, y + 8), opt, font=opt_f, fill=(46, 46, 46, 255))
        y += opt_h

    left_pos = (40, (H - left_h) // 2 + 10)
    right_x = W - 40 - right_w
    right_pos = (right_x, (H - right_h) // 2)
    # Sanity: panels fully on screen
    for name, im, pos in ("left", left, left_pos), ("right", right, right_pos):
        if pos[0] < 0 or pos[1] < 0 or pos[0] + im.size[0] > W or pos[1] + im.size[1] > H:
            raise SystemExit(f"{name} panel does not fit: {pos} {im.size}")
    return np.array(left), np.array(right), left_pos, right_pos


def build_masks(door_l, door_r) -> tuple[np.ndarray, np.ndarray]:
    left = np.zeros((H, W), np.uint8)
    right = np.zeros((H, W), np.uint8)
    cv2.fillPoly(left, [scale_pts(LEFT_POLY_SRC).astype(np.int32)], 255)
    cv2.fillPoly(right, [scale_pts(RIGHT_POLY_SRC).astype(np.int32)], 255)
    for rect in (door_l, door_r):
        x0, y0, x1, y1 = [int(round(v)) for v in rect]
        cv2.rectangle(left, (x0, y0), (x1, y1), 0, -1)
        cv2.rectangle(right, (x0, y0), (x1, y1), 0, -1)
    # Keep the outer silhouette slightly soft. Identical RGB layers hide this at rest.
    left = cv2.GaussianBlur(left, (0, 0), 1.1)
    right = cv2.GaussianBlur(right, (0, 0), 1.1)
    return left, right


def make_dark_opening(h: int, w: int) -> np.ndarray:
    """A recessed doorway: dark, soft, no copied door grain."""
    h = max(2, h)
    w = max(2, w)
    yy = np.linspace(0, 1, h)[:, None]
    xx = np.linspace(0, 1, w)[None, :]
    edge = np.clip(np.minimum(np.minimum(xx, 1 - xx) * 7, np.minimum(yy, 1 - yy) * 5), 0, 1)
    light = (0.42 + 0.58 * edge) * (0.72 + 0.28 * yy)
    img = np.array([26, 20, 16], np.float32) * light[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def pad_left(img: np.ndarray, n: int, strip_w: int = 140) -> np.ndarray:
    if n <= 0:
        return img
    strip = img[:, :strip_w]
    cols = []
    filled = 0
    mirror = True
    while filled < n:
        piece = np.fliplr(strip) if mirror else strip
        room = n - filled
        if piece.shape[1] > room:
            piece = piece[:, -room:]
        cols.append(piece)
        filled += piece.shape[1]
        mirror = not mirror
    cols.reverse()
    return np.concatenate(cols + [img], axis=1)


def pad_right(img: np.ndarray, n: int, strip_w: int = 150) -> np.ndarray:
    if n <= 0:
        return img
    strip = img[:, -strip_w:]
    cols = []
    filled = 0
    mirror = True
    while filled < n:
        piece = np.fliplr(strip) if mirror else strip
        room = n - filled
        if piece.shape[1] > room:
            piece = piece[:, :room]
        cols.append(piece)
        filled += piece.shape[1]
        mirror = not mirror
    return np.concatenate([img] + cols, axis=1)


def pad_top(img: np.ndarray, n: int, strip_h: int = 36) -> np.ndarray:
    if n <= 0:
        return img
    strip = img[:strip_h]
    rows = []
    filled = 0
    mirror = True
    while filled < n:
        piece = np.flipud(strip) if mirror else strip
        room = n - filled
        if piece.shape[0] > room:
            piece = piece[-room:]
        rows.append(piece)
        filled += piece.shape[0]
        mirror = not mirror
    rows.reverse()
    return np.concatenate(rows + [img], axis=0)


def pad_bottom(img: np.ndarray, n: int, strip_h: int = 70) -> np.ndarray:
    if n <= 0:
        return img
    strip = img[-strip_h:]
    rows = []
    filled = 0
    mirror = True
    while filled < n:
        piece = np.flipud(strip) if mirror else strip
        room = n - filled
        if piece.shape[0] > room:
            piece = piece[:room]
        rows.append(piece)
        filled += piece.shape[0]
        mirror = not mirror
    return np.concatenate([img] + rows, axis=0)


def pad_with_edge_brick(img: np.ndarray, left: int, right: int, top: int, bottom: int) -> np.ndarray:
    """Extend a plate using only its brick margins so the sign is not mirrored."""
    out = pad_left(img, left)
    out = pad_right(out, right)
    out = pad_top(out, top)
    out = pad_bottom(out, bottom)
    return np.ascontiguousarray(out)


def door_sprite(photo: np.ndarray, rect) -> np.ndarray:
    x0, y0, x1, y1 = [int(round(v)) for v in rect]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W - 1, x1), min(H - 1, y1)
    return photo[y0:y1, x0:x1].copy()


class Scene:
    def __init__(self) -> None:
        self.photo = load_rgb(ASSET / "gate.jpg", (W, H))
        self.street = load_rgb(ASSET / "street.jpg", (W, H))
        wall = load_rgb(ASSET / "brick_wall_plate.jpg", (W, H))
        self.board = make_board()
        self.plate = compose_plate(wall, self.board)
        self.left_ui, self.right_ui, self.left_pos, self.right_pos = make_popups()
        self.door_l = scale_rect(DOOR_L_SRC)
        self.door_r = scale_rect(DOOR_R_SRC)
        self.mask_l, self.mask_r = build_masks(self.door_l, self.door_r)
        self.wall_l = np.dstack([self.photo, self.mask_l])
        self.wall_r = np.dstack([self.photo, self.mask_r])
        self.sprite_l = door_sprite(self.photo, self.door_l)
        self.sprite_r = door_sprite(self.photo, self.door_r)
        self.far_photo = self._erase_doors(self.photo)
        self.interior_l = make_dark_opening(self.sprite_l.shape[0], self.sprite_l.shape[1])
        self.interior_r = make_dark_opening(self.sprite_r.shape[0], self.sprite_r.shape[1])
        self.street_canvas = self._street_canvas()
        self.fx = (W / 2) / math.tan(HFOV / 2)
        self.fy = self.fx
        self._init_turn()
        self._init_doors()

    def _erase_doors(self, photo: np.ndarray) -> np.ndarray:
        """Replace door leaves on the far plate with real brick, not a stretched column."""
        out = photo.copy()
        # A clean brick patch to the left of the pipe, repeated by a normal resize.
        patch = photo[80:520, 30:150]
        for rect in (self.door_l, self.door_r):
            x0, y0, x1, y1 = [int(round(v)) for v in rect]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            fill = cv2.resize(patch, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
            out[y0:y1, x0:x1] = fill
        return out

    def _street_canvas(self) -> np.ndarray:
        sw, sh = int(round(W * STREET_SCALE0)), int(round(H * STREET_SCALE0))
        small = np.asarray(Image.fromarray(self.street).resize((sw, sh), Image.Resampling.LANCZOS))
        canvas = np.empty_like(self.photo)
        # Edge fill so a partial zoom never shows empty pixels.
        canvas[:] = small[0, 0]
        y1 = min(H, STREET_OY + sh)
        x1 = min(W, STREET_OX + sw)
        canvas[STREET_OY:y1, STREET_OX:x1] = small[: y1 - STREET_OY, : x1 - STREET_OX]
        # Replicate the placed photo to the borders.
        if STREET_OY > 0:
            canvas[:STREET_OY, STREET_OX:x1] = canvas[STREET_OY : STREET_OY + 1, STREET_OX:x1]
        if y1 < H:
            canvas[y1:, STREET_OX:x1] = canvas[y1 - 1 : y1, STREET_OX:x1]
        if STREET_OX > 0:
            canvas[:, :STREET_OX] = canvas[:, STREET_OX : STREET_OX + 1]
        if x1 < W:
            canvas[:, x1:] = canvas[:, x1 - 1 : x1]
        return canvas

    def _init_turn(self) -> None:
        fx, fy = self.fx, self.fy
        dist = 2.75
        depth = 6.0
        half_w = depth * (W / 2) / fx
        # Principal point sits above the image center, so the plane is shifted in Y
        # until its projection is exactly the full frame.
        y_top = CY / fy * depth
        y_bot = (CY - H) / fy * depth
        self.street_quad = np.array(
            [
                [-half_w, y_top, depth],
                [half_w, y_top, depth],
                [half_w, y_bot, depth],
                [-half_w, y_bot, depth],
            ],
            np.float64,
        )
        half_wz = dist * (W / 2) / fx
        y_top_w = CY / fy * dist
        y_bot_w = (CY - H) / fy * dist
        zc = depth - 0.02 - half_wz
        xw = half_w
        self.wall_quad = np.array(
            [
                [xw, y_top_w, zc + half_wz],
                [xw, y_top_w, zc - half_wz],
                [xw, y_bot_w, zc - half_wz],
                [xw, y_bot_w, zc + half_wz],
            ],
            np.float64,
        )
        self.turn_start = (np.array([0.0, 0.0, 0.0]), 0.0)
        self.turn_end = (np.array([xw - dist, 0.0, zc]), math.pi / 2)
        self._build_extensions()

    def _build_extensions(self) -> None:
        """Pad the street and the wall so the turn never opens onto empty space."""
        # Street: more lane to the left and above/below. Reflect is acceptable at the edge.
        sl, st, sb, sr = 860, 420, 460, 80
        self.street_tex = np.pad(self.street, ((st, sb), (sl, sr), (0, 0)), mode="reflect")
        self.street_quad_ext = self._grow_quad(self.street_quad, self.street.shape, sl, sr, st, sb, horizontal="x")
        # Wall plate: brick-only margins, wide enough to cover the view mid-turn.
        wl, wr, wt, wb = 1100, 640, 760, 820
        self.wall_tex = pad_with_edge_brick(self.plate, wl, wr, wt, wb)
        self.wall_quad_ext = self._grow_quad(self.wall_quad, self.plate.shape, wl, wr, wt, wb, horizontal="z")

    def _grow_quad(self, quad, shape, left, right, top, bottom, horizontal: str):
        """Expand a screen-aligned world quad to match padding on its texture.

        `quad` is TL, TR, BR, BL. Image X runs TL->TR. Image Y runs TL->BL.
        """
        h, w = shape[:2]
        tl, tr, br, bl = quad
        x_axis = tr - tl
        y_axis = bl - tl
        # Guard zero lengths.
        def unit(v):
            n = np.linalg.norm(v)
            return v / n if n > 1e-8 else v
        # Step in world units per texture pixel along each image axis.
        step_x = x_axis / w
        step_y = y_axis / h
        tl2 = tl - step_x * left - step_y * top
        tr2 = tr + step_x * right - step_y * top
        br2 = br + step_x * right + step_y * bottom
        bl2 = bl - step_x * left + step_y * bottom
        return np.stack([tl2, tr2, br2, bl2], axis=0)

    def _init_doors(self) -> None:
        def corners(rect):
            x0, y0, x1, y1 = rect
            pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            world = [self.unproject(u, v, Z_WALL) for u, v in pts]
            return np.array(world, np.float64)

        self.door_l_world = corners(self.door_l)
        self.door_r_world = corners(self.door_r)

    def unproject(self, u: float, v: float, z: float) -> np.ndarray:
        x = (u - CX) / self.fx * z
        y = -(v - CY) / self.fy * z
        return np.array([x, y, z], np.float64)

    def project(self, pts: np.ndarray, pos: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
        look = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, look)
        d = pts - pos
        xc = d @ right
        yc = d @ up
        zc = d @ look
        u = CX + self.fx * xc / zc
        v = CY - self.fy * yc / zc
        return np.stack([u, v], axis=1), zc

    def paint_sprite(self, frame: np.ndarray, sprite: np.ndarray, rect, z: float, cam_z: float) -> None:
        x0, y0, x1, y1 = rect
        scale = z / (z - cam_z)
        def tr(x, y):
            return CX + (x - CX) * scale, CY + (y - CY) * scale
        dx0, dy0 = tr(x0, y0)
        dx1, dy1 = tr(x1, y1)
        sh, sw = sprite.shape[:2]
        if sw < 2 or sh < 2:
            return
        mat = np.array(
            [[(dx1 - dx0) / sw, 0, dx0], [0, (dy1 - dy0) / sh, dy0]], np.float32
        )
        warped = cv2.warpAffine(
            sprite, mat, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
        )
        mask = cv2.warpAffine(
            np.full((sh, sw), 255, np.uint8),
            mat,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        a = (mask.astype(np.float32) / 255.0)[..., None]
        base = frame.astype(np.float32)
        frame[:] = np.clip(warped.astype(np.float32) * a + base * (1 - a), 0, 255).astype(np.uint8)

    def paint_door(self, frame: np.ndarray, sprite: np.ndarray, world: np.ndarray, pos, yaw) -> None:
        screen, zc = self.project(world, pos, yaw)
        if np.any(zc <= 0.05):
            return
        sh, sw = sprite.shape[:2]
        src = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], np.float32)
        dst = screen.astype(np.float32)
        if not np.isfinite(dst).all():
            return
        hom = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            sprite, hom, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
        )
        mask = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(mask, np.round(dst).astype(np.int32), 255)
        a = (mask.astype(np.float32) / 255.0)[..., None]
        base = frame.astype(np.float32)
        frame[:] = np.clip(warped.astype(np.float32) * a + base * (1 - a), 0, 255).astype(np.uint8)

    def open_doors(self, deg: float) -> tuple[np.ndarray, np.ndarray]:
        theta = math.radians(deg)

        def swing(world, hinge_ids, angle):
            origin = world[list(hinge_ids)].mean(axis=0)
            p = world - origin
            c, s = math.cos(angle), math.sin(angle)
            x = c * p[:, 0] - s * p[:, 2]
            z = s * p[:, 0] + c * p[:, 2]
            return np.stack([x, p[:, 1], z], axis=1) + origin

        # TL, TR, BR, BL. Left hinge is the left edge (0, 3). Right hinge is the right edge (1, 2).
        left = swing(self.door_l_world, (0, 3), theta)
        right = swing(self.door_r_world, (1, 2), -theta)
        return left, right

    def _paste(self, frame: np.ndarray, sprite: np.ndarray, x: int, y: int) -> None:
        sh, sw = sprite.shape[:2]
        x0, y0 = int(x), int(y)
        x1, y1 = x0 + sw, y0 + sh
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(W, x1), min(H, y1)
        if cx0 >= cx1 or cy0 >= cy1:
            return
        frame[cy0:cy1, cx0:cx1] = sprite[cy0 - y0 : cy0 - y0 + (cy1 - cy0), cx0 - x0 : cx0 - x0 + (cx1 - cx0)]

    def paint_flat_door(self, frame, sprite, rect, deg, hinge, cam_z) -> None:
        """Foreshorten the leaf toward its hinge. The opening itself stays rectangular."""
        x0, y0, x1, y1 = rect
        scale_z = Z_WALL / max(1e-3, Z_WALL - cam_z)

        def tr(x, y):
            return CX + (x - CX) * scale_z, CY + (y - CY) * scale_z

        ax, ay = tr(x0, y0)
        bx, by = tr(x1, y1)
        left, right = (ax, bx) if ax <= bx else (bx, ax)
        top, bot = (ay, by) if ay <= by else (by, ay)
        open_w = right - left
        open_h = bot - top
        if open_w < 2 or open_h < 2:
            return
        leaf = max(0.08, math.cos(math.radians(min(deg, 89.0))))
        leaf_w = max(2, int(round(open_w * leaf)))
        leaf_h = max(2, int(round(open_h)))
        door = cv2.resize(sprite, (leaf_w, leaf_h), interpolation=cv2.INTER_LINEAR)
        shade = 1.0 - 0.34 * math.sin(math.radians(deg))
        door = np.clip(door.astype(np.float32) * shade, 0, 255).astype(np.uint8)
        px = int(round(left if hinge == "left" else right - leaf_w))
        py = int(round(top))
        self._paste(frame, door, px, py)
        if deg > 10:
            tw = max(3, int(round(5 * scale_z)))
            tx = px + leaf_w - tw if hinge == "left" else px
            cv2.rectangle(
                frame,
                (max(0, tx), max(0, py)),
                (min(W - 1, tx + tw), min(H - 1, py + leaf_h)),
                (24, 18, 14),
                -1,
            )

    def _overlay_street(self, frame: np.ndarray, scale: float, mix: float) -> np.ndarray:
        """Lay the street over the gate plate. Outside the real street pixels, the gate remains."""
        if mix <= 1e-4:
            return frame
        if scale >= 0.999:
            street = self.street.astype(np.float32)
            a = np.float32(mix)
            base = frame.astype(np.float32)
            return np.clip(street * a + base * (1.0 - a), 0, 255).astype(np.uint8)
        mat = np.array(
            [[scale, 0, (1 - scale) * CX], [0, scale, (1 - scale) * CY]], np.float32
        )
        street = cv2.warpAffine(
            self.street, mat, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
        ).astype(np.float32)
        left = CX * (1.0 - scale)
        right = CX + scale * (W - CX)
        top = CY * (1.0 - scale)
        bot = CY + scale * (H - CY)
        xs = np.arange(W, dtype=np.float32)
        ys = np.arange(H, dtype=np.float32)
        dx = np.minimum(xs - left, right - xs)
        dy = np.minimum(ys[:, None] - top, bot - ys[:, None])
        # Soft edge so the gate's real floor and foliage meet the street without a hard cut.
        feather = np.clip(np.minimum(dx, dy) / 56.0, 0.0, 1.0).astype(np.float32)
        a = (feather * np.float32(mix))[..., None]
        base = frame.astype(np.float32)
        return np.clip(street * a + base * (1.0 - a), 0, 255).astype(np.uint8)

    def render_gate(self, cam_z: float, deg: float, street_mix: float, street_scale: float, edge_hold: float) -> np.ndarray:
        if deg < 0.2 and cam_z < 1e-4:
            return self.photo.copy()
        gate = warp_plane(self.photo, Z_FAR, cam_z)
        frame = self._overlay_street(gate, street_scale, street_mix)
        # Keep the gate's own side walls until the street plate has grown over them.
        if edge_hold > 1e-3 and street_mix > 1e-3:
            ml = warp_plane(self.mask_l, Z_FAR, cam_z).astype(np.float32) / 255.0
            mr = warp_plane(self.mask_r, Z_FAR, cam_z).astype(np.float32) / 255.0
            xs = np.arange(W, dtype=np.float32)
            feather = 72.0
            left_b = W * edge_hold
            right_b = W * (1.0 - edge_hold)
            geom = np.maximum(
                np.clip((left_b - xs) / feather, 0.0, 1.0),
                np.clip((xs - right_b) / feather, 0.0, 1.0),
            )
            keep = (np.maximum(ml, mr) * geom)[..., None]
            frame = np.clip(
                frame.astype(np.float32) * (1.0 - keep) + gate.astype(np.float32) * keep, 0, 255
            ).astype(np.uint8)
        self.paint_sprite(frame, self.interior_l, self.door_l, Z_WALL, cam_z)
        self.paint_sprite(frame, self.interior_r, self.door_r, Z_WALL, cam_z)
        self.paint_flat_door(frame, self.sprite_l, self.door_l, deg, "left", cam_z)
        self.paint_flat_door(frame, self.sprite_r, self.door_r, deg, "right", cam_z)
        frame = alpha_composite(frame, warp_plane(self.wall_l, Z_WALL, cam_z))
        frame = alpha_composite(frame, warp_plane(self.wall_r, Z_WALL, cam_z))
        return frame

    def paint_quad(self, frame: np.ndarray, texture: np.ndarray, world: np.ndarray, pos, yaw) -> np.ndarray:
        uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], np.float64)
        self._paint_quad_rec(frame, texture, world, uv, pos, yaw, 0)
        return frame

    def _paint_quad_rec(self, frame, texture, world, uv, pos, yaw, depth) -> None:
        _, zc = self.project(world, pos, yaw)
        if not np.isfinite(zc).all():
            return
        if np.any(zc <= 0.15):
            if depth >= 4 or np.all(zc <= 0.15):
                return
            mid = world.mean(axis=0)
            mid_uv = uv.mean(axis=0)
            edge_w = [(world[i] + world[(i + 1) % 4]) / 2 for i in range(4)]
            edge_uv = [(uv[i] + uv[(i + 1) % 4]) / 2 for i in range(4)]
            children = [
                ([world[0], edge_w[0], mid, edge_w[3]], [uv[0], edge_uv[0], mid_uv, edge_uv[3]]),
                ([edge_w[0], world[1], edge_w[1], mid], [edge_uv[0], uv[1], edge_uv[1], mid_uv]),
                ([mid, edge_w[1], world[2], edge_w[2]], [mid_uv, edge_uv[1], uv[2], edge_uv[2]]),
                ([edge_w[3], mid, edge_w[2], world[3]], [edge_uv[3], mid_uv, edge_uv[2], uv[3]]),
            ]
            for cw, cu in children:
                self._paint_quad_rec(frame, texture, np.asarray(cw, np.float64), np.asarray(cu, np.float64), pos, yaw, depth + 1)
            return
        screen, _ = self.project(world, pos, yaw)
        dst = screen.astype(np.float32)
        if dst[:, 0].max() < -30 or dst[:, 0].min() > W + 30 or dst[:, 1].max() < -30 or dst[:, 1].min() > H + 30:
            return
        sh, sw = texture.shape[:2]
        src = np.stack([uv[:, 0] * (sw - 1), uv[:, 1] * (sh - 1)], axis=1).astype(np.float32)
        try:
            hom = cv2.getPerspectiveTransform(src, dst)
        except cv2.error:
            return
        warped = cv2.warpPerspective(
            texture, hom, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        mask = np.zeros((H, W), np.uint8)
        poly = np.round(dst).astype(np.int32)
        if cv2.contourArea(poly.astype(np.float32)) < 4:
            return
        cv2.fillConvexPoly(mask, poly, 255)
        a = (mask.astype(np.float32) / 255.0)[..., None]
        base = frame.astype(np.float32)
        frame[:] = np.clip(warped.astype(np.float32) * a + base * (1.0 - a), 0, 255).astype(np.uint8)

    def render_turn(self, u: float) -> np.ndarray:
        u = float(np.clip(u, 0.0, 1.0))
        if u <= 1e-4:
            return self.street.copy()
        if u >= 0.999:
            return self.plate.copy()
        p0, y0 = self.turn_start
        p1, y1 = self.turn_end
        pos = p0 * (1 - u) + p1 * u
        yaw = y0 * (1 - u) + y1 * u
        frame = np.zeros((H, W, 3), np.uint8)
        frame[:] = (1, 2, 3)
        frame = self.paint_quad(frame, self.street_tex, self.street_quad_ext, pos, yaw)
        frame = self.paint_quad(frame, self.wall_tex, self.wall_quad_ext, pos, yaw)
        return frame

    def render(self, t: float) -> np.ndarray:
        if t < 0.5:
            return self.photo.copy()
        if t < 6.5:
            deg = door_degrees(t)
            u = dolly_progress(t)
            cam_z = CAM_Z_END * u
            # Fade the aligned street into the alley, then grow it to full frame
            # on real pixels only. The gate plate fills anything the street has not reached.
            far_scale = Z_FAR / (Z_FAR - cam_z) if cam_z > 1e-4 else 1.0
            matched = 0.75 * far_scale
            mix = ease_in_out(float(np.clip(u / 0.12, 0.0, 1.0)))
            grow = ease_in_out(float(np.clip((u - 0.12) / 0.70, 0.0, 1.0)))
            street_scale = matched + (1.0 - matched) * grow
            edge_hold = float(np.clip((0.96 - grow) / 0.12, 0.0, 1.0))
            if t < 2.7:
                mix = 0.0
                cam_z = 0.0
                street_scale = 0.75
                edge_hold = 1.0
            return self.render_gate(cam_z, deg, float(mix), float(street_scale), edge_hold)
        if t < 9.7:
            return self.render_turn(turn_progress(t))
        frame = self.plate.copy()
        if t < 10.5:
            return frame
        dim = ease_out((t - 10.5) / 0.25) if t < 10.75 else 1.0
        if dim > 0:
            frame = np.clip(frame.astype(np.float32) * (1.0 - 0.15 * dim), 0, 255).astype(np.uint8)
        left_a = ease_out((t - 10.5) / 0.25) if t >= 10.5 else 0.0
        if left_a > 0:
            layer = self.left_ui.copy()
            layer[:, :, 3] = (layer[:, :, 3].astype(np.float32) * left_a).astype(np.uint8)
            over(frame, layer, self.left_pos[0], self.left_pos[1])
        if t >= 10.75:
            right_a = ease_out((t - 10.75) / 0.35)
            shift = 24.0 * (1.0 - right_a)
            layer = self.right_ui.copy()
            layer[:, :, 3] = (layer[:, :, 3].astype(np.float32) * right_a).astype(np.uint8)
            over(frame, layer, int(round(self.right_pos[0] + shift)), self.right_pos[1])
        return frame


def save(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, quality=92)


def preview(scene: Scene) -> None:
    out = Path("/tmp/preview")
    out.mkdir(parents=True, exist_ok=True)
    # Masks and door sprites for alignment checks.
    overlay = scene.photo.copy()
    overlay[scene.mask_l > 40] = (
        overlay[scene.mask_l > 40] * 0.55 + np.array([220, 40, 40]) * 0.45
    ).astype(np.uint8)
    overlay[scene.mask_r > 40] = (
        overlay[scene.mask_r > 40] * 0.55 + np.array([40, 90, 220]) * 0.45
    ).astype(np.uint8)
    save(out / "masks.jpg", overlay)
    save(out / "plate.jpg", scene.plate)
    save(out / "board.jpg", scene.board[:, :, :3])
    blank = np.full_like(scene.photo, 236)
    over(blank, scene.left_ui, scene.left_pos[0], scene.left_pos[1])
    over(blank, scene.right_ui, scene.right_pos[0], scene.right_pos[1])
    save(out / "ui.jpg", blank)
    times = [0.0, 0.5, 1.3, 2.2, 2.7, 3.6, 4.6, 5.6, 6.5, 7.4, 8.4, 9.2, 9.7, 10.6, 11.2, 12.9]
    for t in times:
        frame = scene.render(t)
        save(out / f"t{t:04.1f}.jpg", frame)
        print(f"preview t={t:.1f}", flush=True)
    # Identity checks
    a = scene.render(0.0)
    b = scene.photo
    print("hold max diff", np.abs(a.astype(int) - b.astype(int)).max())
    end = scene.render(9.7)
    print("plate max diff", np.abs(end.astype(int) - scene.plate.astype(int)).max())
    street = scene.render(6.5)
    print("street junction max diff", np.abs(street.astype(int) - scene.street.astype(int)).max())


def render_video(scene: Scene, path: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
        "-preset", "medium", "-movflags", "+faststart",
        str(path),
    ]
    log_path = Path("/tmp/ffmpeg-antidrug.log")
    log_file = log_path.open("w")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log_file)
    assert proc.stdin is not None
    try:
        for i in range(NFRAMES):
            t = i / FPS
            frame = scene.render(t)
            proc.stdin.write(frame.tobytes())
            if i % 30 == 0:
                print(f"frame {i}/{NFRAMES} t={t:.2f}", flush=True)
    finally:
        proc.stdin.close()
        log_file.close()
    code = proc.wait()
    if code != 0:
        err = log_path.read_text(errors="replace")
        raise SystemExit(err[-4000:])
    print("wrote", path, flush=True)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "preview"
    print("building scene", flush=True)
    scene = Scene()
    # Projection sanity: street at start and wall at end should fill the frame.
    s0, _ = scene.project(scene.street_quad, *scene.turn_start)
    s1, _ = scene.project(scene.wall_quad, *scene.turn_end)
    print("street start corners", np.round(s0, 1))
    print("wall end corners", np.round(s1, 1))
    if mode == "preview":
        preview(scene)
    elif mode == "full":
        render_video(scene, OUT)
    elif mode == "both":
        preview(scene)
        render_video(scene, OUT)
    else:
        raise SystemExit("use preview | full | both")


if __name__ == "__main__":
    main()
