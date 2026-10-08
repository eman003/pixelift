"""AI face restoration with identity protection.

Pipeline per face (after GFPGAN's reference helper, facexlib, MIT):
detect faces and five landmarks with RetinaFace → align each face to the
512×512 template with a similarity transform → restore it with GFPGAN →
blend it back into the photo.

Identity comes first. The restored face is never pasted as-is:

* Only the facial area (a soft ellipse) is replaced, so hair, ears and the
  background keep their original pixels.
* The original face's low frequencies (overall shape, skin tone, lighting)
  are kept: GFPGAN contributes fine detail only (*Natural*) or detail and
  structure with the original tones (*Strong*).
* The blend weight drops for faces that are tiny, uncertain detections, or
  that GFPGAN changed beyond what restoring detail explains (a sign it is
  inventing a different face), so badly damaged faces stay close to the
  original instead of becoming a stranger — and for faces that are already
  sharp, where GFPGAN would only smooth real skin texture away.
* Only GFPGAN's *changes* are added to the photo, so a face larger than
  GFPGAN's 512 px working size keeps its own finer detail.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image

from pixelift.core import device_manager as dm
from pixelift.core.restoration import filters as fl
from pixelift.core.restoration.settings import FACE_OFF, FACE_STRONG

log = logging.getLogger(__name__)

FACE_SIZE = 512
# Five-point template of the 512×512 GFPGAN crop: eyes, nose, mouth corners.
TEMPLATE = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.1936],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float64,
)
# Soft ellipse covering forehead to chin and cheek to cheek in the template.
ELLIPSE = (256.0, 292.0, 152.0, 188.0)  # cx, cy, rx, ry
DETECT_MAX_SIDE = 2048
DETECT_MIN_SCORE = 0.9
SURE_SCORE = 0.97  # facexlib's threshold; below it the blend is reduced
MIN_EYE_DISTANCE = 5.0  # px in the source photo; smaller "faces" are skipped
BORDER_RGB = (132.0, 133.0, 135.0)  # facexlib's grey crop border (RGB order)


@dataclass(frozen=True)
class Face:
    box: tuple[float, float, float, float]
    score: float
    landmarks: np.ndarray  # (5, 2) x, y in image pixels

    @property
    def eye_distance(self) -> float:
        return float(np.linalg.norm(self.landmarks[0] - self.landmarks[1]))


@dataclass
class FaceReport:
    found: int = 0
    restored: int = 0
    protected: int = 0  # restored only lightly to protect identity


# --- detection ----------------------------------------------------------------
def _priors(height: int, width: int, net_cls: type) -> torch.Tensor:
    anchors = []
    for step, sizes in zip(net_cls.STEPS, net_cls.MIN_SIZES, strict=True):
        fh, fw = math.ceil(height / step), math.ceil(width / step)
        ys, xs = torch.meshgrid(torch.arange(fh), torch.arange(fw), indexing="ij")
        cx = ((xs + 0.5) * step / width).float()
        cy = ((ys + 0.5) * step / height).float()
        per_size = [
            torch.stack(
                [cx, cy, torch.full_like(cx, s / width), torch.full_like(cy, s / height)], dim=-1
            )
            for s in sizes
        ]
        anchors.append(torch.stack(per_size, dim=2).reshape(-1, 4))
    return torch.cat(anchors)


def _nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> list[int]:
    order = scores.argsort(descending=True)
    boxes = boxes[order]
    x1, y1, x2, y2 = boxes.unbind(1)
    area = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    keep: list[int] = []
    suppressed = torch.zeros(len(order), dtype=torch.bool)
    for i in range(len(order)):
        if suppressed[i]:
            continue
        keep.append(int(order[i]))
        xx1 = torch.maximum(x1[i], x1)
        yy1 = torch.maximum(y1[i], y1)
        xx2 = torch.minimum(x2[i], x2)
        yy2 = torch.minimum(y2[i], y2)
        inter = (xx2 - xx1).clamp_min(0) * (yy2 - yy1).clamp_min(0)
        iou = inter / (area[i] + area - inter).clamp_min(1e-6)
        suppressed |= iou > threshold
    return keep


def detect_faces(net: torch.nn.Module, device: dm.DeviceInfo, rgb: np.ndarray) -> list[Face]:
    """Faces in ``rgb`` (H, W, 3 uint8), best first, in ``rgb``'s pixel coordinates."""
    height, width = rgb.shape[:2]
    factor = min(1.0, DETECT_MAX_SIDE / max(height, width))
    small = rgb
    if factor < 1:
        size = (max(1, round(width * factor)), max(1, round(height * factor)))
        small = np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR))
    h, w = small.shape[:2]
    dtype = next(net.parameters()).dtype
    torch_device = dm.to_torch(device)
    with torch.inference_mode():
        x = torch.from_numpy(np.ascontiguousarray(small[..., ::-1])).to(torch_device)  # BGR
        x = x.permute(2, 0, 1).unsqueeze(0).float()
        x -= torch.tensor([104.0, 117.0, 123.0], device=torch_device).view(1, 3, 1, 1)
        loc, conf, marks = net(x.to(dtype))
        loc, conf, marks = loc[0].float().cpu(), conf[0].float().cpu(), marks[0].float().cpu()
    scores = conf[:, 1]
    keep = scores > DETECT_MIN_SCORE
    if not bool(keep.any()):
        return []
    priors = _priors(h, w, type(net))[keep]
    loc, marks, scores = loc[keep], marks[keep], scores[keep]
    v0, v1 = 0.1, 0.2
    centers = priors[:, :2] + loc[:, :2] * v0 * priors[:, 2:]
    sizes = priors[:, 2:] * torch.exp(loc[:, 2:] * v1)
    boxes = torch.cat([centers - sizes / 2, centers + sizes / 2], dim=1)
    landmarks = priors[:, None, :2] + marks.view(-1, 5, 2) * v0 * priors[:, None, 2:]
    scale = torch.tensor([w, h], dtype=torch.float32) / factor
    boxes = boxes * scale.repeat(2)
    landmarks = landmarks * scale
    faces = []
    for i in _nms(boxes, scores, 0.4):
        face = Face(
            tuple(float(v) for v in boxes[i]),  # type: ignore[arg-type]
            float(scores[i]),
            landmarks[i].double().numpy(),
        )
        if face.eye_distance >= MIN_EYE_DISTANCE:
            faces.append(face)
    return faces


# --- geometry -----------------------------------------------------------------
def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares rotation + uniform scale + translation (Umeyama), 2×3."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = d_c.T @ s_c / len(src)
    u, sv, vt = np.linalg.svd(cov)
    d = np.diag([1.0, 1.0 if np.linalg.det(u) * np.linalg.det(vt) >= 0 else -1.0])
    rotation = u @ d @ vt
    scale = np.trace(np.diag(sv) @ d) / max((s_c**2).sum() / len(src), 1e-12)
    translation = mu_d - scale * rotation @ mu_s
    return np.hstack([scale * rotation, translation[:, None]])


def invert(m: np.ndarray) -> np.ndarray:
    full = np.vstack([m, [0, 0, 1]])
    return np.linalg.inv(full)[:2]


def _compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a ∘ b (apply b first)."""
    return (np.vstack([a, [0, 0, 1]]) @ np.vstack([b, [0, 0, 1]]))[:2]


def _sample(src: torch.Tensor, m_dst_to_src: np.ndarray, out_h: int, out_w: int) -> torch.Tensor:
    """Bilinear warp: output pixel (x, y) reads ``src`` at ``m · (x, y, 1)``.

    Returns the warped (1, C, out_h, out_w) image and, as an extra last
    channel, the coverage (1 inside ``src``, 0 outside).
    """
    sh, sw = src.shape[2:]
    ys, xs = torch.meshgrid(
        torch.arange(out_h, dtype=torch.float64),
        torch.arange(out_w, dtype=torch.float64),
        indexing="ij",
    )
    m = torch.from_numpy(m_dst_to_src)
    sx = m[0, 0] * xs + m[0, 1] * ys + m[0, 2]
    sy = m[1, 0] * xs + m[1, 1] * ys + m[1, 2]
    grid = torch.stack([2 * sx / max(sw - 1, 1) - 1, 2 * sy / max(sh - 1, 1) - 1], dim=-1)
    grid = grid.unsqueeze(0).to(src.dtype)
    ones = torch.ones_like(src[:, :1])
    return F.grid_sample(
        torch.cat([src, ones], dim=1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


def crop_face(rgb: np.ndarray, image_to_face: np.ndarray) -> np.ndarray:
    """The aligned 512×512 face crop (float32, 0..255, RGB) from ``rgb``."""
    height, width = rgb.shape[:2]
    face_to_image = invert(image_to_face)
    corners = np.array([[0, 0], [FACE_SIZE, 0], [0, FACE_SIZE], [FACE_SIZE, FACE_SIZE]], float)
    pts = corners @ face_to_image[:, :2].T + face_to_image[:, 2]
    x0 = int(max(0, math.floor(pts[:, 0].min()) - 2))
    y0 = int(max(0, math.floor(pts[:, 1].min()) - 2))
    x1 = int(min(width, math.ceil(pts[:, 0].max()) + 3))
    y1 = int(min(height, math.ceil(pts[:, 1].max()) + 3))
    region = rgb[y0:y1, x0:x1]
    # Region pixel (u, v) is image pixel (u + x0, v + y0).
    region_to_image = np.array([[1.0, 0, x0], [0, 1.0, y0]])
    scale = math.sqrt(abs(np.linalg.det(image_to_face[:, :2])))
    if scale < 0.75 and region.size:
        # Shrinking a large face: pre-filter to avoid aliasing.
        f = scale * 1.25
        size = (max(1, round(region.shape[1] * f)), max(1, round(region.shape[0] * f)))
        fx, fy = size[0] / region.shape[1], size[1] / region.shape[0]
        region = np.asarray(Image.fromarray(region).resize(size, Image.Resampling.LANCZOS))
        # Resized pixel (u', v') covers region pixel ((u' + .5) / f - .5, ...).
        resized_to_region = np.array([[1 / fx, 0, 0.5 / fx - 0.5], [0, 1 / fy, 0.5 / fy - 0.5]])
        region_to_image = _compose(region_to_image, resized_to_region)
    if not region.size:
        return np.tile(np.array(BORDER_RGB, np.float32), (FACE_SIZE, FACE_SIZE, 1))
    src = torch.from_numpy(np.array(region)).permute(2, 0, 1).unsqueeze(0).float()
    face_to_region = _compose(invert(region_to_image), face_to_image)
    warped = _sample(src, face_to_region, FACE_SIZE, FACE_SIZE)
    rgb_part, cover = warped[:, :3], warped[:, 3:]
    border = torch.tensor(BORDER_RGB).view(1, 3, 1, 1)
    out = rgb_part + (1 - cover) * border
    return out[0].permute(1, 2, 0).numpy()


def ellipse_mask(size: int = FACE_SIZE) -> torch.Tensor:
    cx, cy, rx, ry = ELLIPSE
    ys, xs = torch.meshgrid(torch.arange(size).float(), torch.arange(size).float(), indexing="ij")
    r = torch.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)
    t = ((1.0 - r) / 0.2).clamp(0, 1)
    return (t * t * (3 - 2 * t)).view(1, 1, size, size)  # smoothstep


# --- blending -----------------------------------------------------------------
def base_weight(mode: str, fidelity: int) -> float:
    """How much of the restored face to use, before the identity safeguards."""
    if mode == FACE_OFF:
        return 0.0
    a = min(100, max(0, fidelity)) / 100
    if mode == FACE_STRONG:
        return 0.6 + 0.4 * a
    return 0.35 + 0.55 * a


def identity_factor(original: torch.Tensor, restored: torch.Tensor, mask: torch.Tensor) -> float:
    """1.0 when the restoration only sharpened the face, lower as it diverges.

    Compares the faces' structure (blurred brightness, so restored detail does
    not count) inside the facial ellipse: a low correlation means GFPGAN
    produced a face that does not match the photo.
    """
    a = fl.gaussian_blur(fl.luma(original), 5.0)
    b = fl.gaussian_blur(fl.luma(restored), 5.0)
    w = mask / mask.sum().clamp_min(1e-6)
    ma, mb = (a * w).sum(), (b * w).sum()
    cov = ((a - ma) * (b - mb) * w).sum()
    va = ((a - ma) ** 2 * w).sum()
    vb = ((b - mb) ** 2 * w).sum()
    corr = float(cov / torch.sqrt(va * vb).clamp_min(1e-8))
    if va < 1e-5:  # a featureless (blank / totally faded) face: nothing to trust
        return 0.25
    return min(1.0, max(0.25, (corr - 0.55) / (0.85 - 0.55)))


def size_factor(eye_distance: float) -> float:
    """Tiny faces carry too little information: restore them only lightly."""
    return min(1.0, max(0.3, (eye_distance - 5.0) / (16.0 - 5.0)))


def detail_factor(original: torch.Tensor, restored: torch.Tensor, mask: torch.Tensor) -> float:
    """How much restoring helps: ~0.15 for an already sharp face, 1 for a degraded one.

    Compares fine-detail energy inside the facial ellipse. GFPGAN *removes*
    texture from faces that are already sharp (gain < 1) and adds a lot to
    blurred or damaged ones (gain > 1.5).
    """

    def energy(face: torch.Tensor) -> float:
        y = fl.luma(face)
        detail = y - fl.gaussian_blur(y, 1.5)
        return float((detail * detail * mask).sum() / mask.sum().clamp_min(1e-6))

    gain = energy(restored) / max(energy(original), 1e-6)
    return min(1.0, max(0.15, (gain - 0.9) / 0.6))


def score_factor(score: float) -> float:
    return min(
        1.0, max(0.5, 0.5 + 0.5 * (score - DETECT_MIN_SCORE) / (SURE_SCORE - DETECT_MIN_SCORE))
    )


def harmonise(original: torch.Tensor, restored: torch.Tensor, mode: str) -> torch.Tensor:
    """Give ``restored`` the original's low frequencies (tone, shape, skin colour)."""
    sigma = 48.0 if mode == FACE_STRONG else 6.0
    return restored + fl.gaussian_blur(original, sigma) - fl.gaussian_blur(restored, sigma)


def run_gfpgan(net: torch.nn.Module, device: dm.DeviceInfo, face: np.ndarray) -> np.ndarray:
    """(512, 512, 3) float 0..255 RGB -> restored, same format."""
    dtype = next(net.parameters()).dtype
    with torch.inference_mode():
        x = torch.from_numpy(np.ascontiguousarray(face)).permute(2, 0, 1).unsqueeze(0)
        x = (x / 127.5 - 1.0).to(dm.to_torch(device), dtype)
        y = net(x).float().clamp_(-1, 1)
        out = ((y + 1) * 127.5)[0].permute(1, 2, 0).cpu().numpy()
    return out


def paste_face(
    image: np.ndarray, change: np.ndarray, image_to_face: np.ndarray, alpha: torch.Tensor
) -> None:
    """Add ``change`` (512² float RGB, restored − original crop) to ``image`` in place.

    Weighted by ``alpha`` (1, 1, 512, 512). Adding the change instead of
    pasting the restored face keeps detail the photo has beyond 512 px.
    """
    height, width = image.shape[:2]
    face_to_image = invert(image_to_face)
    cx, cy, rx, ry = ELLIPSE
    corners = np.array(
        [[cx - rx, cy - ry], [cx + rx, cy - ry], [cx - rx, cy + ry], [cx + rx, cy + ry]]
    )
    pts = corners @ face_to_image[:, :2].T + face_to_image[:, 2]
    x0 = int(max(0, math.floor(pts[:, 0].min()) - 1))
    y0 = int(max(0, math.floor(pts[:, 1].min()) - 1))
    x1 = int(min(width, math.ceil(pts[:, 0].max()) + 2))
    y1 = int(min(height, math.ceil(pts[:, 1].max()) + 2))
    if x1 <= x0 or y1 <= y0:
        return
    region_to_face = _compose(image_to_face, np.array([[1.0, 0, x0], [0, 1.0, y0]]))
    src = torch.cat(
        [torch.from_numpy(np.ascontiguousarray(change)).permute(2, 0, 1).unsqueeze(0), alpha],
        dim=1,
    ).float()
    warped = _sample(src, region_to_face, y1 - y0, x1 - x0)
    delta, a = warped[:, :3], warped[:, 3:4] * warped[:, 4:5]
    base = torch.from_numpy(np.array(image[y0:y1, x0:x1])).permute(2, 0, 1)
    base = base.unsqueeze(0).float()
    out = base + a * delta
    image[y0:y1, x0:x1] = out[0].permute(1, 2, 0).clamp(0, 255).add(0.5).to(torch.uint8).numpy()
