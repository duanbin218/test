"""基于反向象限筛选可达像素的辅助函数与命令行工具。"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - 容错导入以便 CLI 给出提示
    cv2 = None  # type: ignore


_LOGGER = logging.getLogger(__name__)


def _angle_difference_rad(a: float, b: float) -> float:
    """计算两个弧度角的最小差值，结果范围为 [-pi, pi]。"""

    diff = (a - b) % (2 * math.pi)
    if diff > math.pi:
        diff -= 2 * math.pi
    return diff


def _normalize_angle(angle: float) -> float:
    """将任意弧度角归一化到 [-pi, pi) 区间。"""

    normalized = (angle + math.pi) % (2 * math.pi) - math.pi
    return normalized


def _is_white_pixel(pixel: np.ndarray) -> bool:
    """判定像素是否为可达白色，支持一维或多维像素。"""

    if pixel.ndim == 0:
        value = int(pixel)
        return value > 200
    if pixel.ndim == 1:
        return bool(np.all(pixel > 200))
    raise ValueError("像素数据维度异常，无法判定是否为白色")


def _validate_grid(grid: np.ndarray) -> np.ndarray:
    """校验并转换地图数组，确保返回 H×W×3 的三通道数组。"""

    if grid.ndim == 2:
        _LOGGER.warning("输入图像为灰度，将复制三通道以便统一处理。")
        grid = np.stack([grid] * 3, axis=-1)
    elif grid.ndim == 3 and grid.shape[2] == 4:
        _LOGGER.warning("输入图像包含 Alpha 通道，将仅保留前三个通道。")
        grid = grid[:, :, :3]
    elif grid.ndim == 3 and grid.shape[2] == 3:
        pass
    else:
        raise ValueError("地图数组维度不符合要求，需为 H×W 或 H×W×3/4。")
    return grid.astype(np.uint8, copy=False)


def _iter_candidate_pixels(
    cx: int,
    cy: int,
    min_dist: int,
    max_dist: int,
    angle_center: float,
    angle_tolerance: float,
    grid_shape: Tuple[int, int],
    x_sign_constraint: int,
    y_sign_constraint: int,
) -> Iterable[Tuple[int, int, float, float]]:
    """生成满足角度、距离与象限约束的候选像素。"""

    height, width = grid_shape
    max_radius = max_dist
    min_x = max(0, int(math.floor(cx - max_radius)))
    max_x = min(width - 1, int(math.ceil(cx + max_radius)))
    min_y = max(0, int(math.floor(cy - max_radius)))
    max_y = min(height - 1, int(math.ceil(cy + max_radius)))

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            dx = x - cx
            dy = y - cy
            dist = math.hypot(dx, dy)
            if dist < min_dist or dist > max_dist:
                continue
            if x_sign_constraint != 0 and dx * x_sign_constraint >= 0:
                # 当约束为 -1 时 dx 需 <0；约束为 +1 时 dx 需 >0
                continue
            if y_sign_constraint != 0 and dy * y_sign_constraint >= 0:
                continue
            if dist == 0:
                continue
            angle = math.atan2(dy, dx)
            if abs(_angle_difference_rad(angle, angle_center)) > angle_tolerance:
                continue
            yield x, y, dist, abs(_angle_difference_rad(angle, angle_center))


def select_opposite_quadrant_point(
    B: tuple[int, int],
    C: tuple[int, int],
    grid: "np.ndarray",
    *,
    min_dist_from_C: int,
    max_dist_from_C: int,
    angle_tolerance_deg: float = 90.0,
    prefer: str = "closest_to_C",
) -> tuple[int, int] | None:
    """在 C 点的反向象限环带内选取可达像素 D。

    参数：
        B: 点 B 的整数像素坐标 (x, y)。
        C: 点 C 的整数像素坐标 (x, y)。
        grid: 黑白地图数组，0/255 或近似值，白色表示可达。
        min_dist_from_C: 允许的最小半径，单位像素。
        max_dist_from_C: 允许的最大半径，单位像素。
        angle_tolerance_deg: 目标方向允许的最大角度偏差（度）。
        prefer: 结果优先策略，可选 "closest_to_C" 或 "farthest_in_band"。

    返回：
        找到满足条件的像素坐标 (x, y) 则返回；否则返回 None。
    """

    if min_dist_from_C < 0 or max_dist_from_C < 0:
        raise ValueError("最小与最大半径需为非负整数。")
    if min_dist_from_C > max_dist_from_C:
        raise ValueError("最小半径不得大于最大半径。")
    if angle_tolerance_deg < 0 or angle_tolerance_deg > 180:
        raise ValueError("角度容差需位于 0 至 180 度之间。")
    if prefer not in {"closest_to_C", "farthest_in_band"}:
        raise ValueError("prefer 参数仅支持 'closest_to_C' 或 'farthest_in_band'。")

    if not isinstance(grid, np.ndarray):
        raise TypeError("grid 参数需为 numpy.ndarray。")

    grid = _validate_grid(grid)
    height, width, _ = grid.shape

    bx, by = B
    cx, cy = C

    if not (0 <= cx < width and 0 <= cy < height):
        raise ValueError("点 C 坐标超出地图范围。")
    if not (0 <= bx < width and 0 <= by < height):
        raise ValueError("点 B 坐标超出地图范围。")

    dx_bc = bx - cx
    dy_bc = by - cy
    if dx_bc == 0 and dy_bc == 0:
        raise ValueError("点 B 与点 C 重合，方向未定义。")

    angle_bc = math.atan2(dy_bc, dx_bc)
    target_angle = _normalize_angle(angle_bc + math.pi)
    angle_tolerance_rad = math.radians(angle_tolerance_deg)

    # 依据 B 相对 C 的方向设置象限符号约束；0 表示无约束
    x_constraint = 0
    y_constraint = 0
    if dx_bc > 0:
        x_constraint = -1  # 期望 D 在 C 左侧
    elif dx_bc < 0:
        x_constraint = 1   # 期望 D 在 C 右侧
    if dy_bc > 0:
        y_constraint = -1  # 期望 D 在 C 上方（图像坐标向下为正）
    elif dy_bc < 0:
        y_constraint = 1   # 期望 D 在 C 下方

    candidates: List[Tuple[float, float, int, int]] = []
    for x, y, dist, angle_diff in _iter_candidate_pixels(
        cx,
        cy,
        min_dist_from_C,
        max_dist_from_C,
        target_angle,
        angle_tolerance_rad,
        (height, width),
        x_constraint,
        y_constraint,
    ):
        pixel = grid[y, x]
        if not _is_white_pixel(pixel):
            continue
        candidates.append((angle_diff, dist, x, y))

    if not candidates:
        return None

    reverse_radius = prefer == "farthest_in_band"

    def _candidate_key(item: Tuple[float, float, int, int]) -> Tuple[float, float, int, int]:
        angle_diff, radius, x_val, y_val = item
        radius_key = radius if not reverse_radius else -radius
        return angle_diff, radius_key, x_val, y_val

    candidates.sort(key=_candidate_key)
    _, _, best_x, best_y = candidates[0]
    return best_x, best_y


def _load_grid_from_path(path: Path) -> np.ndarray:
    """从文件加载地图，支持 BMP/PNG/JPG 或 numpy.npy。"""

    if not path.exists():
        raise FileNotFoundError(f"地图文件不存在：{path}")
    if path.suffix.lower() == ".npy":
        grid = np.load(path)
        _LOGGER.info("已从 NPY 文件读取地图数组。")
        return grid
    if cv2 is None:
        raise RuntimeError("未安装 OpenCV，无法读取图像文件，请安装 opencv-python。")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"无法读取图像文件：{path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _LOGGER.info("已成功读取图像并转换为 RGB。")
    return image


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    """解析命令行参数，返回包含所有配置项的命名空间。"""

    parser = argparse.ArgumentParser(
        description="在反向象限的环带内挑选可达像素点",
    )
    parser.add_argument("--map", required=True, help="地图文件路径，可为 BMP/PNG/JPG 或 NPY。")
    parser.add_argument("--bx", type=int, required=True, help="点 B 的 x 坐标。")
    parser.add_argument("--by", type=int, required=True, help="点 B 的 y 坐标。")
    parser.add_argument("--cx", type=int, required=True, help="点 C 的 x 坐标。")
    parser.add_argument("--cy", type=int, required=True, help="点 C 的 y 坐标。")
    parser.add_argument("--minC", type=int, required=True, help="环带的最小半径。")
    parser.add_argument("--maxC", type=int, required=True, help="环带的最大半径。")
    parser.add_argument(
        "--angle-tol",
        type=float,
        default=90.0,
        help="角度容差（度），默认为 90 覆盖整个反向象限。",
    )
    parser.add_argument(
        "--prefer",
        choices=["closest_to_C", "farthest_in_band"],
        default="closest_to_C",
        help="候选排序策略，可选最近或最远。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="启用详细日志输出。",
    )
    return parser.parse_args(argv)


def _configure_logging(verbose: bool) -> None:
    """根据命令行开关配置中文日志。"""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def main(argv: Iterable[str] | None = None) -> int:
    """命令行入口，处理参数解析、地图加载与结果输出。"""

    args = _parse_args(argv or sys.argv[1:])
    _configure_logging(args.verbose)

    grid_path = Path(args.map)
    try:
        grid = _load_grid_from_path(grid_path)
        result = select_opposite_quadrant_point(
            B=(args.bx, args.by),
            C=(args.cx, args.cy),
            grid=grid,
            min_dist_from_C=args.minC,
            max_dist_from_C=args.maxC,
            angle_tolerance_deg=args.angle_tol,
            prefer=args.prefer,
        )
    except Exception as exc:  # pragma: no cover - CLI 容错输出
        _LOGGER.error("执行失败：%s", exc)
        return 2

    if result is None:
        print("NOT_FOUND")
        return 1
    print(f"{result[0]},{result[1]}")
    return 0


if __name__ == "__main__":  # pragma: no cover - 供命令行执行
    sys.exit(main())
