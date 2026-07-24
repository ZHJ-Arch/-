import sys, os, math, csv, json, time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.optimize import least_squares, differential_evolution

from PySide6.QtCore import Qt, QTimer, QObject, Signal, QThread
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
    QTextEdit, QMessageBox, QSlider, QDoubleSpinBox, QSpinBox, QComboBox,
    QSplitter, QScrollArea, QFileDialog
)

APP_VERSION = "V1.31"


def resource_path(relative_path: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    minutes = int(seconds // 60)
    remain = seconds - minutes * 60
    return f"{minutes} min {remain:.2f} s"


POINT_ORDER = list("ABCDEFGHI")
DEFAULT_POINTS = {
    "A": np.array([2233.905, 263.982], float),
    "B": np.array([2154.859, 294.381], float),
    "C": np.array([2138.375, 337.672], float),
    "D": np.array([2116.659, 238.934], float),
    "E": np.array([2117.604, 168.794], float),
    "F": np.array([2105.435, 268.474], float),
    "G": np.array([2109.457, 336.396], float),
    "H": np.array([2149.539, 285.269], float),
    "I": np.array([2117.076, 195.461], float),
}
DEFAULT_Y_VALUES = {p: 0.0 for p in POINT_ORDER}

LINK_SEGMENTS = [
    ("B-C", "B", "C"), ("C-D", "C", "D"), ("D-E", "D", "E"),
    ("E-F", "E", "F"), ("F-G", "F", "G"), ("G-D", "G", "D"),
    ("D-I", "D", "I"), ("H-I", "H", "I")
]
DRAW_LINKS = [("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "G"), ("G", "D"), ("D", "I"), ("H", "I"), ("J", "B")]
DEAD_ANGLE_DEFS = [
    ("C-D-G@D", "C", "D", "G"), ("C-D-I@D", "C", "D", "I"),
    ("C-E-F@E", "C", "E", "F"), ("E-F-G@F", "E", "F", "G"),
    ("F-G-D@G", "F", "G", "D"), ("D-I-H@I", "D", "I", "H")
]


@dataclass
class PointRange:
    enabled: bool = False
    x_min: float = 0.0
    x_max: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0


@dataclass
class Settings:
    output_angle_deg: float = -75.0
    rod_rear_length: float = 70.0
    min_hole_distance: float = 40.0
    min_dead_angle: float = 15.0
    angle_steps: int = 25
    maxiter: int = 100
    popsize: int = 30
    target_mode: str = "length"  # length / height / combined


# =============================
# 几何与运动学
# =============================

def R(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]], float)


def dist(p, q) -> float:
    return float(np.linalg.norm(p - q))


def unit(v):
    n = float(np.linalg.norm(v))
    return np.array([1.0, 0.0]) if n < 1e-12 else v / n


def angle_between(u, v) -> float:
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    c = max(-1.0, min(1.0, float(np.dot(u, v) / (nu * nv))))
    return math.degrees(math.acos(c))


def link_lengths(points):
    return {name: dist(points[a], points[b]) for name, a, b in LINK_SEGMENTS}


def min_hole_distance(points):
    vals = link_lengths(points)
    k = min(vals, key=lambda name: vals[name])
    return vals[k], k


def state_dead_angles(st):
    out = {}
    for name, p1, pc, p2 in DEAD_ANGLE_DEFS:
        th = angle_between(st[p1] - st[pc], st[p2] - st[pc])
        out[name] = min(th, 180.0 - th)
    return out


def min_dead_angle(states):
    mn, key, step = 999.0, "", 0
    for st in states:
        for k, v in state_dead_angles(st).items():
            if v < mn:
                mn, key, step = v, k, int(st.get("step", 0))
    return mn, key, step


def rod_points_for_state(A, B_closed, B_now, rear_length_closed):
    total = dist(A, B_closed) + max(0.0, float(rear_length_closed))
    direction = unit(B_now - A)
    J = B_now - total * direction
    ab_now = dist(A, B_now)
    return J, ab_now, total - ab_now, total


def solve_path(points: Dict[str, np.ndarray], settings: Settings):
    n = max(3, int(settings.angle_steps))
    pts = {k: np.array(v, float).copy() for k, v in points.items()}
    A, C, H = pts["A"], pts["C"], pts["H"]
    B0, D0, E0, F0, G0, I0 = pts["B"], pts["D"], pts["E"], pts["F"], pts["G"], pts["I"]

    L_EF, L_HI = dist(E0, F0), dist(H, I0)
    vB, vD, vE = B0 - C, D0 - C, E0 - C
    vG, vI, vFG = G0 - D0, I0 - D0, F0 - G0

    states, prev, max_res = [], np.array([0.0, 0.0]), 0.0
    for k in range(n + 1):
        phi = math.radians(settings.output_angle_deg * k / n)

        def residual(ab):
            alpha, beta = ab
            r1, r4 = R(alpha), R(beta)
            D = C + r1.dot(vD)
            E = C + r1.dot(vE)
            G = D + r4.dot(vG)
            I = D + r4.dot(vI)
            F = G + R(phi).dot(vFG)
            return np.array([np.linalg.norm(I - H) - L_HI, np.linalg.norm(F - E) - L_EF], float)

        guesses = [
            prev, prev + [.03, .03], prev + [-.03, -.03], prev + [.03, -.03], prev + [-.03, .03],
            np.array([0., 0.]), np.array([.5, 0.]), np.array([-.5, 0.]), np.array([0., .5]), np.array([0., -.5])
        ]
        cand = []
        for g in guesses:
            res = least_squares(residual, g, max_nfev=120, xtol=1e-9, ftol=1e-9, gtol=1e-9)
            err = float(np.linalg.norm(residual(res.x)))
            if err < 1e-4:
                cand.append((float(np.linalg.norm(res.x - prev)), err, res.x))
        if not cand:
            return None, False, f"运动学求解失败：step={k}"

        _, err, ab = sorted(cand, key=lambda x: (x[0], x[1]))[0]
        max_res = max(max_res, err)
        prev = ab
        alpha, beta = ab
        r1, r4 = R(alpha), R(beta)
        B = C + r1.dot(vB)
        D = C + r1.dot(vD)
        E = C + r1.dot(vE)
        G = D + r4.dot(vG)
        I = D + r4.dot(vI)
        F = G + R(phi).dot(vFG)
        J, ab_len, rear_now, total = rod_points_for_state(A, B0, B, settings.rod_rear_length)
        states.append({
            "step": k, "angle_deg": settings.output_angle_deg * k / n,
            "A": A.copy(), "B": B, "C": C.copy(), "D": D, "E": E,
            "F": F, "G": G, "H": H.copy(), "I": I, "J": J,
            "AB_len": ab_len, "rod_rear_len": rear_now, "rod_total_len": total,
            "solve_residual": err
        })
    return states, True, f"max residual={max_res:.6g}"


def evaluate(points, settings):
    states, ok, msg = solve_path(points, settings)
    if not ok:
        return {"ok": False, "message": msg}
    f0, fop = points["F"], states[-1]["F"]
    mh, mhk = min_hole_distance(points)
    md, dk, ds = min_dead_angle(states)
    return {
        "ok": True, "message": msg, "states": states,
        "dx": float(fop[0] - f0[0]), "dz": float(fop[1] - f0[1]), "f_open": fop,
        "min_hole": mh, "min_hole_key": mhk, "min_dead": md, "dead_key": dk, "dead_step": ds,
        "ab_closed": float(states[0]["AB_len"]), "ab_open": float(states[-1]["AB_len"]),
        "stroke": float(states[-1]["AB_len"] - states[0]["AB_len"]),
        "rod_rear_min": min(float(s["rod_rear_len"]) for s in states)
    }


def enabled_variables(ranges):
    vs = []
    for p in POINT_ORDER:
        r = ranges[p]
        if not r.enabled:
            continue
        if abs(r.x_max - r.x_min) > 1e-12:
            vs.append((p, "x"))
        if abs(r.z_max - r.z_min) > 1e-12:
            vs.append((p, "z"))
    return vs


def variable_bounds(ranges):
    bs = []
    for p, ax in enabled_variables(ranges):
        r = ranges[p]
        bs.append((r.x_min, r.x_max) if ax == "x" else (r.z_min, r.z_max))
    return bs


def apply_vector(points, ranges, x):
    pts = {k: np.array(v, float).copy() for k, v in points.items()}
    for val, (p, ax) in zip(x, enabled_variables(ranges)):
        pts[p][0 if ax == "x" else 1] += float(val)
    return pts


def improvement_metrics(base_eval, res):
    # V1.3 固定定义：长度方向 = F点打开后向左，即 X 变小；高度方向 = F点打开后向上，即 Z 变大。
    length_improve = float(base_eval["f_open"][0] - res["f_open"][0])
    height_improve = float(res["f_open"][1] - base_eval["f_open"][1])
    return length_improve, height_improve


def direction_pass(settings: Settings, length_improve: float, height_improve: float, tol: float = 1e-7):
    if settings.target_mode == "length":
        return length_improve >= -tol
    if settings.target_mode == "height":
        return height_improve >= -tol
    # combined：长度不能更靠右，高度不能更低。
    return length_improve >= -tol and height_improve >= -tol


def objective_score(settings: Settings, length_improve: float, height_improve: float):
    if settings.target_mode == "height":
        return height_improve
    if settings.target_mode == "combined":
        return length_improve + height_improve
    return length_improve


def constraints_pass(points, res, settings: Settings):
    if not res.get("ok", False):
        return False
    if res["min_hole"] < settings.min_hole_distance:
        return False
    if res["rod_rear_min"] < 0:
        return False
    return True


def build_candidate_record(points, vector, variables, base_eval, res, settings, label=""):
    length_improve, height_improve = improvement_metrics(base_eval, res)
    score = objective_score(settings, length_improve, height_improve)
    return {
        "label": label,
        "points": points,
        "vector": [float(v) for v in vector],
        "variables": list(variables),
        "eval": res,
        "score": float(score),
        "length_improve": float(length_improve),
        "height_improve": float(height_improve),
        "direction_pass": direction_pass(settings, length_improve, height_improve),
        "constraint_pass": constraints_pass(points, res, settings)
    }


def optimize_case(points, ranges, settings: Settings):
    base = evaluate(points, settings)
    if not base.get("ok", False):
        raise RuntimeError("原始方案无法求解：" + base.get("message", ""))

    bounds = variable_bounds(ranges)
    variables = enabled_variables(ranges)
    zero = [0.0] * len(bounds)
    baseline_record = build_candidate_record(points, zero, variables, base, base, settings, label="原方案")
    all_records = [baseline_record]

    if not bounds:
        return {
            "baseline": base, "optimized": base, "optimized_points": points,
            "variables": [], "vector": [], "candidates": [baseline_record],
            "message": "未启用优化变量，已返回原方案。", "is_baseline_fallback": True
        }

    def obj(x):
        pts = apply_vector(points, ranges, x)
        res = evaluate(pts, settings)
        if not res.get("ok", False):
            return 1e12
        length_improve, height_improve = improvement_metrics(base, res)
        score = objective_score(settings, length_improve, height_improve)
        pen = 0.0

        # 孔距和丝杆后方长度仍作为硬性可用条件。
        if res["min_hole"] < settings.min_hole_distance:
            pen += 1e8 * (settings.min_hole_distance - res["min_hole"] + 1.0) ** 2
        if res["rod_rear_min"] < 0:
            pen += 1e8 * (abs(res["rod_rear_min"]) + 1.0) ** 2

        # V1.3：死点角不作为硬约束，只作为结果风险提示。
        # V1.3：高度限制不再提供复选框，由目标模式自动决定。
        if not direction_pass(settings, length_improve, height_improve):
            if settings.target_mode == "length":
                pen += 1e8 * (abs(min(0.0, length_improve)) + 1.0) ** 2
            elif settings.target_mode == "height":
                pen += 1e8 * (abs(min(0.0, height_improve)) + 1.0) ** 2
            else:
                pen += 1e8 * (abs(min(0.0, length_improve)) + abs(min(0.0, height_improve)) + 1.0) ** 2

        all_records.append(build_candidate_record(pts, x, variables, base, res, settings))
        return -score + pen

    opt = differential_evolution(
        obj, bounds=bounds, seed=42, maxiter=max(1, int(settings.maxiter)),
        popsize=max(3, int(settings.popsize)), tol=0.01, polish=False,
        updating="immediate", workers=1
    )

    opt_pts = apply_vector(points, ranges, opt.x)
    opt_eval = evaluate(opt_pts, settings)
    all_records.append(build_candidate_record(opt_pts, opt.x, variables, base, opt_eval, settings, label="DE最终"))

    # 去重：按优化变量四舍五入去重，并保留分数最高者。
    unique = {}
    for rec in all_records:
        key = tuple(round(v, 4) for v in rec["vector"])
        old = unique.get(key)
        if old is None or rec["score"] > old["score"]:
            unique[key] = rec

    valid = [r for r in unique.values() if r["constraint_pass"] and r["direction_pass"]]
    if not valid:
        valid = [baseline_record]

    # 排序：先目标分数，再孔距，再死点角。基准方案永远可兜底。
    valid_sorted = sorted(valid, key=lambda r: (r["score"], r["eval"]["min_hole"], r["eval"]["min_dead"]), reverse=True)
    best = valid_sorted[0]

    # 如果最佳方案没有真正超过原方案，回退原方案；避免输出“负优化”。
    eps = 1e-7
    if best["score"] <= baseline_record["score"] + eps:
        best = baseline_record
        message = "未找到比原方案更优且满足方向规则的方案，已返回原方案。"
        fallback = True
    else:
        message = "优化完成。"
        fallback = False

    top10 = valid_sorted[:10]
    if best is baseline_record and all(r is not baseline_record for r in top10):
        top10 = [baseline_record] + top10[:9]

    return {
        "baseline": base,
        "optimized": best["eval"],
        "optimized_points": best["points"],
        "variables": variables,
        "vector": best["vector"],
        "candidates": top10,
        "message": message,
        "is_baseline_fallback": fallback
    }


# =============================
# 后台线程
# =============================

class OptimizeWorker(QObject):
    result_ready = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, points, ranges, settings):
        super().__init__()
        self.points = points
        self.ranges = ranges
        self.settings = settings

    def run(self):
        start_time = time.perf_counter()
        try:
            result = optimize_case(self.points, self.ranges, self.settings)
            result["elapsed_seconds"] = time.perf_counter() - start_time
            self.result_ready.emit(result)
        except Exception as exc:
            self.error.emit(f"{str(exc)}\n本次计算用时：{format_elapsed(time.perf_counter() - start_time)}")
        finally:
            self.finished.emit()


# =============================
# 视图
# =============================

class ReferenceImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap_original = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(260)
        self.setStyleSheet("QLabel { background-color: #f7f7f7; border: 1px solid #cccccc; }")

    def set_image(self, path):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self._pixmap_original = None
            self.setText("示意图加载失败：图片文件损坏")
            return
        self._pixmap_original = pixmap
        self._update_scaled()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled()

    def _update_scaled(self):
        if self._pixmap_original is None:
            return
        scaled = self._pixmap_original.scaled(max(1, self.width() - 10), max(1, self.height() - 10), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)


class LinkageView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.baseline_states = None
        self.optimized_states = None
        self.step = 0
        self.setMinimumHeight(430)

    def set_states(self, baseline, optimized=None):
        self.baseline_states = baseline
        self.optimized_states = optimized
        self.step = 0
        self.update()

    def set_step(self, step):
        self.step = int(step)
        self.update()

    def _all_points(self):
        pts = []
        for states in [self.baseline_states, self.optimized_states]:
            if states:
                for st in states:
                    for p in "ABCDEFGHIJ":
                        if p in st:
                            pts.append(st[p])
        return pts

    def _tr(self):
        pts = self._all_points()
        if not pts:
            return lambda p: (0, 0)
        xs = [float(p[0]) for p in pts]
        zs = [float(p[1]) for p in pts]
        minx, maxx, minz, maxz = min(xs) - 40, max(xs) + 40, min(zs) - 40, max(zs) + 40
        w, h = max(1, self.width() - 30), max(1, self.height() - 30)
        s = min(w / max(1e-9, maxx - minx), h / max(1e-9, maxz - minz))
        ox = 15 - minx * s + (w - (maxx - minx) * s) / 2
        oy = 15 + maxz * s + (h - (maxz - minz) * s) / 2
        return lambda p: (float(p[0]) * s + ox, oy - float(p[1]) * s)

    def _draw_f_trajectory(self, painter, states, color, dashed=False):
        if not states or len(states) < 2:
            return
        tr = self._tr()
        pen = QPen(color, 1)
        if dashed:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        prev = None
        for st in states:
            if "F" not in st:
                continue
            x, y = tr(st["F"])
            if prev is not None:
                painter.drawLine(int(prev[0]), int(prev[1]), int(x), int(y))
            prev = (x, y)

    def _draw_state(self, painter, st, color, dashed=False, width=2):
        tr = self._tr()
        pen = QPen(color, width)
        if dashed:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        for a, b in DRAW_LINKS:
            if a in st and b in st:
                x1, y1 = tr(st[a])
                x2, y2 = tr(st[b])
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color, 1))
        for p in "ABCDEFGHIJ":
            if p in st:
                x, y = tr(st[p])
                painter.drawEllipse(int(x) - 4, int(y) - 4, 8, 8)
                painter.drawText(int(x) + 6, int(y) - 6, p)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(250, 250, 250))
        painter.setFont(QFont("Microsoft YaHei", 9))
        if not self.baseline_states:
            painter.setPen(QColor(80, 80, 80))
            painter.drawText(self.rect(), Qt.AlignCenter, "输入点位后点击“运行优化”显示连杆动画")
            return

        self._draw_f_trajectory(painter, self.baseline_states, QColor(130, 130, 130), dashed=True)
        if self.optimized_states:
            self._draw_f_trajectory(painter, self.optimized_states, QColor(20, 80, 160), dashed=False)

        i = max(0, min(self.step, len(self.baseline_states) - 1))
        self._draw_state(painter, self.baseline_states[i], QColor(120, 120, 120), True, 2)
        if self.optimized_states:
            self._draw_state(painter, self.optimized_states[i], QColor(20, 80, 160), False, 2)
        painter.setPen(QColor(40, 40, 40))
        painter.drawText(12, 24, f"当前角度：{self.baseline_states[i].get('angle_deg',0):.1f}°    灰色虚线：优化前    蓝色实线：优化后    F点轨迹已显示    J-A-B：丝杆")


# =============================
# 主窗口
# =============================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Legrest Linkage Optimizer - 连杆优化桌面原型 {APP_VERSION}")
        self.resize(1540, 940)
        self.points = {k: v.copy() for k, v in DEFAULT_POINTS.items()}
        self.y_values = dict(DEFAULT_Y_VALUES)
        self.ranges = self.default_ranges()
        self.settings = Settings()
        self.result = None
        self.worker_thread = None
        self.worker = None

        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        spl = QSplitter(Qt.Horizontal)
        main.addWidget(spl)

        left_content = QWidget()
        ll = QVBoxLayout(left_content)
        ll.setContentsMargins(8, 8, 8, 8)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_content)
        spl.addWidget(left_scroll)

        right = QWidget()
        rl = QVBoxLayout(right)
        spl.addWidget(right)
        spl.setSizes([590, 950])

        self.build_reference_image(ll)
        self.build_point_table(ll)
        self.build_range_table(ll)
        self.build_settings(ll)
        self.build_io_buttons(ll)
        self.build_run_buttons(ll)
        ll.addStretch(1)

        self.view = LinkageView()
        rl.addWidget(self.view)
        row = QHBoxLayout()
        self.step_slider = QSlider(Qt.Horizontal)
        self.step_slider.setMinimum(0)
        self.step_slider.setMaximum(self.settings.angle_steps)
        self.step_slider.valueChanged.connect(self.on_slider)
        row.addWidget(QLabel("动画进度"))
        row.addWidget(self.step_slider)
        rl.addLayout(row)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(145)
        rl.addWidget(self.summary)
        self.build_result_table(rl)
        self.build_candidate_table(rl)

        self.timer = QTimer()
        self.timer.timeout.connect(self.play_next)
        self.load_to_ui()
        self.summary.setText(
            f"已加载默认样例（{APP_VERSION}）。\n"
            "V1.31规则：长度优先默认F点向左，高度优先默认F点向上，综合优先要求不向右且不向下。\n"
            "死点角仅作为风险提示，不再作为默认硬约束。"
        )

    def default_ranges(self):
        r = {p: PointRange(False, 0, 0, 0, 0) for p in POINT_ORDER}
        r["D"] = PointRange(True, 0, 0, -2, 2)
        r["E"] = PointRange(True, -1, 1, 0, 2)
        r["G"] = PointRange(True, -1, 1, -1, 1)
        r["I"] = PointRange(True, -1, 1, 0, 2)
        return r

    def build_reference_image(self, layout):
        g = QGroupBox("0. 点位示意图 / 机构说明")
        l = QVBoxLayout(g)
        tip = QLabel("请参考下图理解 A、B、C、D、E、F、G、H、I 各点位置。F 点为输出点，F-G 为输出连杆。")
        tip.setWordWrap(True)
        l.addWidget(tip)
        self.reference_image_label = ReferenceImageLabel()
        image_path = resource_path("resources/linkage_points_guide.png")
        if os.path.exists(image_path):
            self.reference_image_label.set_image(image_path)
        else:
            self.reference_image_label.setText("未找到点位示意图：resources/linkage_points_guide.png")
        l.addWidget(self.reference_image_label)
        layout.addWidget(g)

    def build_point_table(self, layout):
        g = QGroupBox("1. 关闭状态 A-I 点位坐标，单位 mm")
        l = QVBoxLayout(g)
        self.point_table = QTableWidget(len(POINT_ORDER), 4)
        self.point_table.setHorizontalHeaderLabels(["点位", "X", "Y(CATIA保留)", "Z"])
        self.point_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for row, p in enumerate(POINT_ORDER):
            it = QTableWidgetItem(p)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.point_table.setItem(row, 0, it)
        l.addWidget(self.point_table)
        layout.addWidget(g)

    def build_range_table(self, layout):
        g = QGroupBox("2. 可修改点与修改范围")
        l = QVBoxLayout(g)
        self.range_table = QTableWidget(len(POINT_ORDER), 6)
        self.range_table.setHorizontalHeaderLabels(["点位", "启用", "X最小", "X最大", "Z最小", "Z最大"])
        self.range_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for row, p in enumerate(POINT_ORDER):
            it = QTableWidgetItem(p)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.range_table.setItem(row, 0, it)
            en = QTableWidgetItem()
            en.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            en.setCheckState(Qt.Unchecked)
            self.range_table.setItem(row, 1, en)
        l.addWidget(self.range_table)
        layout.addWidget(g)

    def build_settings(self, layout):
        g = QGroupBox("3. 目标与约束")
        grid = QGridLayout(g)
        self.angle_spin = QDoubleSpinBox(); self.angle_spin.setRange(-180, 180); self.angle_spin.setDecimals(2); self.angle_spin.setValue(-75); self.angle_spin.setSuffix(" °")
        self.rod_spin = QDoubleSpinBox(); self.rod_spin.setRange(0, 500); self.rod_spin.setDecimals(2); self.rod_spin.setValue(70); self.rod_spin.setSuffix(" mm")
        self.hole_spin = QDoubleSpinBox(); self.hole_spin.setRange(0, 500); self.hole_spin.setDecimals(2); self.hole_spin.setValue(40); self.hole_spin.setSuffix(" mm")
        self.dead_spin = QDoubleSpinBox(); self.dead_spin.setRange(0, 90); self.dead_spin.setDecimals(2); self.dead_spin.setValue(15); self.dead_spin.setSuffix(" °")
        self.target_combo = QComboBox()
        self.target_combo.addItem("长度优先：F点向左最大", "length")
        self.target_combo.addItem("高度优先：F点向上最大", "height")
        self.target_combo.addItem("综合优先：不向右、不向下，长度+高度最大", "combined")
        self.step_spin = QSpinBox(); self.step_spin.setRange(5, 80); self.step_spin.setValue(25)
        self.maxiter_spin = QSpinBox(); self.maxiter_spin.setRange(1, 1000); self.maxiter_spin.setValue(100)
        self.popsize_spin = QSpinBox(); self.popsize_spin.setRange(3, 100); self.popsize_spin.setValue(30)

        grid.addWidget(QLabel("F-G打开角度"), 0, 0); grid.addWidget(self.angle_spin, 0, 1)
        grid.addWidget(QLabel("A点后方丝杆长度"), 0, 2); grid.addWidget(self.rod_spin, 0, 3)
        grid.addWidget(QLabel("最小孔距"), 1, 0); grid.addWidget(self.hole_spin, 1, 1)
        grid.addWidget(QLabel("最小死点角(仅提示)"), 1, 2); grid.addWidget(self.dead_spin, 1, 3)
        grid.addWidget(QLabel("优化目标"), 2, 0); grid.addWidget(self.target_combo, 2, 1, 1, 3)
        grid.addWidget(QLabel("动画步数"), 3, 0); grid.addWidget(self.step_spin, 3, 1)
        grid.addWidget(QLabel("优化迭代"), 3, 2); grid.addWidget(self.maxiter_spin, 3, 3)
        grid.addWidget(QLabel("种群规模"), 4, 0); grid.addWidget(self.popsize_spin, 4, 1)
        layout.addWidget(g)

    def build_io_buttons(self, layout):
        g = QGroupBox("4. CSV与配置")
        v = QVBoxLayout(g)
        row1 = QHBoxLayout()
        self.btn_import_csv = QPushButton("导入CSV点坐标")
        self.btn_export_current = QPushButton("导出当前点CSV")
        self.btn_export_opt = QPushButton("导出优化点CSV")
        self.btn_import_csv.clicked.connect(self.import_points_csv)
        self.btn_export_current.clicked.connect(self.export_current_csv)
        self.btn_export_opt.clicked.connect(self.export_optimized_csv)
        for b in [self.btn_import_csv, self.btn_export_current, self.btn_export_opt]:
            row1.addWidget(b)
        row2 = QHBoxLayout()
        self.btn_save_config = QPushButton("保存方案配置")
        self.btn_load_config = QPushButton("读取方案配置")
        self.btn_save_config.clicked.connect(self.save_config)
        self.btn_load_config.clicked.connect(self.load_config)
        row2.addWidget(self.btn_save_config)
        row2.addWidget(self.btn_load_config)
        v.addLayout(row1)
        v.addLayout(row2)
        layout.addWidget(g)

    def build_run_buttons(self, layout):
        row = QHBoxLayout()
        self.btn_default = QPushButton("恢复默认样例")
        self.btn_run = QPushButton("运行优化")
        self.btn_play = QPushButton("播放动画")
        self.btn_copy = QPushButton("复制优化后坐标")
        self.btn_default.clicked.connect(self.reload_default)
        self.btn_run.clicked.connect(self.run_optimization)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_copy.clicked.connect(self.copy_result)
        for b in [self.btn_default, self.btn_run, self.btn_play, self.btn_copy]:
            row.addWidget(b)
        layout.addLayout(row)

    def build_result_table(self, layout):
        g = QGroupBox("优化后所有点坐标")
        l = QVBoxLayout(g)
        self.result_table = QTableWidget(len(POINT_ORDER), 6)
        self.result_table.setHorizontalHeaderLabels(["点位", "优化前X", "优化前Z", "优化后X", "优化后Z", "变化量"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        l.addWidget(self.result_table)
        layout.addWidget(g)

    def build_candidate_table(self, layout):
        g = QGroupBox("Top 10候选方案")
        l = QVBoxLayout(g)
        self.candidate_table = QTableWidget(0, 10)
        self.candidate_table.setHorizontalHeaderLabels(["Rank", "说明", "Score", "长度改善", "高度改善", "F_open_X", "F_open_Z", "最小孔距", "最小死点角", "变量"])
        self.candidate_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        l.addWidget(self.candidate_table)
        layout.addWidget(g)

    def load_to_ui(self):
        for row, p in enumerate(POINT_ORDER):
            pt = self.points[p]
            self.point_table.setItem(row, 1, QTableWidgetItem(f"{pt[0]:.3f}"))
            self.point_table.setItem(row, 2, QTableWidgetItem(f"{self.y_values.get(p, 0.0):.3f}"))
            self.point_table.setItem(row, 3, QTableWidgetItem(f"{pt[1]:.3f}"))
            rg = self.ranges[p]
            self.range_table.item(row, 1).setCheckState(Qt.Checked if rg.enabled else Qt.Unchecked)
            for col, val in enumerate([rg.x_min, rg.x_max, rg.z_min, rg.z_max], 2):
                self.range_table.setItem(row, col, QTableWidgetItem(f"{val:.3f}"))
        self.angle_spin.setValue(self.settings.output_angle_deg)
        self.rod_spin.setValue(self.settings.rod_rear_length)
        self.hole_spin.setValue(self.settings.min_hole_distance)
        self.dead_spin.setValue(self.settings.min_dead_angle)
        self.step_spin.setValue(self.settings.angle_steps)
        self.maxiter_spin.setValue(self.settings.maxiter)
        self.popsize_spin.setValue(self.settings.popsize)
        idx = self.target_combo.findData(self.settings.target_mode)
        if idx >= 0:
            self.target_combo.setCurrentIndex(idx)

    def read_ui(self):
        pts, yvals, ranges = {}, {}, {}
        for row, p in enumerate(POINT_ORDER):
            try:
                x = float(self.point_table.item(row, 1).text())
                y = float(self.point_table.item(row, 2).text())
                z = float(self.point_table.item(row, 3).text())
            except Exception:
                raise ValueError(f"{p} 点坐标输入错误")
            pts[p] = np.array([x, z], float)
            yvals[p] = y
            en = self.range_table.item(row, 1).checkState() == Qt.Checked
            try:
                vals = [float(self.range_table.item(row, c).text()) for c in range(2, 6)]
            except Exception:
                raise ValueError(f"{p} 点修改范围输入错误")
            if vals[0] > vals[1] or vals[2] > vals[3]:
                raise ValueError(f"{p} 点修改范围下限大于上限")
            ranges[p] = PointRange(en, vals[0], vals[1], vals[2], vals[3])
        st = Settings(
            output_angle_deg=float(self.angle_spin.value()),
            rod_rear_length=float(self.rod_spin.value()),
            min_hole_distance=float(self.hole_spin.value()),
            min_dead_angle=float(self.dead_spin.value()),
            angle_steps=int(self.step_spin.value()),
            maxiter=int(self.maxiter_spin.value()),
            popsize=int(self.popsize_spin.value()),
            target_mode=str(self.target_combo.currentData())
        )
        return pts, yvals, ranges, st

    # ---------- CSV / Config ----------
    def _read_csv_rows(self, file_path):
        for enc in ["utf-8-sig", "gbk", "utf-8"]:
            try:
                with open(file_path, "r", newline="", encoding=enc) as f:
                    return list(csv.DictReader(f))
            except UnicodeDecodeError:
                continue
        with open(file_path, "r", newline="") as f:
            return list(csv.DictReader(f))

    def import_points_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "导入CSV点坐标", "", "CSV Files (*.csv);;All Files (*)")
        if not file_path:
            return
        try:
            rows = self._read_csv_rows(file_path)
            if not rows:
                raise ValueError("CSV为空或没有有效数据。")
            new_points, new_y = {}, {}
            headers = set(rows[0].keys())
            use_opt = {"Opt_X", "Opt_Y", "Opt_Z"}.issubset(headers)
            for row in rows:
                p = str(row.get("Point", "")).strip().replace('"', "")
                if p not in POINT_ORDER:
                    continue
                if use_opt:
                    x = float(row["Opt_X"]); y = float(row.get("Opt_Y", 0.0)); z = float(row["Opt_Z"])
                else:
                    x = float(row["X"]); y = float(row.get("Y", 0.0)); z = float(row["Z"])
                new_points[p] = np.array([x, z], float)
                new_y[p] = y
            missing = [p for p in POINT_ORDER if p not in new_points]
            if missing:
                raise ValueError("CSV缺少点：" + ", ".join(missing))
            self.points = new_points
            self.y_values = new_y
            self.result = None
            self.load_to_ui()
            self.view.set_states(None, None)
            self.result_table.clearContents()
            self.candidate_table.setRowCount(0)
            self.summary.setText(f"已导入CSV：\n{file_path}\n\n说明：软件X=CSV X，软件Z=CSV Z，Y值保留用于导出给CATIA。")
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))

    def export_current_csv(self):
        try:
            self.points, self.y_values, self.ranges, self.settings = self.read_ui()
            file_path, _ = QFileDialog.getSaveFileName(self, "导出当前点CSV", "catia_points_export.csv", "CSV Files (*.csv)")
            if not file_path:
                return
            if not file_path.lower().endswith(".csv"):
                file_path += ".csv"
            with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["Point", "X", "Y", "Z"])
                for p in POINT_ORDER:
                    writer.writerow([p, f"{self.points[p][0]:.3f}", f"{self.y_values.get(p,0.0):.3f}", f"{self.points[p][1]:.3f}"])
            QMessageBox.information(self, "导出完成", f"当前点CSV已导出：\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def export_optimized_csv(self):
        try:
            if self.result:
                opt_points = self.result["optimized_points"]
            else:
                opt_points = self.points
            file_path, _ = QFileDialog.getSaveFileName(self, "导出优化点CSV", "optimized_points.csv", "CSV Files (*.csv)")
            if not file_path:
                return
            if not file_path.lower().endswith(".csv"):
                file_path += ".csv"
            with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["Point", "Base_X", "Base_Y", "Base_Z", "Opt_X", "Opt_Y", "Opt_Z", "Delta_X", "Delta_Z"])
                for p in POINT_ORDER:
                    b = self.points[p]
                    o = opt_points[p]
                    y = self.y_values.get(p, 0.0)
                    writer.writerow([p, f"{b[0]:.3f}", f"{y:.3f}", f"{b[1]:.3f}", f"{o[0]:.3f}", f"{y:.3f}", f"{o[1]:.3f}", f"{o[0]-b[0]:.3f}", f"{o[1]-b[1]:.3f}"])
            QMessageBox.information(self, "导出完成", f"优化点CSV已导出：\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def save_config(self):
        try:
            self.points, self.y_values, self.ranges, self.settings = self.read_ui()
            file_path, _ = QFileDialog.getSaveFileName(self, "保存方案配置", "legrest_config.json", "JSON Files (*.json)")
            if not file_path:
                return
            if not file_path.lower().endswith(".json"):
                file_path += ".json"
            data = {
                "version": APP_VERSION,
                "points": {p: [float(self.points[p][0]), float(self.points[p][1])] for p in POINT_ORDER},
                "y_values": {p: float(self.y_values.get(p, 0.0)) for p in POINT_ORDER},
                "ranges": {p: vars(self.ranges[p]) for p in POINT_ORDER},
                "settings": vars(self.settings)
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "保存完成", f"方案配置已保存：\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    def load_config(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "读取方案配置", "", "JSON Files (*.json);;All Files (*)")
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.points = {p: np.array(data["points"][p], float) for p in POINT_ORDER}
            self.y_values = {p: float(data.get("y_values", {}).get(p, 0.0)) for p in POINT_ORDER}
            self.ranges = {}
            for p in POINT_ORDER:
                r = data["ranges"][p]
                self.ranges[p] = PointRange(bool(r.get("enabled", False)), float(r.get("x_min", 0)), float(r.get("x_max", 0)), float(r.get("z_min", 0)), float(r.get("z_max", 0)))
            s = data.get("settings", {})
            self.settings = Settings(
                output_angle_deg=float(s.get("output_angle_deg", -75.0)),
                rod_rear_length=float(s.get("rod_rear_length", 70.0)),
                min_hole_distance=float(s.get("min_hole_distance", 40.0)),
                min_dead_angle=float(s.get("min_dead_angle", 15.0)),
                angle_steps=int(s.get("angle_steps", 25)),
                maxiter=int(s.get("maxiter", 100)),
                popsize=int(s.get("popsize", 30)),
                target_mode=str(s.get("target_mode", "length"))
            )
            self.result = None
            self.load_to_ui()
            self.view.set_states(None, None)
            self.result_table.clearContents()
            self.candidate_table.setRowCount(0)
            self.summary.setText(f"已读取方案配置：\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))

    # ---------- Run / Results ----------
    def reload_default(self):
        self.points = {k: v.copy() for k, v in DEFAULT_POINTS.items()}
        self.y_values = dict(DEFAULT_Y_VALUES)
        self.ranges = self.default_ranges()
        self.settings = Settings()
        self.result = None
        self.load_to_ui()
        self.view.set_states(None, None)
        self.summary.setText("已恢复默认样例。")
        self.result_table.clearContents()
        self.candidate_table.setRowCount(0)

    def set_running_state(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_import_csv.setEnabled(not running)
        self.btn_load_config.setEnabled(not running)
        self.btn_default.setEnabled(not running)
        self.btn_run.setText("计算中..." if running else "运行优化")

    def run_optimization(self):
        try:
            self.points, self.y_values, self.ranges, self.settings = self.read_ui()
        except Exception as e:
            QMessageBox.critical(self, "输入错误", str(e))
            return
        self.summary.setText("正在后台计算，请稍等。界面不会再长时间未响应。")
        self.set_running_state(True)
        self.worker_thread = QThread()
        self.worker = OptimizeWorker(self.points, self.ranges, self.settings)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.result_ready.connect(self.on_optimization_result)
        self.worker.error.connect(self.on_optimization_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(lambda: self.set_running_state(False))
        self.worker_thread.start()

    def on_optimization_result(self, result):
        self.result = result
        base = result["baseline"]
        opt = result["optimized"]
        opt_pts = result["optimized_points"]
        self.view.set_states(base["states"], opt["states"])
        self.step_slider.setMaximum(len(base["states"]) - 1)
        self.step_slider.setValue(0)
        self.fill_result_table(opt_pts)
        self.fill_candidate_table(result.get("candidates", []))
        self.fill_summary()

    def on_optimization_error(self, msg):
        QMessageBox.critical(self, "计算失败", msg)
        self.summary.setText("计算失败：" + msg)

    def fill_result_table(self, opt_pts):
        for row, p in enumerate(POINT_ORDER):
            b, o = self.points[p], opt_pts[p]
            vals = [p, f"{b[0]:.3f}", f"{b[1]:.3f}", f"{o[0]:.3f}", f"{o[1]:.3f}", f"ΔX={o[0]-b[0]:.3f}, ΔZ={o[1]-b[1]:.3f}"]
            for c, v in enumerate(vals):
                self.result_table.setItem(row, c, QTableWidgetItem(v))

    def fill_candidate_table(self, candidates):
        self.candidate_table.setRowCount(len(candidates))
        for r, rec in enumerate(candidates):
            ev = rec["eval"]
            vec_text = "; ".join(f"{p}_{ax}={v:.3f}" for (p, ax), v in zip(rec.get("variables", []), rec.get("vector", [])))
            vals = [
                str(r + 1),
                rec.get("label", "候选"),
                f"{rec['score']:.3f}",
                f"{rec['length_improve']:.3f}",
                f"{rec['height_improve']:.3f}",
                f"{ev['f_open'][0]:.3f}",
                f"{ev['f_open'][1]:.3f}",
                f"{ev['min_hole']:.3f}",
                f"{ev['min_dead']:.3f}",
                vec_text
            ]
            for c, v in enumerate(vals):
                self.candidate_table.setItem(r, c, QTableWidgetItem(v))

    def fill_summary(self):
        base, opt = self.result["baseline"], self.result["optimized"]
        length_improve, height_improve = improvement_metrics(base, opt)
        target_text = {"length": "长度优先", "height": "高度优先", "combined": "综合优先"}.get(self.settings.target_mode, self.settings.target_mode)
        fallback = "是" if self.result.get("is_baseline_fallback", False) else "否"
        dead_status = "OK" if opt["min_dead"] >= self.settings.min_dead_angle else "Risk"
        elapsed_text = format_elapsed(self.result.get("elapsed_seconds", 0.0))
        lines = [
            self.result.get("message", "优化完成。"),
            f"本次计算用时：{elapsed_text}", "",
            f"目标模式：{target_text}",
            "方向规则：长度=F点打开后向左；高度=F点打开后向上。",
            f"是否回退原方案：{fallback}", "",
            f"原方案 F_open=({base['f_open'][0]:.3f}, {base['f_open'][1]:.3f})",
            f"优化后 F_open=({opt['f_open'][0]:.3f}, {opt['f_open'][1]:.3f})",
            f"长度改善量(向左为正)={length_improve:.3f} mm",
            f"高度改善量(向上为正)={height_improve:.3f} mm", "",
            f"最小孔距={opt['min_hole']:.3f} mm，位置={opt['min_hole_key']}，限制={self.settings.min_hole_distance:.1f} mm",
            f"最小死点角={opt['min_dead']:.3f}°，位置={opt['dead_key']}，step={opt['dead_step']}，参考限制={self.settings.min_dead_angle:.1f}°，状态={dead_status}",
            "说明：V1.31中死点角仅作为风险提示，不作为默认硬约束。", "",
            f"AB关闭有效长度={opt['ab_closed']:.3f} mm",
            f"AB打开有效长度={opt['ab_open']:.3f} mm",
            f"所需丝杆行程={opt['stroke']:.3f} mm",
            f"运动过程中 A 点后方丝杆最小剩余长度={opt['rod_rear_min']:.3f} mm"
        ]
        if self.result["variables"]:
            lines.append("")
            lines.append("输出方案变量：")
            for (p, ax), v in zip(self.result["variables"], self.result["vector"]):
                lines.append(f"  {p}_{ax} = {v:.4f} mm")
        self.summary.setText("\n".join(lines))

    def on_slider(self, value):
        self.view.set_step(value)

    def toggle_play(self):
        if not self.result:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("播放动画")
        else:
            self.timer.start(120)
            self.btn_play.setText("暂停动画")

    def play_next(self):
        v = self.step_slider.value() + 1
        if v > self.step_slider.maximum():
            v = 0
        self.step_slider.setValue(v)

    def copy_result(self):
        if not self.result:
            QMessageBox.information(self, "提示", "还没有优化结果。")
            return
        rows = ["Point\tBase_X\tBase_Y\tBase_Z\tOpt_X\tOpt_Y\tOpt_Z\tDelta_X\tDelta_Z"]
        opt_pts = self.result["optimized_points"]
        for p in POINT_ORDER:
            b, o = self.points[p], opt_pts[p]
            y = self.y_values.get(p, 0.0)
            rows.append(f"{p}\t{b[0]:.3f}\t{y:.3f}\t{b[1]:.3f}\t{o[0]:.3f}\t{y:.3f}\t{o[1]:.3f}\t{o[0]-b[0]:.3f}\t{o[1]-b[1]:.3f}")
        QApplication.clipboard().setText("\n".join(rows))
        QMessageBox.information(self, "已复制", "优化后坐标已复制到剪贴板，可直接粘贴到 Excel。")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Legrest Linkage Optimizer")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
