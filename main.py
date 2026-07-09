import sys, os, math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.optimize import least_squares, differential_evolution
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
    QTextEdit, QMessageBox, QSlider, QDoubleSpinBox, QSpinBox, QComboBox,
    QCheckBox, QSplitter, QScrollArea

)

def resource_path(relative_path: str) -> str:
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

POINT_ORDER = list('ABCDEFGHI')
DEFAULT_POINTS = {
    'A': np.array([2233.905, 263.982], float),
    'B': np.array([2154.859, 294.381], float),
    'C': np.array([2138.375, 337.672], float),
    'D': np.array([2116.659, 238.934], float),
    'E': np.array([2117.604, 168.794], float),
    'F': np.array([2105.435, 268.474], float),
    'G': np.array([2109.457, 336.396], float),
    'H': np.array([2149.539, 285.269], float),
    'I': np.array([2117.076, 195.461], float),
}
LINK_SEGMENTS = [('B-C','B','C'),('C-D','C','D'),('D-E','D','E'),('E-F','E','F'),('F-G','F','G'),('G-D','G','D'),('D-I','D','I'),('H-I','H','I')]
DRAW_LINKS = [('B','C'),('C','D'),('D','E'),('E','F'),('F','G'),('G','D'),('D','I'),('H','I'),('J','B')]
DEAD_ANGLE_DEFS = [('C-D-G@D','C','D','G'),('C-D-I@D','C','D','I'),('C-E-F@E','C','E','F'),('E-F-G@F','E','F','G'),('F-G-D@G','F','G','D'),('D-I-H@I','D','I','H')]

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
    target_mode: str = 'length'
    # 前向方向：+1 表示 X+ 为前，-1 表示 X- 为前。
    # 当前腿托示意中，F 点向左/小 X 通常才是前伸，所以默认 -1。
    x_forward_sign: float = -1.0
    height_not_lower: bool = True
    strict_dead_angle: bool = True

def R(a):
    c,s = math.cos(a), math.sin(a)
    return np.array([[c,-s],[s,c]], float)

def dist(p,q): return float(np.linalg.norm(p-q))

def unit(v):
    n = float(np.linalg.norm(v))
    return np.array([1.0,0.0]) if n < 1e-12 else v/n

def angle_between(u,v):
    nu,nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-12 or nv < 1e-12: return 0.0
    c = max(-1.0, min(1.0, float(np.dot(u,v)/(nu*nv))))
    return math.degrees(math.acos(c))

def link_lengths(points): return {n: dist(points[a],points[b]) for n,a,b in LINK_SEGMENTS}

def min_hole_distance(points):
    vals = link_lengths(points); k = min(vals, key=lambda x: vals[x])
    return vals[k], k

def state_dead_angles(st):
    out = {}
    for name,p1,pc,p2 in DEAD_ANGLE_DEFS:
        th = angle_between(st[p1]-st[pc], st[p2]-st[pc])
        out[name] = min(th, 180.0-th)
    return out

def min_dead_angle(states):
    mn,key,step = 999.0,'',0
    for st in states:
        for k,v in state_dead_angles(st).items():
            if v < mn: mn,key,step = v,k,int(st.get('step',0))
    return mn,key,step

def rod_points_for_state(A, B_closed, B_now, rear_length_closed):
    total = dist(A,B_closed) + max(0.0, float(rear_length_closed))
    direction = unit(B_now-A)
    J = B_now - total*direction
    ab_now = dist(A,B_now)
    return J, ab_now, total-ab_now, total

def solve_path(points: Dict[str,np.ndarray], settings: Settings):
    n = max(3, int(settings.angle_steps))
    pts = {k:np.array(v,float).copy() for k,v in points.items()}
    A,C,H = pts['A'], pts['C'], pts['H']
    B0,D0,E0,F0,G0,I0 = pts['B'],pts['D'],pts['E'],pts['F'],pts['G'],pts['I']
    L_EF, L_HI = dist(E0,F0), dist(H,I0)
    vB,vD,vE = B0-C, D0-C, E0-C
    vG,vI,vFG = G0-D0, I0-D0, F0-G0
    states, prev, max_res = [], np.array([0.0,0.0]), 0.0
    for k in range(n+1):
        phi = math.radians(settings.output_angle_deg*k/n)
        def residual(ab):
            alpha,beta = ab
            r1,r4 = R(alpha), R(beta)
            D = C + r1.dot(vD); E = C + r1.dot(vE)
            G = D + r4.dot(vG); I = D + r4.dot(vI)
            F = G + R(phi).dot(vFG)
            return np.array([np.linalg.norm(I-H)-L_HI, np.linalg.norm(F-E)-L_EF], float)
        guesses = [prev, prev+[.03,.03], prev+[-.03,-.03], prev+[.03,-.03], prev+[-.03,.03], np.array([0.,0.]), np.array([.5,0.]), np.array([-.5,0.]), np.array([0.,.5]), np.array([0.,-.5])]
        cand = []
        for g in guesses:
            res = least_squares(residual, g, max_nfev=120, xtol=1e-9, ftol=1e-9, gtol=1e-9)
            err = float(np.linalg.norm(residual(res.x)))
            if err < 1e-4: cand.append((float(np.linalg.norm(res.x-prev)), err, res.x))
        if not cand: return None, False, f'运动学求解失败：step={k}'
        _,err,ab = sorted(cand, key=lambda x:(x[0],x[1]))[0]
        max_res = max(max_res, err); prev = ab
        alpha,beta = ab; r1,r4 = R(alpha),R(beta)
        B = C+r1.dot(vB); D = C+r1.dot(vD); E = C+r1.dot(vE)
        G = D+r4.dot(vG); I = D+r4.dot(vI); F = G+R(phi).dot(vFG)
        J,ab_len,rear_now,total = rod_points_for_state(A,B0,B,settings.rod_rear_length)
        states.append({'step':k,'angle_deg':settings.output_angle_deg*k/n,'A':A.copy(),'B':B,'C':C.copy(),'D':D,'E':E,'F':F,'G':G,'H':H.copy(),'I':I,'J':J,'AB_len':ab_len,'rod_rear_len':rear_now,'rod_total_len':total,'solve_residual':err})
    return states, True, f'max residual={max_res:.6g}'

def evaluate(points, settings, baseline_open_z: Optional[float]=None):
    states,ok,msg = solve_path(points, settings)
    if not ok: return {'ok':False,'message':msg}
    f0, fop = points['F'], states[-1]['F']
    mh,mhk = min_hole_distance(points); md,dk,ds = min_dead_angle(states)
    return {'ok':True,'message':msg,'states':states,'dx':float(fop[0]-f0[0]),'dz':float(fop[1]-f0[1]),'f_open':fop,'min_hole':mh,'min_hole_key':mhk,'min_dead':md,'dead_key':dk,'dead_step':ds,'ab_closed':float(states[0]['AB_len']),'ab_open':float(states[-1]['AB_len']),'stroke':float(states[-1]['AB_len']-states[0]['AB_len']),'rod_rear_min':min(float(s['rod_rear_len']) for s in states),'height_pass': True if baseline_open_z is None else float(fop[1]) >= float(baseline_open_z)-1e-9}

def enabled_variables(ranges):
    vs=[]
    for p in POINT_ORDER:
        r = ranges[p]
        if not r.enabled: continue
        if abs(r.x_max-r.x_min)>1e-12: vs.append((p,'x'))
        if abs(r.z_max-r.z_min)>1e-12: vs.append((p,'z'))
    return vs

def variable_bounds(ranges):
    bs=[]
    for p,ax in enabled_variables(ranges):
        r=ranges[p]; bs.append((r.x_min,r.x_max) if ax=='x' else (r.z_min,r.z_max))
    return bs

def apply_vector(points, ranges, x):
    pts={k:np.array(v,float).copy() for k,v in points.items()}
    for val,(p,ax) in zip(x, enabled_variables(ranges)):
        pts[p][0 if ax=='x' else 1] += float(val)
    return pts

def optimize_case(points, ranges, settings):
    base = evaluate(points, settings)
    if not base.get('ok',False): raise RuntimeError('原始方案无法求解：'+base.get('message',''))
    base_forward = float(settings.x_forward_sign * base['dx'])
    bounds = variable_bounds(ranges); variables = enabled_variables(ranges)
    if not bounds:
        return {'baseline':base,'optimized':base,'optimized_points':points,'variables':[],'vector':[],'candidates':[],'message':'未启用优化变量。'}
    candidates=[]
    def val(res):
        # length 优先按“前向位移”判断，不再固定认为 +X 就是前。
        # forward_dx = +dx 表示 X+ 为前；forward_dx = -dx 表示 X- 为前。
        forward_dx = settings.x_forward_sign * res['dx']
        if settings.target_mode == 'height':
            return res['dz']
        if settings.target_mode == 'combined':
            return forward_dx + res['dz']
        return forward_dx
    def constraint_penalty(res):
        pen = 0.0
        if res['min_hole'] < settings.min_hole_distance:
            pen += 100000.0*(settings.min_hole_distance-res['min_hole']+1.0)**2
        if settings.strict_dead_angle and res['min_dead'] < settings.min_dead_angle:
            pen += 100000.0*(settings.min_dead_angle-res['min_dead']+1.0)**2
        if settings.height_not_lower and res['f_open'][1] < base['f_open'][1]:
            pen += 100000.0*(base['f_open'][1]-res['f_open'][1]+1.0)**2
        if res['rod_rear_min'] < 0:
            pen += 100000.0*(abs(res['rod_rear_min'])+1.0)**2
        return pen
    def is_feasible(res):
        if not res.get('ok', False):
            return False
        if res['min_hole'] < settings.min_hole_distance - 1e-9:
            return False
        if settings.strict_dead_angle and res['min_dead'] < settings.min_dead_angle - 1e-9:
            return False
        if settings.height_not_lower and res['f_open'][1] < base['f_open'][1] - 1e-9:
            return False
        if res['rod_rear_min'] < -1e-9:
            return False
        return True
    def length_not_worse(res):
        return float(settings.x_forward_sign * res['dx']) >= base_forward - 1e-9
    def obj(x):
        pts = apply_vector(points, ranges, x)
        res = evaluate(pts, settings, baseline_open_z=base['f_open'][1])
        if not res.get('ok',False): return 1e9
        pen = constraint_penalty(res)
        forward_dx = float(settings.x_forward_sign * res['dx'])
        if settings.target_mode == 'length' and forward_dx < base_forward:
            pen += 1000000.0*(base_forward-forward_dx+1.0)**2
        tv = val(res)
        candidates.append({'x':[float(v) for v in x],'target':float(tv),'dx':float(res['dx']),'dz':float(res['dz']),'forward_dx':forward_dx,'min_dead':float(res['min_dead']),'min_hole':float(res['min_hole']),'penalty':float(pen)})
        return -tv + pen
    opt = differential_evolution(obj, bounds=bounds, seed=42, maxiter=max(1,int(settings.maxiter)), popsize=max(3,int(settings.popsize)), tol=0.01, polish=False, updating='immediate', workers=1)
    result_vectors = [np.array(opt.x, dtype=float)]
    for cand in sorted(candidates, key=lambda c: (c['penalty'], -c['target']))[:80]:
        result_vectors.append(np.array(cand['x'], dtype=float))

    best_vector = None
    best_points = points
    best_eval = base
    best_value = val(base)
    for vector in result_vectors:
        pts = apply_vector(points, ranges, vector)
        res = evaluate(pts, settings, baseline_open_z=base['f_open'][1])
        if not is_feasible(res):
            continue
        if settings.target_mode == 'length' and not length_not_worse(res):
            continue
        score = val(res)
        if score > best_value + 1e-9:
            best_value = score
            best_vector = vector
            best_points = pts
            best_eval = res

    if best_vector is None:
        msg = '未找到满足约束且前向位移不低于原方案的更优解，已保留原方案。'
        return {'baseline':base,'optimized':base,'optimized_points':points,'variables':variables,'vector':[0.0 for _ in variables],'candidates':candidates,'message':msg}

    return {'baseline':base,'optimized':best_eval,'optimized_points':best_points,'variables':variables,'vector':[float(v) for v in best_vector],'candidates':candidates,'message':'优化完成。'}


class ReferenceImageLabel(QLabel):
    def __init__(self,parent=None):
        super().__init__(parent); self._pixmap_original=None; self.setAlignment(Qt.AlignCenter); self.setMinimumHeight(260); self.setStyleSheet('QLabel { background-color: #f7f7f7; border: 1px solid #cccccc; }')
    def set_image(self,path):
        pixmap=QPixmap(path)
        if pixmap.isNull():
            self._pixmap_original=None; self.setText('示意图加载失败：图片文件损坏'); return
        self._pixmap_original=pixmap; self._update_scaled()
    def resizeEvent(self,event):
        super().resizeEvent(event); self._update_scaled()
    def _update_scaled(self):
        if self._pixmap_original is None: return
        scaled=self._pixmap_original.scaled(max(1,self.width()-10), max(1,self.height()-10), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

class LinkageView(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent); self.baseline_states=None; self.optimized_states=None; self.step=0; self.setMinimumHeight(430)
    def set_states(self,b,o=None): self.baseline_states=b; self.optimized_states=o; self.step=0; self.update()
    def set_step(self,step): self.step=int(step); self.update()
    def _all_points(self):
        pts=[]
        for states in [self.baseline_states,self.optimized_states]:
            if states:
                for st in states:
                    for p in 'ABCDEFGHIJ':
                        if p in st: pts.append(st[p])
        return pts
    def _tr(self):
        pts=self._all_points()
        if not pts: return lambda p:(0,0)
        xs=[float(p[0]) for p in pts]; zs=[float(p[1]) for p in pts]
        minx,maxx,minz,maxz=min(xs)-40,max(xs)+40,min(zs)-40,max(zs)+40
        w,h=max(1,self.width()-30),max(1,self.height()-30)
        s=min(w/max(1e-9,maxx-minx), h/max(1e-9,maxz-minz))
        ox=15-minx*s+(w-(maxx-minx)*s)/2; oy=15+maxz*s+(h-(maxz-minz)*s)/2
        return lambda p:(float(p[0])*s+ox, oy-float(p[1])*s)
    def _draw_state(self,painter,st,color,dashed=False,width=2):
        tr=self._tr(); pen=QPen(color,width)
        if dashed: pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        for a,b in DRAW_LINKS:
            if a in st and b in st:
                x1,y1=tr(st[a]); x2,y2=tr(st[b]); painter.drawLine(int(x1),int(y1),int(x2),int(y2))
        painter.setBrush(QBrush(color)); painter.setPen(QPen(color,1))
        for p in 'ABCDEFGHIJ':
            if p in st:
                x,y=tr(st[p]); painter.drawEllipse(int(x)-4,int(y)-4,8,8); painter.drawText(int(x)+6,int(y)-6,p)
    def paintEvent(self,event):
        painter=QPainter(self); painter.setRenderHint(QPainter.Antialiasing,True); painter.fillRect(self.rect(),QColor(250,250,250)); painter.setFont(QFont('Microsoft YaHei',9))
        if not self.baseline_states:
            painter.setPen(QColor(80,80,80)); painter.drawText(self.rect(),Qt.AlignCenter,'输入点位后点击“运行优化”显示连杆动画'); return
        i=max(0,min(self.step,len(self.baseline_states)-1)); self._draw_state(painter,self.baseline_states[i],QColor(120,120,120),True,2)
        if self.optimized_states: self._draw_state(painter,self.optimized_states[i],QColor(20,80,160),False,2)
        painter.setPen(QColor(40,40,40)); painter.drawText(12,24,f"当前角度：{self.baseline_states[i].get('angle_deg',0):.1f}°    灰色虚线：优化前    蓝色实线：优化后    J-A-B：丝杆")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle('Legrest Linkage Optimizer - 连杆优化桌面原型'); self.resize(1480,900)
        self.points={k:v.copy() for k,v in DEFAULT_POINTS.items()}; self.ranges=self.default_ranges(); self.settings=Settings(); self.result=None
        central=QWidget(); self.setCentralWidget(central); main=QVBoxLayout(central); spl=QSplitter(Qt.Horizontal); main.addWidget(spl)
        left_content=QWidget(); ll=QVBoxLayout(left_content); ll.setContentsMargins(8,8,8,8)
        left_scroll=QScrollArea(); left_scroll.setWidgetResizable(True); left_scroll.setWidget(left_content); spl.addWidget(left_scroll)
        right=QWidget(); rl=QVBoxLayout(right); spl.addWidget(right); spl.setSizes([560,920])
        self.build_reference_image(ll); self.build_point_table(ll); self.build_range_table(ll); self.build_settings(ll); self.build_buttons(ll); ll.addStretch(1)
        self.view=LinkageView(); rl.addWidget(self.view)
        row=QHBoxLayout(); self.step_slider=QSlider(Qt.Horizontal); self.step_slider.setMinimum(0); self.step_slider.setMaximum(self.settings.angle_steps); self.step_slider.valueChanged.connect(self.on_slider); row.addWidget(QLabel('动画进度')); row.addWidget(self.step_slider); rl.addLayout(row)
        self.summary=QTextEdit(); self.summary.setReadOnly(True); self.summary.setMinimumHeight(150); rl.addWidget(self.summary); self.build_result_table(rl)
        self.timer=QTimer(); self.timer.timeout.connect(self.play_next); self.load_to_ui(); self.summary.setText('已加载默认样例。流程：参考左侧点位示意图 → 输入 A-I 点坐标 → 选择可优化点和范围 → 设置目标和约束 → 点击“运行优化”。\n说明：A 为丝杆经过点，软件按 J-A-B 显示丝杆；默认 A 点后方丝杆长度 70mm；默认前向方向为 X-，即 F 点向左/小 X 为前伸。')
    def default_ranges(self):
        r={p:PointRange(False,0,0,0,0) for p in POINT_ORDER}
        r['D']=PointRange(True,0,0,-2,2); r['E']=PointRange(True,-1,1,0,2); r['G']=PointRange(True,-1,1,-1,1); r['I']=PointRange(True,-1,1,0,2)
        return r

    def build_reference_image(self,layout):
        g=QGroupBox('0. 点位示意图 / 机构说明'); l=QVBoxLayout(g)
        tip=QLabel('请参考下图理解 A、B、C、D、E、F、G、H、I 各点在腿托机构上的对应位置。图中 F 点为输出点，F-G 为输出连杆。')
        tip.setWordWrap(True); l.addWidget(tip)
        self.reference_image_label=ReferenceImageLabel()
        image_path=resource_path('resources/linkage_points_guide.png')
        if os.path.exists(image_path):
            self.reference_image_label.set_image(image_path)
        else:
            self.reference_image_label.setText('未找到点位示意图。\n请确认 resources/linkage_points_guide.png 已放入项目中。')
        l.addWidget(self.reference_image_label); layout.addWidget(g)
    def build_point_table(self,layout):
        g=QGroupBox('1. 关闭状态 A-I 点位坐标，单位 mm'); l=QVBoxLayout(g); self.point_table=QTableWidget(len(POINT_ORDER),3); self.point_table.setHorizontalHeaderLabels(['点位','X','Z']); self.point_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for row,p in enumerate(POINT_ORDER):
            it=QTableWidgetItem(p); it.setFlags(it.flags() & ~Qt.ItemIsEditable); self.point_table.setItem(row,0,it)
        l.addWidget(self.point_table); layout.addWidget(g)
    def build_range_table(self,layout):
        g=QGroupBox('2. 可修改点与修改范围'); l=QVBoxLayout(g); self.range_table=QTableWidget(len(POINT_ORDER),6); self.range_table.setHorizontalHeaderLabels(['点位','启用','X最小','X最大','Z最小','Z最大']); self.range_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for row,p in enumerate(POINT_ORDER):
            it=QTableWidgetItem(p); it.setFlags(it.flags() & ~Qt.ItemIsEditable); self.range_table.setItem(row,0,it)
            en=QTableWidgetItem(); en.setFlags(Qt.ItemIsUserCheckable|Qt.ItemIsEnabled); en.setCheckState(Qt.Unchecked); self.range_table.setItem(row,1,en)
        l.addWidget(self.range_table); layout.addWidget(g)
    def build_settings(self,layout):
        g=QGroupBox('3. 目标与约束'); grid=QGridLayout(g)
        self.angle_spin=QDoubleSpinBox(); self.angle_spin.setRange(-180,180); self.angle_spin.setDecimals(2); self.angle_spin.setValue(-75); self.angle_spin.setSuffix(' °')
        self.rod_spin=QDoubleSpinBox(); self.rod_spin.setRange(0,500); self.rod_spin.setDecimals(2); self.rod_spin.setValue(70); self.rod_spin.setSuffix(' mm')
        self.hole_spin=QDoubleSpinBox(); self.hole_spin.setRange(0,500); self.hole_spin.setDecimals(2); self.hole_spin.setValue(40); self.hole_spin.setSuffix(' mm')
        self.dead_spin=QDoubleSpinBox(); self.dead_spin.setRange(0,90); self.dead_spin.setDecimals(2); self.dead_spin.setValue(15); self.dead_spin.setSuffix(' °')
        self.target_combo=QComboBox(); self.target_combo.addItem('长度优先：F点向前位移最大','length'); self.target_combo.addItem('高度优先：F点向上位移最大','height'); self.target_combo.addItem('综合优先：前向位移 + ΔZ 最大','combined')
        self.forward_combo=QComboBox(); self.forward_combo.addItem('X- 向前（F点向左/小X为前）', -1.0); self.forward_combo.addItem('X+ 向前（大X为前）', 1.0)
        self.height_check=QCheckBox('优化后 F 点打开高度不低于原方案'); self.height_check.setChecked(True)
        self.strict_dead_check=QCheckBox('死点角低于限制时作为硬约束'); self.strict_dead_check.setChecked(True)
        self.step_spin=QSpinBox(); self.step_spin.setRange(5,80); self.step_spin.setValue(25)
        self.maxiter_spin=QSpinBox(); self.maxiter_spin.setRange(1,1000); self.maxiter_spin.setValue(100)
        self.popsize_spin=QSpinBox(); self.popsize_spin.setRange(3,100); self.popsize_spin.setValue(30)
        grid.addWidget(QLabel('F-G打开角度'),0,0); grid.addWidget(self.angle_spin,0,1); grid.addWidget(QLabel('A点后方丝杆长度'),0,2); grid.addWidget(self.rod_spin,0,3)
        grid.addWidget(QLabel('最小孔距'),1,0); grid.addWidget(self.hole_spin,1,1); grid.addWidget(QLabel('最小死点角'),1,2); grid.addWidget(self.dead_spin,1,3)
        grid.addWidget(QLabel('优化目标'),2,0); grid.addWidget(self.target_combo,2,1,1,3)
        grid.addWidget(QLabel('前向方向'),3,0); grid.addWidget(self.forward_combo,3,1,1,3)
        grid.addWidget(self.height_check,4,0,1,2); grid.addWidget(self.strict_dead_check,4,2,1,2)
        grid.addWidget(QLabel('动画步数'),5,0); grid.addWidget(self.step_spin,5,1); grid.addWidget(QLabel('优化迭代'),5,2); grid.addWidget(self.maxiter_spin,5,3); grid.addWidget(QLabel('种群规模'),6,0); grid.addWidget(self.popsize_spin,6,1)
        layout.addWidget(g)
    def build_buttons(self,layout):
        row=QHBoxLayout(); self.btn_default=QPushButton('恢复默认样例'); self.btn_run=QPushButton('运行优化'); self.btn_play=QPushButton('播放动画'); self.btn_copy=QPushButton('复制优化后坐标')
        self.btn_default.clicked.connect(self.reload_default); self.btn_run.clicked.connect(self.run_optimization); self.btn_play.clicked.connect(self.toggle_play); self.btn_copy.clicked.connect(self.copy_result)
        for b in [self.btn_default,self.btn_run,self.btn_play,self.btn_copy]: row.addWidget(b)
        layout.addLayout(row)
    def build_result_table(self,layout):
        g=QGroupBox('优化后所有点坐标'); l=QVBoxLayout(g); self.result_table=QTableWidget(len(POINT_ORDER),6); self.result_table.setHorizontalHeaderLabels(['点位','优化前X','优化前Z','优化后X','优化后Z','变化量']); self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); l.addWidget(self.result_table); layout.addWidget(g)
    def load_to_ui(self):
        for row,p in enumerate(POINT_ORDER):
            pt=self.points[p]; self.point_table.setItem(row,1,QTableWidgetItem(f'{pt[0]:.3f}')); self.point_table.setItem(row,2,QTableWidgetItem(f'{pt[1]:.3f}'))
            rg=self.ranges[p]; self.range_table.item(row,1).setCheckState(Qt.Checked if rg.enabled else Qt.Unchecked)
            for col,val in enumerate([rg.x_min,rg.x_max,rg.z_min,rg.z_max],2): self.range_table.setItem(row,col,QTableWidgetItem(f'{val:.3f}'))
        self.angle_spin.setValue(self.settings.output_angle_deg); self.rod_spin.setValue(self.settings.rod_rear_length); self.hole_spin.setValue(self.settings.min_hole_distance); self.dead_spin.setValue(self.settings.min_dead_angle); self.step_spin.setValue(self.settings.angle_steps); self.maxiter_spin.setValue(self.settings.maxiter); self.popsize_spin.setValue(self.settings.popsize)
        if hasattr(self, 'forward_combo'):
            idx = self.forward_combo.findData(float(self.settings.x_forward_sign))
            self.forward_combo.setCurrentIndex(idx if idx >= 0 else 0)
    def read_ui(self):
        pts={}; ranges={}
        for row,p in enumerate(POINT_ORDER):
            try: pts[p]=np.array([float(self.point_table.item(row,1).text()), float(self.point_table.item(row,2).text())], float)
            except Exception: raise ValueError(f'{p} 点坐标输入错误')
            en=self.range_table.item(row,1).checkState()==Qt.Checked
            try:
                vals=[float(self.range_table.item(row,c).text()) for c in range(2,6)]
            except Exception: raise ValueError(f'{p} 点修改范围输入错误')
            if vals[0]>vals[1] or vals[2]>vals[3]: raise ValueError(f'{p} 点修改范围下限大于上限')
            ranges[p]=PointRange(en,vals[0],vals[1],vals[2],vals[3])
        st=Settings(
            output_angle_deg=float(self.angle_spin.value()),
            rod_rear_length=float(self.rod_spin.value()),
            min_hole_distance=float(self.hole_spin.value()),
            min_dead_angle=float(self.dead_spin.value()),
            angle_steps=int(self.step_spin.value()),
            maxiter=int(self.maxiter_spin.value()),
            popsize=int(self.popsize_spin.value()),
            target_mode=str(self.target_combo.currentData()),
            x_forward_sign=float(self.forward_combo.currentData()),
            height_not_lower=bool(self.height_check.isChecked()),
            strict_dead_angle=bool(self.strict_dead_check.isChecked())
        )
        return pts,ranges,st
    def reload_default(self):
        self.points={k:v.copy() for k,v in DEFAULT_POINTS.items()}; self.ranges=self.default_ranges(); self.settings=Settings(); self.result=None; self.load_to_ui(); self.view.set_states(None,None); self.summary.setText('已恢复默认样例。'); self.result_table.clearContents()
    def run_optimization(self):
        try:
            self.points,self.ranges,self.settings=self.read_ui(); self.summary.setText('正在计算，请稍等。计算期间界面可能短暂无响应。'); QApplication.processEvents()
            self.result=optimize_case(self.points,self.ranges,self.settings); base=self.result['baseline']; opt=self.result['optimized']; opt_pts=self.result['optimized_points']
            self.view.set_states(base['states'],opt['states']); self.step_slider.setMaximum(len(base['states'])-1); self.step_slider.setValue(0); self.fill_result_table(opt_pts); self.fill_summary()
        except Exception as e:
            QMessageBox.critical(self,'计算失败',str(e)); self.summary.setText('计算失败：'+str(e))
    def fill_result_table(self,opt_pts):
        for row,p in enumerate(POINT_ORDER):
            b,o=self.points[p],opt_pts[p]; vals=[p,f'{b[0]:.3f}',f'{b[1]:.3f}',f'{o[0]:.3f}',f'{o[1]:.3f}',f'ΔX={o[0]-b[0]:.3f}, ΔZ={o[1]-b[1]:.3f}']
            for c,v in enumerate(vals): self.result_table.setItem(row,c,QTableWidgetItem(v))
    def fill_summary(self):
        base,opt=self.result['baseline'],self.result['optimized']; lines=[]
        base_forward = self.settings.x_forward_sign * base['dx']
        opt_forward = self.settings.x_forward_sign * opt['dx']
        forward_text = 'X+ 向前' if self.settings.x_forward_sign > 0 else 'X- 向前'
        lines += [
            self.result.get('message', '优化完成。'),
            '',
            f'当前前向方向：{forward_text}',
            f"原方案 F点 原始ΔX={base['dx']:.3f} mm，前向位移={base_forward:.3f} mm，ΔZ={base['dz']:.3f} mm",
            f"优化后 F点 原始ΔX={opt['dx']:.3f} mm，前向位移={opt_forward:.3f} mm，ΔZ={opt['dz']:.3f} mm",
            f"前向改善量={opt_forward-base_forward:.3f} mm，原始ΔX变化={opt['dx']-base['dx']:.3f} mm，ΔZ变化={opt['dz']-base['dz']:.3f} mm",
            '',
            f"最小孔距={opt['min_hole']:.3f} mm，位置={opt['min_hole_key']}，限制={self.settings.min_hole_distance:.1f} mm",
            f"最小死点角={opt['min_dead']:.3f}°，位置={opt['dead_key']}，step={opt['dead_step']}，限制={self.settings.min_dead_angle:.1f}°",
            '',
            f"AB关闭有效长度={opt['ab_closed']:.3f} mm",
            f"AB打开有效长度={opt['ab_open']:.3f} mm",
            f"所需丝杆行程={opt['stroke']:.3f} mm",
            f"运动过程中 A 点后方丝杆最小剩余长度={opt['rod_rear_min']:.3f} mm",
            ''
        ]
        if self.result['variables']:
            lines.append('优化变量：')
            for (p,ax),v in zip(self.result['variables'], self.result['vector']): lines.append(f'  {p}_{ax} = {v:.4f} mm')
        lines.append(''); lines.append('说明：A 为丝杆经过点，软件显示 J-A-B；关闭状态 A 点后方丝杆长度按界面输入值计算；长度优先按“前向方向”计算，不再固定按 +X。')
        self.summary.setText('\n'.join(lines))
    def on_slider(self,value): self.view.set_step(value)
    def toggle_play(self):
        if not self.result: return
        if self.timer.isActive(): self.timer.stop(); self.btn_play.setText('播放动画')
        else: self.timer.start(120); self.btn_play.setText('暂停动画')
    def play_next(self):
        v=self.step_slider.value()+1
        if v>self.step_slider.maximum(): v=0
        self.step_slider.setValue(v)
    def copy_result(self):
        if not self.result: QMessageBox.information(self,'提示','还没有优化结果。'); return
        rows=['Point\tBase_X\tBase_Z\tOpt_X\tOpt_Z\tDelta_X\tDelta_Z']; opt_pts=self.result['optimized_points']
        for p in POINT_ORDER:
            b,o=self.points[p],opt_pts[p]; rows.append(f'{p}\t{b[0]:.3f}\t{b[1]:.3f}\t{o[0]:.3f}\t{o[1]:.3f}\t{o[0]-b[0]:.3f}\t{o[1]-b[1]:.3f}')
        QApplication.clipboard().setText('\n'.join(rows)); QMessageBox.information(self,'已复制','优化后坐标已复制到剪贴板，可直接粘贴到 Excel。')

def main():
    app=QApplication(sys.argv); app.setApplicationName('Legrest Linkage Optimizer'); win=MainWindow(); win.show(); sys.exit(app.exec())

if __name__ == '__main__': main()
