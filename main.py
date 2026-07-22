import os
import sys
import cv2
import numpy as np
from collections import deque
from threading import Thread, Event

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, Line
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.checkbox import CheckBox
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.scrollview import ScrollView
from kivy.utils import platform

# ==================== 去水印算法 (原样复用) ====================
class StructureGuidedInpainter:
    def __init__(self, angle_threshold=50, edge_extend_steps=100):
        self.angle_threshold = angle_threshold
        self.edge_extend_steps = edge_extend_steps

    def inpaint(self, image, mask, radius=None):
        if mask is None or not np.any(mask):
            return image
        mask_binary = (mask > 0).astype(np.uint8)
        result = image.copy()
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges_full = cv2.Canny(gray, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(mask_binary, kernel, iterations=2)
        outer_ring = (dilated_mask - mask_binary) > 0
        reliable_edges = edges_full & outer_ring.astype(np.uint8)
        boundary = cv2.Canny(mask_binary * 255, 100, 200)
        intersection = reliable_edges & boundary
        if np.count_nonzero(intersection) == 0:
            r = max(3, min(int(np.sqrt(np.count_nonzero(mask_binary) / np.pi)), 20))
            inpainted = cv2.inpaint(result, mask_binary, r, cv2.INPAINT_TELEA)
            mf = mask_binary.astype(np.float32)
            m3 = np.stack([mf] * 3, axis=-1)
            result = (result * (1 - m3) + inpainted * m3).astype(np.uint8)
            return cv2.bilateralFilter(result, 5, 50, 50)
        pts = np.column_stack(np.where(intersection > 0))
        extended_lines_mask = np.zeros((h, w), dtype=np.uint8)
        for (y0, x0) in pts:
            if y0 < 1 or y0 >= h - 1 or x0 < 1 or x0 >= w - 1:
                continue
            gx = float(gray[y0, x0 + 1]) - float(gray[y0, x0 - 1])
            gy = float(gray[y0 + 1, x0]) - float(gray[y0 - 1, x0])
            dx, dy = -gy, gx
            length = np.sqrt(dx * dx + dy * dy)
            if length == 0:
                continue
            dx /= length
            dy /= length
            for sign in (1, -1):
                cur_dx = dx * sign
                cur_dy = dy * sign
                dir_history = deque(maxlen=5)
                line_pts = [(x0, y0)]
                cur_x, cur_y = float(x0), float(y0)
                for step in range(1, self.edge_extend_steps):
                    cur_x += cur_dx
                    cur_y += cur_dy
                    ix, iy = int(round(cur_x)), int(round(cur_y))
                    if ix < 0 or ix >= w or iy < 0 or iy >= h:
                        break
                    if mask_binary[iy, ix] == 0:
                        if boundary[iy, ix]:
                            line_pts.append((ix, iy))
                        break
                    line_pts.append((ix, iy))
                    if len(line_pts) >= 3:
                        p1 = np.array(line_pts[-3])
                        p2 = np.array(line_pts[-2])
                        p3 = np.array(line_pts[-1])
                        v1 = p2 - p1
                        v2 = p3 - p2
                        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                        if n1 > 0 and n2 > 0:
                            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                            if np.degrees(np.arccos(cos_a)) > self.angle_threshold:
                                line_pts.pop()
                                break
                    dir_history.append((cur_dx, cur_dy))
                    if len(dir_history) > 1:
                        avg_dx = np.mean([d[0] for d in dir_history])
                        avg_dy = np.mean([d[1] for d in dir_history])
                        cur_dx = cur_dx * 0.3 + avg_dx * 0.7
                        norm = np.sqrt(cur_dx ** 2 + cur_dy ** 2)
                        if norm > 0:
                            cur_dx /= norm
                            cur_dy /= norm
                if len(line_pts) >= 2:
                    pts_arr = np.array(line_pts, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(extended_lines_mask, [pts_arr], False, 255, thickness=1, lineType=cv2.LINE_AA)
        if np.any(extended_lines_mask):
            outer_ring_dilated = cv2.dilate(mask_binary, np.ones((3, 3), np.uint8), iterations=3)
            outer_ring_only = (outer_ring_dilated - mask_binary) > 0
            ring_coords = np.column_stack(np.where(outer_ring_only))
            if len(ring_coords) > 0:
                ring_colors = image[ring_coords[:, 0], ring_coords[:, 1]]
                lines_ys, lines_xs = np.where(extended_lines_mask > 0)
                for ly, lx in zip(lines_ys, lines_xs):
                    diffs = ring_coords - np.array([ly, lx])
                    dists = np.sum(diffs ** 2, axis=1)
                    result[ly, lx] = ring_colors[np.argmin(dists)]
        if np.any(mask_binary):
            r = max(3, min(int(np.sqrt(np.count_nonzero(mask_binary) / np.pi)), 20))
            inpainted = cv2.inpaint(result, mask_binary, r, cv2.INPAINT_TELEA)
            mf = mask_binary.astype(np.float32)
            m3 = np.stack([mf] * 3, axis=-1)
            result = (result * (1 - m3) + inpainted * m3).astype(np.uint8)
        return cv2.bilateralFilter(result, 5, 50, 50)

class EnhancedInpainter:
    def inpaint(self, image, mask, radius):
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=2)
        inp = cv2.inpaint(image, mask_dilated, radius, cv2.INPAINT_TELEA)
        try:
            smoothed = cv2.ximgproc.guidedFilter(guide=inp, src=inp, radius=radius * 2, eps=1e-4)
        except Exception:
            smoothed = cv2.bilateralFilter(inp, 9, 50, 50)
        mf = mask_dilated.astype(np.float32) / 255.0
        m3 = np.stack([mf] * 3, axis=-1)
        return (inp * (1 - m3) + smoothed * m3).astype(np.uint8)

class WatermarkRemover:
    def __init__(self):
        self.target_hsv = None
        self.roi = None
        self.tolerance = 30
        self.inpaint_radius = 5
        self.method = "enhanced"
        self._structure = StructureGuidedInpainter()
        self._enhanced = EnhancedInpainter()

    def set_target(self, bgr):
        arr = np.uint8([[list(bgr)]])
        self.target_hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)[0][0].astype(float)

    def set_roi(self, x, y, w, h):
        self.roi = (int(x), int(y), int(w), int(h))

    def generate_mask(self, frame):
        if self.target_hsv is None or self.roi is None:
            return None
        x, y, w, h = self.roi
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(float)
        roi_hsv = hsv[y:y + h, x:x + w].reshape(-1, 3)
        h_diff = np.abs(roi_hsv[:, 0] - self.target_hsv[0])
        h_diff = np.minimum(h_diff, 180 - h_diff)
        s_diff = np.abs(roi_hsv[:, 1] - self.target_hsv[1])
        v_diff = np.abs(roi_hsv[:, 2] - self.target_hsv[2])
        dist = np.sqrt((h_diff / 180) ** 2 + (s_diff / 255) ** 2 + (v_diff / 255) ** 2) / np.sqrt(3)
        mask_roi = (dist < self.tolerance / 100.0).astype(np.uint8) * 255
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        mask[y:y + h, x:x + w] = mask_roi.reshape(h, w)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return cv2.GaussianBlur(mask, (3, 3), 0)

    def remove(self, frame):
        try:
            mask = self.generate_mask(frame)
            if mask is None or not np.any(mask):
                return frame
            mask_bin = (mask > 0).astype(np.uint8)
            expand_pixels = max(1, self.inpaint_radius)
            ksize = expand_pixels * 2 + 1
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            mask_expanded = cv2.dilate(mask_bin, k, iterations=1)
            inv = (1 - mask_bin) * 255
            dist = cv2.distanceTransform(inv.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
            weight = np.zeros_like(dist)
            weight[mask_bin > 0] = 1.0
            ring = (mask_expanded > 0) & (mask_bin == 0)
            if np.any(ring):
                weight[ring] = np.clip(1.0 - dist[ring] / float(expand_pixels), 0, 1)
            if self.method == "structure":
                inpainted = self._structure.inpaint(frame, mask_expanded * 255)
            elif self.method == "enhanced":
                inpainted = self._enhanced.inpaint(frame, mask_expanded * 255, self.inpaint_radius)
            else:
                inpainted = self._inpaint_classic(mask_expanded * 255, frame)
            w3 = np.stack([weight] * 3, axis=-1)
            return (frame * (1 - w3) + inpainted * w3).astype(np.uint8)
        except Exception as e:
            import traceback
            try:
                log_path = os.path.join(os.environ.get("EXTERNAL_STORAGE", "/sdcard"), "zuuixin_crash.log")
                with open(log_path, "w", encoding="utf-8") as f:
                    traceback.print_exc(file=f)
            except Exception:
                pass
            raise

    def _inpaint_classic(self, mask, frame):
        x, y, w, h = self.roi
        hf, wf = frame.shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = frame.copy()
        for contour in contours:
            cx, cy, cw, ch = cv2.boundingRect(contour)
            if cw <= 0 or ch <= 0:
                continue
            region = frame[cy:cy + ch, cx:cx + cw]
            mean_color = region.mean(axis=(0, 1))
            best_match = None
            best_dist = float("inf")
            step = max(cw, ch)
            for sy in range(0, hf - ch, step):
                for sx in range(0, wf - cw, step):
                    if (x <= sx < x + w) and (y <= sy < y + h):
                        continue
                    candidate = frame[sy:sy + ch, sx:sx + cw]
                    cd = np.linalg.norm(candidate.mean(axis=(0, 1)) - mean_color)
                    if cd < best_dist:
                        best_dist = cd
                        best_match = candidate
            if best_match is None:
                continue
            src = cv2.resize(best_match, (cw, ch))
            cm = np.zeros((hf, wf), dtype=np.uint8)
            cv2.drawContours(cm, [contour], -1, 255, -1)
            m3 = (cm[cy:cy + ch, cx:cx + cw] > 0).astype(np.float32)
            m3 = np.stack([m3] * 3, axis=-1)
            roi_area = result[cy:cy + ch, cx:cx + cw]
            result[cy:cy + ch, cx:cx + cw] = (roi_area * (1 - m3) + src * m3).astype(np.uint8)
        return result

# ==================== Kivy UI ====================
class VideoWidget(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_bgr = None
        self.texture = None
        self.roi_start = None
        self.roi_end = None
        self.drawing = False
        self.display_scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self._bind_events()

    def _bind_events(self):
        self.bind(size=self._update_display, pos=self._update_display)

    def _update_display(self, *args):
        if self.frame_bgr is not None:
            self._render_frame()

    def set_frame(self, frame_bgr):
        self.frame_bgr = frame_bgr
        self._render_frame()

    def _render_frame(self):
        if self.frame_bgr is None:
            return
        h, w = self.frame_bgr.shape[:2]
        vw, vh = self.width, self.height
        if vw <= 0 or vh <= 0:
            return
        scale = min(vw / w, vh / h)
        nw, nh = int(w * scale), int(h * scale)
        self.display_scale = scale
        self.offset_x = (vw - nw) / 2
        self.offset_y = (vh - nh) / 2
        disp = cv2.resize(self.frame_bgr, (nw, nh))
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        rgb = np.flipud(rgb)
        if self.texture is None or self.texture.size != (nw, nh):
            self.texture = Texture.create(size=(nw, nh), colorfmt="rgb")
            self.texture.flip_vertical()
        self.texture.blit_buffer(rgb.tobytes(), size=(nw, nh), colorfmt="rgb")
        self.canvas.clear()
        with self.canvas:
            Color(1, 1, 1, 1)
            Rectangle(texture=self.texture, pos=(self.offset_x, self.offset_y), size=(nw, nh))
            if self.roi_start and self.roi_end:
                Color(0, 1, 0, 0.7)
                x1, y1 = self.roi_start
                x2, y2 = self.roi_end
                rx = min(x1, x2)
                ry = min(y1, y2)
                rw = abs(x2 - x1)
                rh = abs(y2 - y1)
                Line(rectangle=(rx, ry, rw, rh), width=2)

    def _to_frame_coords(self, tx, ty):
        x = (tx - self.offset_x) / self.display_scale
        y = ((self.height - ty) - self.offset_y) / self.display_scale
        # In Kivy, y increases upward
        return int(x), int(y)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.drawing = True
            fx, fy = self._to_frame_coords(touch.x, touch.y)
            self.roi_start = (fx, fy)
            self.roi_end = (fx, fy)
            self._render_frame()
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.drawing:
            fx, fy = self._to_frame_coords(touch.x, touch.y)
            self.roi_end = (fx, fy)
            self._render_frame()
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.drawing:
            self.drawing = False
            return True
        return super().on_touch_up(touch)

    def get_roi(self):
        if self.roi_start is None or self.roi_end is None:
            return None
        x1, y1 = self.roi_start
        x2, y2 = self.roi_end
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        if w < 5 or h < 5:
            return None
        return (x, y, w, h)


class MainApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.remover = WatermarkRemover()
        self.cap = None
        self.video_path = None
        self.total_frames = 0
        self.fps = 0
        self.preview_frame = None
        self.last_result = None
        self.is_playing = False
        self.is_seeking = False
        self._play_event = None
        self._process_thread = None
        self._process_cancel = Event()

    def build(self):
        Window.size = (900, 600)
        root = BoxLayout(orientation="vertical", spacing=4, padding=4)

        # Top control bar
        controls = BoxLayout(orientation="horizontal", size_hint=(1, None), height=40, spacing=4)
        self.btn_load = Button(text="加载视频", size_hint=(None, 1), width=80)
        self.btn_load.bind(on_press=self._pick_video)
        self.btn_pick = Button(text="取色", size_hint=(None, 1), width=60)
        self.btn_pick.bind(on_press=self._pick_color)
        self.btn_preview = Button(text="预览", size_hint=(None, 1), width=60)
        self.btn_preview.bind(on_press=self._preview)
        self.btn_process = Button(text="处理", size_hint=(None, 1), width=60)
        self.btn_process.bind(on_press=self._start_process)
        self.btn_method = ToggleButton(text="增强", size_hint=(None, 1), width=60, state="down")
        self.btn_method.bind(on_press=self._toggle_method)
        controls.add_widget(self.btn_load)
        controls.add_widget(self.btn_pick)
        controls.add_widget(self.btn_preview)
        controls.add_widget(self.btn_process)
        controls.add_widget(self.btn_method)

        tol_lbl = Label(text="容差", size_hint=(None, 1), width=30)
        self.slider_tol = Slider(min=1, max=100, value=self.remover.tolerance, size_hint=(0.15, 1))
        self.slider_tol.bind(value=self._on_tol)
        self.lbl_tol = Label(text=str(self.remover.tolerance), size_hint=(None, 1), width=25)
        controls.add_widget(tol_lbl)
        controls.add_widget(self.slider_tol)
        controls.add_widget(self.lbl_tol)

        rad_lbl = Label(text="半径", size_hint=(None, 1), width=30)
        self.slider_radius = Slider(min=1, max=30, value=self.remover.inpaint_radius, size_hint=(0.1, 1))
        self.slider_radius.bind(value=self._on_radius)
        self.lbl_radius = Label(text=str(self.remover.inpaint_radius), size_hint=(None, 1), width=25)
        controls.add_widget(rad_lbl)
        controls.add_widget(self.slider_radius)
        controls.add_widget(self.lbl_radius)

        root.add_widget(controls)

        # Video display
        self.video = VideoWidget()
        root.add_widget(self.video)

        # Bottom seek bar
        seek = BoxLayout(orientation="horizontal", size_hint=(1, None), height=36, spacing=4)
        self.btn_play = Button(text="▶", size_hint=(None, 1), width=36)
        self.btn_play.bind(on_press=self._toggle_play)
        self.btn_prev = Button(text="◀", size_hint=(None, 1), width=36)
        self.btn_prev.bind(on_press=self._prev_frame)
        self.seek_slider = Slider(min=0, max=0, value=0, size_hint=(1, 1))
        self.seek_slider.bind(value=self._on_seek_drag)
        self.lbl_time = Label(text="00:00 / 00:00", size_hint=(None, 1), width=120)
        seek.add_widget(self.btn_play)
        seek.add_widget(self.btn_prev)
        seek.add_widget(self.seek_slider)
        seek.add_widget(self.lbl_time)
        root.add_widget(seek)

        # Status label
        self.lbl_status = Label(text="就绪", size_hint=(1, None), height=20, halign="left", valign="middle")
        self.lbl_status.bind(size=self.lbl_status.setter("text_size"))
        root.add_widget(self.lbl_status)

        return root

    # --- callbacks ---
    def _on_tol(self, inst, val):
        v = int(val)
        self.remover.tolerance = v
        self.lbl_tol.text = str(v)

    def _on_radius(self, inst, val):
        v = int(val)
        self.remover.inpaint_radius = v
        self.lbl_radius.text = str(v)

    def _toggle_method(self, btn):
        if btn.state == "down":
            self.remover.method = "enhanced"
            btn.text = "增强"
        else:
            self.remover.method = "structure"
            btn.text = "结构"

    def _pick_video(self, btn):
        if platform == "android":
            from androidstorage import AndroidStorage
            self._android_pick_video()
        else:
            self._show_file_chooser()

    def _show_file_chooser(self):
        content = BoxLayout(orientation="vertical")
        fc = FileChooserListView(path=os.path.expanduser("~"))
        btn_ok = Button(text="选择", size_hint=(1, None), height=40)
        popup = Popup(title="选择视频文件", content=content, size_hint=(0.9, 0.9))
        content.add_widget(fc)
        content.add_widget(btn_ok)
        def on_select(inst):
            if fc.selection:
                self._open_video(fc.selection[0])
            popup.dismiss()
        btn_ok.bind(on_press=on_select)
        popup.open()

    def _android_pick_video(self):
        try:
            from android import activity, mActivity
            from jnius import autoclass
            Intent = autoclass("android.content.Intent")
            Uri = autoclass("android.net.Uri")
            intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
            intent.setType("video/*")
            intent.addCategory(Intent.CATEGORY_OPENABLE)
            mActivity.startActivityForResult(intent, 1001)
            activity.bind(on_activity_result=self._on_android_result)
        except Exception as e:
            self.lbl_status.text = f"无法打开文件选择器: {e}"

    def _on_android_result(self, requestCode, resultCode, intent):
        if requestCode == 1001 and resultCode == -1:
            from jnius import autoclass
            Uri = autoclass("android.net.Uri")
            uri = intent.getData()
            ContentResolver = autoclass("android.content.ContentResolver")
            cr = App.get_running_app().mActivity.getContentResolver()
            input = cr.openInputStream(uri)
            from androidstorage import AndroidStorage
            dst = os.path.join(os.environ.get("EXTERNAL_STORAGE", "/sdcard"), "temp_input_video.mp4")
            with open(dst, "wb") as f:
                f.write(input.read())
            input.close()
            self._open_video(dst)

    def _open_video(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.lbl_status.text = "无法打开视频文件"
            return
        self.video_path = path
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        ret, frame = self.cap.read()
        if ret:
            self.preview_frame = frame
            self.last_result = None
            self.video.set_frame(frame)
        self.seek_slider.max = max(0, self.total_frames - 1)
        self.seek_slider.value = 0
        self._update_time()
        self.lbl_status.text = f"已加载: {os.path.basename(path)}  ({self.total_frames}帧)"

    def _pick_color(self, btn):
        if self.preview_frame is None:
            self.lbl_status.text = "请先加载视频帧"
            return
        roi = self.video.get_roi()
        if roi is None:
            self.lbl_status.text = "请在画面上框选水印区域"
            return
        x, y, w, h = roi
        self.remover.set_roi(x, y, w, h)
        center = self.preview_frame[y + h//2, x + w//2]
        self.remover.set_target(tuple(center))
        self.lbl_status.text = f"已取色: HSV={self.remover.target_hsv}"

    def _preview(self, btn):
        if self.preview_frame is None:
            self.lbl_status.text = "请先加载视频"
            return
        if self.remover.target_hsv is None or self.remover.roi is None:
            self.lbl_status.text = "请先框选并取色"
            return
        result = self.remover.remove(self.preview_frame)
        self.last_result = result
        self.video.set_frame(result)
        self.lbl_status.text = "预览完成"

    def _start_process(self, btn):
        if self.cap is None or not self.cap.isOpened():
            self.lbl_status.text = "请先加载视频"
            return
        if self.remover.target_hsv is None:
            self.lbl_status.text = "请先取色"
            return
        if self.is_playing:
            self._stop_play()
        self._process_cancel.clear()
        self.btn_process.text = "停止"
        self.btn_process.unbind(on_press=self._start_process)
        self.btn_process.bind(on_press=self._stop_process)
        self.lbl_status.text = "处理中..."
        self._process_thread = Thread(target=self._process_worker, daemon=True)
        self._process_thread.start()

    def _stop_process(self, btn):
        self._process_cancel.set()
        self.btn_process.text = "处理"
        self.btn_process.unbind(on_press=self._stop_process)
        self.btn_process.bind(on_press=self._start_process)

    def _process_worker(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                Clock.schedule_once(lambda dt: setattr(self.lbl_status, "text", "无法重新打开视频"))
                return
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_dir = os.path.dirname(self.video_path) or "."
            out_name = os.path.splitext(os.path.basename(self.video_path))[0] + "_无反色.mp4"
            out_path = os.path.join(out_dir, out_name)
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            if not writer.isOpened():
                Clock.schedule_once(lambda dt: setattr(self.lbl_status, "text", "无法创建输出视频"))
                cap.release()
                return
            idx = 0
            while not self._process_cancel.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                repaired = self.remover.remove(frame)
                writer.write(repaired)
                idx += 1
                if idx % 10 == 0:
                    prog = f"处理中: {idx}/{total}"
                    Clock.schedule_once(lambda dt, p=prog: setattr(self.lbl_status, "text", p))
            cap.release()
            writer.release()
            if self._process_cancel.is_set():
                try:
                    os.remove(out_path)
                except Exception:
                    pass
                Clock.schedule_once(lambda dt: setattr(self.lbl_status, "text", "已取消处理"))
            else:
                Clock.schedule_once(lambda dt: setattr(self.lbl_status, "text", f"完成 → {out_name}"))
        except Exception as e:
            import traceback
            try:
                log_path = os.path.join(os.environ.get("EXTERNAL_STORAGE", "/sdcard"), "zuuixin_crash.log")
                with open(log_path, "w", encoding="utf-8") as f:
                    traceback.print_exc(file=f)
            except Exception:
                pass
            Clock.schedule_once(lambda dt, err=str(e): setattr(self.lbl_status, "text", f"错误: {err}"))
        finally:
            Clock.schedule_once(lambda dt: self._reset_process_btn())

    def _reset_process_btn(self):
        self.btn_process.text = "处理"
        self.btn_process.unbind(on_press=self._stop_process)
        self.btn_process.bind(on_press=self._start_process)

    # --- playback ---
    def _toggle_play(self, btn):
        if self.cap is None:
            return
        if self.is_playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self):
        self.is_playing = True
        self.btn_play.text = "⏸"
        self._play_event = Clock.schedule_interval(self._play_tick, 1.0 / self.fps)

    def _stop_play(self):
        self.is_playing = False
        self.btn_play.text = "▶"
        if self._play_event:
            self._play_event.cancel()
            self._play_event = None

    def _play_tick(self, dt):
        if self.cap and self.cap.isOpened():
            if not self.is_seeking:
                ret, frame = self.cap.read()
                if ret:
                    self.preview_frame = frame
                    self.last_result = None
                    self.video.set_frame(frame)
                    pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.seek_slider.value = min(pos, self.total_frames - 1)
                    self._update_time()
                else:
                    self._stop_play()
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _prev_frame(self, btn):
        if self.cap and self.cap.isOpened():
            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            target = max(0, pos - 2)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = self.cap.read()
            if ret:
                self.preview_frame = frame
                self.last_result = None
                self.video.set_frame(frame)
                self.seek_slider.value = target
                self._update_time()

    def _on_seek_drag(self, inst, val):
        if self.cap and self.cap.isOpened():
            target = int(val)
            self.is_seeking = True
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = self.cap.read()
            if ret:
                self.preview_frame = frame
                self.last_result = None
                self.video.set_frame(frame)
            self._update_time()
            self.is_seeking = False

    def _update_time(self):
        if self.cap and self.fps > 0:
            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            cur = f"{pos // 60 // 60:02d}:{pos // 60 % 60:02d}:{pos % 60:02d}" if self.total_frames > 3600 else f"{pos // 60:02d}:{pos % 60:02d}"
            total = f"{self.total_frames // 60 // 60:02d}:{self.total_frames // 60 % 60:02d}:{self.total_frames % 60:02d}" if self.total_frames > 3600 else f"{self.total_frames // 60:02d}:{self.total_frames % 60:02d}"
            self.lbl_time.text = f"{cur} / {total}"


if __name__ == "__main__":
    MainApp().run()
