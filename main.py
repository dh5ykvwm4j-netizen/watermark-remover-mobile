import os
import sys
import math
import time
from collections import deque
from threading import Thread, Event

from PIL import Image, ImageDraw, ImageFilter, ImageChops
import colorsys

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, Line
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.utils import platform

ANDROID = (platform == "android")

if ANDROID:
    from jnius import autoclass
    from android import activity as android_activity
    mActivity = autoclass("org.kivy.android.PythonActivity").mActivity
    Context = autoclass("android.content.Context")
    ContentResolver = autoclass("android.content.ContentResolver")
    Uri = autoclass("android.net.Uri")
    Intent = autoclass("android.content.Intent")
    MediaMetadataRetriever = autoclass("android.media.MediaMetadataRetriever")
    MediaCodec = autoclass("android.media.MediaCodec")
    MediaFormat = autoclass("android.media.MediaFormat")
    MediaMuxer = autoclass("android.media.MediaMuxer")
    Bitmap = autoclass("android.graphics.Bitmap")
    BitmapFactory = autoclass("android.graphics.BitmapFactory")
    ColorJava = autoclass("android.graphics.Color")
    Canvas = autoclass("android.graphics.Canvas")
    ByteArrayOutputStream = autoclass("java.io.ByteArrayOutputStream")
    BitmapFactory_Options = autoclass("android.graphics.BitmapFactory$Options")


def pil_to_texture(pil_img):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    tex = Texture.create(size=(w, h), colorfmt="rgb")
    tex.flip_vertical()
    data = pil_img.tobytes()
    tex.blit_buffer(data, size=(w, h), colorfmt="rgb")
    return tex


def android_bitmap_to_pil(bitmap):
    w = bitmap.getWidth()
    h = bitmap.getHeight()
    pixels = [0] * (w * h)
    bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            c = pixels[y * w + x]
            r = (c >> 16) & 0xFF
            g = (c >> 8) & 0xFF
            b = c & 0xFF
            px[x, y] = (r, g, b)
    return img


def extract_frame_android(video_path, time_us):
    mmr = MediaMetadataRetriever()
    mmr.setDataSource(video_path)
    bitmap = mmr.getFrameAtTime(time_us, 0)
    mmr.release()
    if bitmap is None:
        return None
    return android_bitmap_to_pil(bitmap)


class WatermarkRemover:
    def __init__(self):
        self.target_rgb = None
        self.roi = None
        self.tolerance = 30
        self.inpaint_radius = 5
        self.method = "enhanced"

    def set_target(self, rgb):
        self.target_rgb = rgb

    def set_roi(self, x, y, w, h):
        self.roi = (int(x), int(y), int(w), int(h))

    def _color_distance(self, c1, c2):
        r1, g1, b1 = c1
        r2, g2, b2 = c2
        return math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) / 441.67

    def generate_mask(self, frame_pil):
        if self.target_rgb is None or self.roi is None:
            return None
        x, y, w, h = self.roi
        mask = Image.new("L", frame_pil.size, 0)
        frame_px = frame_pil.load()
        mask_px = mask.load()
        tr, tg, tb = self.target_rgb
        tol = self.tolerance / 100.0
        for py in range(max(0, y), min(frame_pil.height, y + h)):
            for px in range(max(0, x), min(frame_pil.width, x + w)):
                r, g, b = frame_px[px, py]
                dist = math.sqrt(((r - tr) / 255.0) ** 2 + ((g - tg) / 255.0) ** 2 + ((b - tb) / 255.0) ** 2) / math.sqrt(3)
                if dist < tol:
                    mask_px[px, py] = 255
        mask = mask.filter(ImageFilter.MinFilter(3))
        mask = mask.filter(ImageFilter.MaxFilter(5))
        mask = mask.filter(ImageFilter.GaussianBlur(1))
        return mask

    def _simple_inpaint(self, frame_pil, mask_pil, radius):
        result = frame_pil.copy()
        result_px = result.load()
        mask_px = mask_pil.load()
        w, h = frame_pil.size
        for y in range(h):
            for x in range(w):
                if mask_px[x, y] > 0:
                    neighbors = []
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < w and 0 <= ny < h and mask_px[nx, ny] == 0:
                                neighbors.append(result_px[nx, ny])
                    if neighbors:
                        r = sum(c[0] for c in neighbors) // len(neighbors)
                        g = sum(c[1] for c in neighbors) // len(neighbors)
                        b = sum(c[2] for c in neighbors) // len(neighbors)
                        result_px[x, y] = (r, g, b)
        return result

    def _structure_inpaint(self, frame_pil, mask_pil):
        w, h = frame_pil.size
        mask_px = mask_pil.load()
        frame_px = frame_pil.load()
        result = frame_pil.copy()
        result_px = result.load()
        outer = mask_pil.filter(ImageFilter.MaxFilter(7))
        outer_px = outer.load()
        ring_pixels = []
        for y in range(h):
            for x in range(w):
                if outer_px[x, y] > 0 and mask_px[x, y] == 0:
                    ring_pixels.append((x, y, frame_px[x, y]))
        for y in range(h):
            for x in range(w):
                if mask_px[x, y] > 0 and ring_pixels:
                    min_dist = float("inf")
                    best_color = (128, 128, 128)
                    src_r, src_g, src_b = frame_px[x, y]
                    for rx, ry, (rr, rg, rb) in ring_pixels:
                        d = (rx - x) ** 2 + (ry - y) ** 2 + ((rr - src_r) ** 2 + (rg - src_g) ** 2 + (rb - src_b) ** 2) * 0.01
                        if d < min_dist:
                            min_dist = d
                            best_color = (rr, rg, rb)
                    result_px[x, y] = best_color
        return result

    def _enhanced_inpaint(self, frame_pil, mask_pil, radius):
        blurred = frame_pil.filter(ImageFilter.GaussianBlur(radius * 2))
        result = frame_pil.copy()
        result_px = result.load()
        blurred_px = blurred.load()
        mask_px = mask_pil.load()
        w, h = frame_pil.size
        expanded = mask_pil.filter(ImageFilter.MaxFilter(max(3, radius * 2 + 1)))
        expanded_px = expanded.load()
        for y in range(h):
            for x in range(w):
                if expanded_px[x, y] > 0:
                    m_val = mask_px[x, y]
                    if m_val > 0:
                        alpha = 1.0
                    else:
                        edge_dist = 0
                        for d in range(1, radius + 1):
                            found = False
                            for dx in range(-d, d + 1):
                                for dy in range(-d, d + 1):
                                    if abs(dx) == d or abs(dy) == d:
                                        nx, ny = x + dx, y + dy
                                        if 0 <= nx < w and 0 <= ny < h and mask_px[nx, ny] > 0:
                                            found = True
                                            break
                                if found:
                                    break
                            if found:
                                edge_dist = d
                                break
                        alpha = max(0, 1.0 - edge_dist / max(1, radius))
                    orig = result_px[x, y]
                    blur = blurred_px[x, y]
                    r = int(orig[0] * (1 - alpha) + blur[0] * alpha)
                    g = int(orig[1] * (1 - alpha) + blur[1] * alpha)
                    b = int(orig[2] * (1 - alpha) + blur[2] * alpha)
                    result_px[x, y] = (min(255, r), min(255, g), min(255, b))
        return result

    def remove(self, frame_pil):
        try:
            mask = self.generate_mask(frame_pil)
            if mask is None:
                return frame_pil
            if not any(mask.load()[x, y] > 0 for y in range(mask.height) for x in range(mask.width)):
                return frame_pil
            expanded = mask.filter(ImageFilter.MaxFilter(max(3, self.inpaint_radius * 2 + 1)))
            if self.method == "structure":
                return self._structure_inpaint(frame_pil, expanded)
            elif self.method == "enhanced":
                return self._enhanced_inpaint(frame_pil, expanded, self.inpaint_radius)
            else:
                return self._simple_inpaint(frame_pil, expanded, self.inpaint_radius)
        except Exception as e:
            import traceback
            try:
                log_path = os.path.join(os.environ.get("EXTERNAL_STORAGE", "/sdcard"), "zuuixin_crash.log")
                with open(log_path, "w", encoding="utf-8") as f:
                    traceback.print_exc(file=f)
            except Exception:
                pass
            raise


class VideoWidget(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_pil = None
        self.texture = None
        self.roi_start = None
        self.roi_end = None
        self.drawing = False
        self.display_scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.bind(size=self._update_display, pos=self._update_display)

    def _update_display(self, *args):
        if self.frame_pil is not None:
            self._render_frame()

    def set_frame(self, frame_pil):
        self.frame_pil = frame_pil
        self._render_frame()

    def _render_frame(self):
        if self.frame_pil is None:
            return
        w, h = self.frame_pil.size
        vw, vh = self.width, self.height
        if vw <= 0 or vh <= 0:
            return
        scale = min(vw / w, vh / h)
        nw, nh = int(w * scale), int(h * scale)
        self.display_scale = scale
        self.offset_x = (vw - nw) / 2
        self.offset_y = (vh - nh) / 2
        disp = self.frame_pil.resize((nw, nh), Image.LANCZOS)
        self.texture = pil_to_texture(disp)
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
        self.video_path = None
        self.total_frames = 0
        self.fps = 30
        self.current_frame_idx = 0
        self.preview_frame = None
        self.last_result = None
        self.is_playing = False
        self.is_seeking = False
        self._play_event = None
        self._process_thread = None
        self._process_cancel = Event()
        self.video_duration_us = 0
        self.video_width = 0
        self.video_height = 0

    def build(self):
        Window.size = (900, 600)
        root = BoxLayout(orientation="vertical", spacing=4, padding=4)

        controls = BoxLayout(orientation="horizontal", size_hint=(1, None), height=40, spacing=4)
        self.btn_load = Button(text="Load Video", size_hint=(None, 1), width=80)
        self.btn_load.bind(on_press=self._pick_video)
        self.btn_pick = Button(text="Pick Color", size_hint=(None, 1), width=80)
        self.btn_pick.bind(on_press=self._pick_color)
        self.btn_preview = Button(text="Preview", size_hint=(None, 1), width=70)
        self.btn_preview.bind(on_press=self._preview)
        self.btn_process = Button(text="Process", size_hint=(None, 1), width=70)
        self.btn_process.bind(on_press=self._start_process)
        self.btn_method = ToggleButton(text="Enhanced", size_hint=(None, 1), width=80, state="down")
        self.btn_method.bind(on_press=self._toggle_method)
        controls.add_widget(self.btn_load)
        controls.add_widget(self.btn_pick)
        controls.add_widget(self.btn_preview)
        controls.add_widget(self.btn_process)
        controls.add_widget(self.btn_method)

        tol_lbl = Label(text="Tol", size_hint=(None, 1), width=30)
        self.slider_tol = Slider(min=1, max=100, value=self.remover.tolerance, size_hint=(0.15, 1))
        self.slider_tol.bind(value=self._on_tol)
        self.lbl_tol = Label(text=str(self.remover.tolerance), size_hint=(None, 1), width=30)
        controls.add_widget(tol_lbl)
        controls.add_widget(self.slider_tol)
        controls.add_widget(self.lbl_tol)

        rad_lbl = Label(text="Rad", size_hint=(None, 1), width=30)
        self.slider_radius = Slider(min=1, max=30, value=self.remover.inpaint_radius, size_hint=(0.1, 1))
        self.slider_radius.bind(value=self._on_radius)
        self.lbl_radius = Label(text=str(self.remover.inpaint_radius), size_hint=(None, 1), width=30)
        controls.add_widget(rad_lbl)
        controls.add_widget(self.slider_radius)
        controls.add_widget(self.lbl_radius)

        root.add_widget(controls)
        self.video = VideoWidget()
        root.add_widget(self.video)

        seek = BoxLayout(orientation="horizontal", size_hint=(1, None), height=36, spacing=4)
        self.btn_play = Button(text="Play", size_hint=(None, 1), width=60)
        self.btn_play.bind(on_press=self._toggle_play)
        self.btn_prev = Button(text="Prev", size_hint=(None, 1), width=60)
        self.btn_prev.bind(on_press=self._prev_frame)
        self.seek_slider = Slider(min=0, max=100, value=0, size_hint=(1, 1))
        self.seek_slider.bind(value=self._on_seek_drag)
        self.lbl_time = Label(text="00:00 / 00:00", size_hint=(None, 1), width=120)
        seek.add_widget(self.btn_play)
        seek.add_widget(self.btn_prev)
        seek.add_widget(self.seek_slider)
        seek.add_widget(self.lbl_time)
        root.add_widget(seek)

        self.lbl_status = Label(text="Ready", size_hint=(1, None), height=20, halign="left", valign="middle")
        self.lbl_status.bind(size=self.lbl_status.setter("text_size"))
        root.add_widget(self.lbl_status)

        return root

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
            btn.text = "Enhanced"
        else:
            self.remover.method = "structure"
            btn.text = "Structure"

    def _pick_video(self, btn):
        if ANDROID:
            self._android_pick_video()
        else:
            self._show_file_chooser()

    def _show_file_chooser(self):
        content = BoxLayout(orientation="vertical")
        fc = FileChooserListView(path=os.path.expanduser("~"))
        btn_ok = Button(text="Select", size_hint=(1, None), height=40)
        popup = Popup(title="Select video file", content=content, size_hint=(0.9, 0.9))
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
            intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
            intent.setType("video/*")
            intent.addCategory(Intent.CATEGORY_OPENABLE)
            mActivity.startActivityForResult(intent, 1001)
            android_activity.bind(on_activity_result=self._on_android_result)
        except Exception as e:
            self.lbl_status.text = f"Error: {e}"

    def _on_android_result(self, requestCode, resultCode, intent):
        if requestCode == 1001 and resultCode == -1:
            uri = intent.getData()
            cr = mActivity.getContentResolver()
            inp = cr.openInputStream(uri)
            dst = os.path.join(os.environ.get("EXTERNAL_STORAGE", "/sdcard"), "temp_input.mp4")
            with open(dst, "wb") as f:
                while True:
                    data = inp.read(65536)
                    if not data:
                        break
                    f.write(data)
            inp.close()
            self._open_video(dst)

    def _open_video(self, path):
        self.video_path = path
        if ANDROID:
            self._open_video_android(path)
        else:
            self._open_video_pil(path)

    def _open_video_android(self, path):
        try:
            mmr = MediaMetadataRetriever()
            mmr.setDataSource(path)
            dur_str = mmr.extractMetadata(9)
            self.video_duration_us = int(dur_str) if dur_str else 0
            w_str = mmr.extractMetadata(18)
            h_str = mmr.extractMetadata(19)
            self.video_width = int(w_str) if w_str else 640
            self.video_height = int(h_str) if h_str else 480
            self.fps = 30
            if self.video_duration_us > 0:
                self.total_frames = max(1, int(self.video_duration_us * self.fps / 1000000))
            else:
                self.total_frames = 1
            self.current_frame_idx = 0
            frame = self._get_frame_android(0)
            mmr.release()
            if frame:
                self.preview_frame = frame
                self.last_result = None
                self.video.set_frame(frame)
            self.seek_slider.max = max(0, self.total_frames - 1)
            self.seek_slider.value = 0
            self._update_time()
            self.lbl_status.text = f"Loaded: {os.path.basename(path)} ({self.total_frames} frames)"
        except Exception as e:
            self.lbl_status.text = f"Error loading: {e}"

    def _open_video_pil(self, path):
        try:
            from PIL import ImageSequence
            img = Image.open(path)
            self.total_frames = getattr(img, "n_frames", 1)
            self.fps = getattr(img, "info", {}).get("fps", 10)
            self.current_frame_idx = 0
            self.video_width, self.video_height = img.size
            self.preview_frame = img.copy().convert("RGB")
            self.last_result = None
            self.video.set_frame(self.preview_frame)
            self.seek_slider.max = max(0, self.total_frames - 1)
            self.seek_slider.value = 0
            self._update_time()
            self.lbl_status.text = f"Loaded: {os.path.basename(path)} ({self.total_frames} frames)"
        except Exception as e:
            self.lbl_status.text = f"Error: {e}"

    def _get_frame_android(self, frame_idx):
        if not self.video_path:
            return None
        time_us = int(frame_idx * 1000000 / self.fps)
        return extract_frame_android(self.video_path, time_us)

    def _get_frame_pil(self, frame_idx):
        try:
            from PIL import ImageSequence
            img = Image.open(self.video_path)
            img.seek(min(frame_idx, self.total_frames - 1))
            return img.copy().convert("RGB")
        except Exception:
            return None

    def _get_frame(self, frame_idx):
        if ANDROID:
            return self._get_frame_android(frame_idx)
        else:
            return self._get_frame_pil(frame_idx)

    def _pick_color(self, btn):
        if self.preview_frame is None:
            self.lbl_status.text = "Load a video first"
            return
        roi = self.video.get_roi()
        if roi is None:
            self.lbl_status.text = "Draw a ROI on the video first"
            return
        x, y, w, h = roi
        self.remover.set_roi(x, y, w, h)
        cx, cy = x + w // 2, y + h // 2
        px = self.preview_frame.load()
        r, g, b = px[min(cx, self.preview_frame.width - 1), min(cy, self.preview_frame.height - 1)]
        self.remover.set_target((r, g, b))
        self.lbl_status.text = f"Color picked: RGB=({r},{g},{b})"

    def _preview(self, btn):
        if self.preview_frame is None:
            self.lbl_status.text = "Load a video first"
            return
        if self.remover.target_rgb is None or self.remover.roi is None:
            self.lbl_status.text = "Draw ROI and pick color first"
            return
        result = self.remover.remove(self.preview_frame)
        self.last_result = result
        self.video.set_frame(result)
        self.lbl_status.text = "Preview done"

    def _start_process(self, btn):
        if self.video_path is None:
            self.lbl_status.text = "Load a video first"
            return
        if self.remover.target_rgb is None:
            self.lbl_status.text = "Pick color first"
            return
        if self.is_playing:
            self._stop_play()
        self._process_cancel.clear()
        self.btn_process.text = "Stop"
        self.btn_process.unbind(on_press=self._start_process)
        self.btn_process.bind(on_press=self._stop_process)
        self.lbl_status.text = "Processing..."
        self._process_thread = Thread(target=self._process_worker, daemon=True)
        self._process_thread.start()

    def _stop_process(self, btn):
        self._process_cancel.set()
        self.btn_process.text = "Process"
        self.btn_process.unbind(on_press=self._stop_process)
        self.btn_process.bind(on_press=self._start_process)

    def _process_worker(self):
        try:
            if ANDROID:
                self._process_worker_android()
            else:
                self._process_worker_pil()
        except Exception as e:
            import traceback
            Clock.schedule_once(lambda dt, err=str(e): setattr(self.lbl_status, "text", f"Error: {err}"))
            try:
                log_path = os.path.join(os.environ.get("EXTERNAL_STORAGE", "/sdcard"), "zuuixin_crash.log")
                with open(log_path, "w", encoding="utf-8") as f:
                    traceback.print_exc(file=f)
            except Exception:
                pass
        finally:
            Clock.schedule_once(lambda dt: self._reset_process_btn())

    def _process_worker_pil(self):
        from PIL import ImageSequence
        img = Image.open(self.video_path)
        fps = getattr(img, "info", {}).get("fps", 10)
        total = getattr(img, "n_frames", 1)
        w, h = img.size
        out_path = os.path.splitext(self.video_path)[0] + "_clean.gif"
        frames_out = []
        idx = 0
        for frame in ImageSequence.Iterator(img):
            if self._process_cancel.is_set():
                break
            rgb = frame.convert("RGB")
            result = self.remover.remove(rgb)
            frames_out.append(result)
            idx += 1
            if idx % 5 == 0:
                prog = f"Processing: {idx}/{total}"
                Clock.schedule_once(lambda dt, p=prog: setattr(self.lbl_status, "text", p))
        if not self._process_cancel.is_set() and frames_out:
            frames_out[0].save(out_path, save_all=True, append_images=frames_out[1:], fps=int(fps), loop=0)
            Clock.schedule_once(lambda dt: setattr(self.lbl_status, "text", f"Done -> {os.path.basename(out_path)}"))
        elif self._process_cancel.is_set():
            Clock.schedule_once(lambda dt: setattr(self.lbl_status, "text", "Cancelled"))

    def _process_worker_android(self):
        try:
            mmr = MediaMetadataRetriever()
            mmr.setDataSource(self.video_path)
            out_path = os.path.join(os.environ.get("EXTERNAL_STORAGE", "/sdcard"), "output_clean.mp4")
            mime = mmr.extractMetadata(0) or "video/mp4"
            fmt = MediaFormat.createVideoFormat(mime, self.video_width, self.video_height)
            fmt.setInteger(2, 19)
            fmt.setInteger(1, self.video_width)
            fmt.setInteger(4, self.video_height)
            codec = MediaCodec.createEncoderByType(mime)
            codec.configure(fmt, None, None, 1)
            codec.start()
            muxer = MediaMuxer(out_path, 0)
            muxer_started = False
            input_done = False
            output_done = False
            total = max(1, int(self.video_duration_us / 33333))
            idx = 0
            while not (input_done and output_done):
                if self._process_cancel.is_set():
                    break
                if not input_done:
                    in_idx = codec.dequeueInputBuffer(10000)
                    if in_idx >= 0:
                        time_us = int(idx * 33333)
                        if time_us >= self.video_duration_us:
                            codec.queueInputBuffer(in_idx, 0, 0, 0, 4)
                            input_done = True
                        else:
                            bitmap = mmr.getFrameAtTime(time_us, 0)
                            if bitmap:
                                pil_frame = android_bitmap_to_pil(bitmap)
                                result = self.remover.remove(pil_frame)
                                result_rgb = result.convert("RGB")
                                w, h = result_rgb.size
                                pixels = list(result_rgb.getdata())
                                buf = bytearray(w * h * 3)
                                i = 0
                                for r, g, b in pixels:
                                    buf[i] = r
                                    buf[i + 1] = g
                                    buf[i + 2] = b
                                    i += 3
                                codec.queueInputBuffer(in_idx, 0, len(buf), time_us, 0)
                                bitmap.recycle()
                            else:
                                codec.queueInputBuffer(in_idx, 0, 0, time_us, 0)
                        idx += 1
                        if idx % 10 == 0:
                            prog = f"Processing: {idx}/{total}"
                            Clock.schedule_once(lambda dt, p=prog: setattr(self.lbl_status, "text", p))
                buf_info = MediaCodec.BufferInfo()
                out_idx = codec.dequeueOutputBuffer(buf_info, 10000)
                if out_idx >= 0:
                    out_buf = codec.getOutputBuffer(out_idx)
                    if buf_info.size > 0 and (buf_info.flags & 2) == 0:
                        if not muxer_started:
                            muxer.addTrack(fmt)
                            muxer.start()
                            muxer_started = True
                        muxer.writeSampleData(0, out_buf, buf_info)
                    codec.releaseOutputBuffer(out_idx, False)
                    if buf_info.flags & 4:
                        output_done = True
            codec.stop()
            codec.release()
            if not self._process_cancel.is_set():
                if not muxer_started:
                    muxer.addTrack(fmt)
                    muxer.start()
                muxer.stop()
                muxer.release()
                Clock.schedule_once(lambda dt: setattr(self.lbl_status, "text", f"Done -> {os.path.basename(out_path)}"))
            else:
                Clock.schedule_once(lambda dt: setattr(self.lbl_status, "text", "Cancelled"))
            mmr.release()
        except Exception as e:
            Clock.schedule_once(lambda dt, err=str(e): setattr(self.lbl_status, "text", f"Error: {err}"))

    def _reset_process_btn(self):
        self.btn_process.text = "Process"
        self.btn_process.unbind(on_press=self._stop_process)
        self.btn_process.bind(on_press=self._start_process)

    def _toggle_play(self, btn):
        if self.video_path is None:
            return
        if self.is_playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self):
        self.is_playing = True
        self.btn_play.text = "Pause"
        self._play_event = Clock.schedule_interval(self._play_tick, 1.0 / self.fps)

    def _stop_play(self):
        self.is_playing = False
        self.btn_play.text = "Play"
        if self._play_event:
            self._play_event.cancel()
            self._play_event = None

    def _play_tick(self, dt):
        if self.video_path is None:
            return
        if not self.is_seeking:
            self.current_frame_idx += 1
            if self.current_frame_idx >= self.total_frames:
                self.current_frame_idx = 0
            frame = self._get_frame(self.current_frame_idx)
            if frame:
                self.preview_frame = frame
                self.last_result = None
                self.video.set_frame(frame)
                self.seek_slider.value = self.current_frame_idx
                self._update_time()

    def _prev_frame(self, btn):
        if self.video_path is None:
            return
        self.current_frame_idx = max(0, self.current_frame_idx - 1)
        frame = self._get_frame(self.current_frame_idx)
        if frame:
            self.preview_frame = frame
            self.last_result = None
            self.video.set_frame(frame)
            self.seek_slider.value = self.current_frame_idx
            self._update_time()

    def _on_seek_drag(self, inst, val):
        if self.video_path is None:
            return
        target = int(val)
        self.is_seeking = True
        self.current_frame_idx = target
        frame = self._get_frame(target)
        if frame:
            self.preview_frame = frame
            self.last_result = None
            self.video.set_frame(frame)
        self._update_time()
        self.is_seeking = False

    def _update_time(self):
        if self.video_path and self.fps > 0:
            pos = self.current_frame_idx
            cur_s = pos / self.fps
            tot_s = self.total_frames / self.fps
            cur = f"{int(cur_s)//60:02d}:{int(cur_s)%60:02d}"
            tot = f"{int(tot_s)//60:02d}:{int(tot_s)%60:02d}"
            self.lbl_time.text = f"{cur} / {tot}"


if __name__ == "__main__":
    MainApp().run()
