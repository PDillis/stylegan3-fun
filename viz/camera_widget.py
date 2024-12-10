import imgui
from gui_utils import imgui_utils
import cv2

#----------------------------------------------------------------------------


class CameraWidget:
    def __init__(self, viz):
        self.viz = viz
        self.camera_enabled = False
        self.cap = None
        self.camera_texture = None
        self.frame_shape = None

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        if show:
            imgui.text('Camera')
            imgui.same_line(self.viz.label_w)
            _, self.camera_enabled = imgui.checkbox('Enable##camera', self.camera_enabled)

            if self.camera_enabled and self.cap is None:
                self.cap = cv2.VideoCapture(0)
            elif not self.camera_enabled and self.cap is not None:
                self.cap.release()
                self.cap = None
                self.camera_texture = None
                self.frame_shape = None

    def get_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))  # Adjust size as needed
                self.frame_shape = frame.shape
                return frame
        return None

    def update_texture(self, gl_utils):
        frame = self.get_frame()
        if frame is not None:
            # Crop the frame to maintain 2:1 aspect ratio
            height = frame.shape[0]
            crop_height = frame.shape[1] // 2
            frame = frame[:crop_height, :, :]

            if self.camera_texture is None or self.frame_shape != frame.shape:
                self.camera_texture = gl_utils.Texture(image=frame, bilinear=False, mipmap=False)
            else:
                self.camera_texture.update(frame)

    def draw(self, pos, zoom):
        if self.camera_texture is not None:
            self.camera_texture.draw(pos=pos, zoom=zoom, align=0.5, rint=True)