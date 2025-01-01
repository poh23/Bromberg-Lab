import numpy as np
import time
# from BrombergLab.ExperimentTools.utils.image_result import VimbaImage

try:
    # from vimba import Vimba
    import vmbpy
except ImportError:
    print("can't use vimba camera")

import matplotlib.pyplot as plt


class VimbaCamera(object):
    PIXEL_SIZE = 4.8e-6

    def __init__(self, camera_num, exposure_time=None):
        """
            You can check camera_num using vimb.get_all_cameras()[0].get_model()
            exposure_time in us (micro seconds)
        """
        self.camera_num = camera_num

        # self._vimb = Vimba.get_instance()
        self._vimb = vmbpy.VmbSystem.get_instance()
        self._vimb.__enter__()
        self._cam = self._vimb.get_all_cameras()[camera_num]
        self._cam.__enter__()

        if exposure_time:
            self.set_exposure_time(exposure_time)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_image(self) -> np.ndarray:
        # takes ~42ms
        frame = self._cam.get_frame()
        # Make sure pixel format if mono8 (can be set through the VimbaViewer->all->ImageFormatControl.
        # Probably can also from python, but see no reason to change it from here :P
        im = frame.as_numpy_ndarray()[:, :, 0]
        return im.astype(float)

    def get_averaged_image(self, amount=10):
        """ return average on $amount$ pictures.
        This is like a longer exposure time, but without getting to saturation """
        im = self.get_image().astype(float)
        for i in range(1, amount):
            im += self.get_image().astype(float)

        im = im / amount

        return im

    def show_image(self, im=None, title=None):
        if im is None:
            im = self.get_image()
        fig, axes = plt.subplots()
        im = axes.imshow(im)
        fig.colorbar(im, ax=axes)
        if title:
            axes.set_title(title)
        fig.show()
        return fig, axes

    def save_image(self, path):
        im = self.get_image()
        vim = VimbaImage()
        vim.image = im
        vim.path = path
        vim.exposure_time = self.get_exposure_time()
        vim.timestamp = time.time()
        vim.saveto(path)

        return im

    def get_max_value(self):
        image = self.get_averaged_image()
        return image.max()

    def get_exposure_time(self):
        return self._cam.ExposureTime.get()

    def set_exposure_time(self, exposure_time):
        return self._cam.ExposureTime.set(exposure_time)

    def auto_exposure_to_max_value(self, max_value):
        MAX_PIXEL_VALUE = 255
        DOWN_PERCENTAGE = 0.8
        UP_PERCENTAGE = 1.2
        VALUE_ERROR = 5

        currExposure = self.get_exposure_time()
        currMaxValue = self.get_max_value()

        while currMaxValue > MAX_PIXEL_VALUE:
            currExposure = DOWN_PERCENTAGE*currExposure
            self.set_exposure_time(currExposure)

        newExposure = 0
        newMaxValue = max_value
        if currMaxValue > max_value:
            newExposure = DOWN_PERCENTAGE*currExposure
            self.set_exposure_time(newExposure)
            newMaxValue = self.get_max_value()

        if currMaxValue < max_value:
            newExposure = UP_PERCENTAGE * currExposure
            self.set_exposure_time(newExposure)
            newMaxValue = self.get_max_value()

        while newMaxValue < max_value - VALUE_ERROR or newMaxValue > max_value + VALUE_ERROR:
            fit = np.polyfit([newMaxValue, currMaxValue], [newExposure, currExposure], 1, full=False)
            currExposure = newExposure
            currMaxValue = newMaxValue
            newExposure = np.polyval(fit, max_value*1.1)
            newMaxValue = self.get_max_value()
            print(newMaxValue)

        return

    def show_log_histogram(self, im=None, title=None):
        if im is None:
            im = self.get_image()
        fig, axes = plt.subplots()
        im = axes.imshow(im)
        fig.colorbar(im, ax=axes)
        if title:
            axes.set_title(title)
        fig.show()
        return fig, axes


    def set_roi(self, width, height, offset_x, offset_y):
        pixel_step = 8  # Only multiples of this are accepted
        # Setting offset to (0,0) to make sure we are able to resize the ROI according the required values
        self._cam.OffsetX.set(0)
        self._cam.OffsetY.set(0)
        self._cam.Width.set(width // pixel_step * pixel_step)
        self._cam.Height.set(height // pixel_step * pixel_step)
        self._cam.OffsetX.set(offset_x // pixel_step * pixel_step)
        self._cam.OffsetY.set(offset_y // pixel_step * pixel_step)

    def get_roi(self):
        offset_x = self._cam.OffsetX.get()
        offset_y = self._cam.OffsetY.get()
        width = self._cam.Width.get()
        height = self._cam.Height.get()
        return width, height, offset_x, offset_y

    def close(self):
        self._cam.__exit__(0, 0, 0)
        self._vimb.__exit__(0, 0, 0)

