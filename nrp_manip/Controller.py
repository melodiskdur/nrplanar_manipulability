import matplotlib.pyplot as plt
import matplotlib.widgets as plt_widgets
from typing import Callable
import numpy as np


class ControllerWindow:
    def __init__(self, num_of_joints: int, update_callback: Callable):
        self._width = 8
        self._height = 0.3 * num_of_joints
        self._figsize = (self._width, self._height + 1)
        slider_height = 0.25
        vertical_gap = 0.5 * self._height / (num_of_joints - 1)
        self._fig = plt.figure("Controller Window", figsize=self._figsize)
        self._sliders = [plt_widgets.Slider(plt.axes([0.25, 0.05 + i * vertical_gap, 0.65, slider_height]), label="q"+str(i+1), valmin=-np.pi, valmax=np.pi, valinit=0.05, valstep=0.1)
                                            for i in range(num_of_joints)]
        for slider in self._sliders:
            slider.on_changed(update_callback)

    def config(self):
        return [slider.val for slider in self._sliders]