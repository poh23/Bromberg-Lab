from ExperimentTools.utils.vimba_camera import VimbaCamera

cam = VimbaCamera(0)
print(cam.get_max_value())
print(cam.get_exposure_time())

cam.auto_exposure_to_max_value(200)

print(cam.get_max_value())
print(cam.get_exposure_time())


