from agent_display import TsukuyomichanVisualizationGenerator, TsukuyomichanVisualizer, WallClock
import cv2
import numpy as np

generator = TsukuyomichanVisualizationGenerator("character_images/sample2.png", WallClock(), transparent_background=False)

import time

while True:
    pil_image = generator.generate()
    s = time.time()
    cv2.imshow("Test Image", np.array(pil_image)[:, :, ::-1])
    cv2.waitKey(3)
    # print(time.time() - s)