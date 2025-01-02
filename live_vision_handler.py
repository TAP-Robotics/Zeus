import asyncio
import base64
import cv2 as cv
import numpy as np

from inference import FrameInference

async def image_handler(data, inferer: FrameInference):
    # Decode image in a separate thread
    imageDecoded = await asyncio.to_thread(base64.b64decode, data)
    image = await asyncio.to_thread(
        lambda: cv.imdecode(np.frombuffer(imageDecoded, np.uint8), cv.IMREAD_COLOR)
    )

    un_im = await asyncio.to_thread(lambda: inferer.get_undistort(image))

    inferedImage = await asyncio.to_thread(lambda: inferer.forward(image))


    cv.imshow("Server / Inference (Python)", un_im)
    cv.waitKey(1)
