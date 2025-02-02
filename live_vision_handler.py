import asyncio
import base64
import cv2 as cv
import numpy as np

from inference import FrameInference

async def image_handler(data, inferer: FrameInference):
    # Decode image in a separate thread
    imageDecoded = await asyncio.get_running_loop().run_in_executor(None, base64.b64decode, data)
    print(imageDecoded)
    image = await asyncio.get_running_loop().run_in_executor(None,
        lambda: cv.imdecode(np.frombuffer(imageDecoded, np.uint8), cv.IMREAD_COLOR)
    )

    cv.imshow("Server / Inference (Python)", image)
    cv.waitKey(1)
