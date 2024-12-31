from sys import maxsize
import cv2 as cv
import asyncio
import json
import base64
from websockets.asyncio.server import ServerConnection, serve
from inference import FrameInference
import numpy as np

inferer = FrameInference()

cont = False

queue = asyncio.Queue(maxsize=10)

async def message_handler(websocket: ServerConnection) -> None:
    async for message in websocket:
        data = json.loads(json.loads(message))

        if data["message"] == "infer":
            await process_frame(data)

async def process_frame(data):
    # Decode image in a separate thread
    imageDecoded = await asyncio.to_thread(base64.b64decode, data["content"])
    image = await asyncio.to_thread(
        lambda: cv.imdecode(np.frombuffer(imageDecoded, np.uint8), cv.IMREAD_COLOR)
    )

    inferedImage = await asyncio.to_thread(lambda: inferer.forward(image))

    cv.imshow("Frame", inferedImage)
    cv.waitKey(1)



async def main():
    async with serve(message_handler, "localhost", 7207) as server:
        await server.serve_forever()

if __name__ == "__main__":
    print("TAP Vision System")
    # get per frame camera
    asyncio.run(main())
    cv.destroyAllWindows()
