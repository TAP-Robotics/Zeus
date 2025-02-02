import json
import asyncio
import logging
import io

from concurrent.futures import ThreadPoolExecutor
from pynng import Pair0

from inference import FrameInference
from live_vision_handler import image_handler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, handlers=[
    logging.FileHandler("zeusy.log"),
    logging.StreamHandler()
])

inferer = FrameInference()



async def main():
    with Pair0(listen="tcp://127.0.0.1:7207") as socket:
        while True:
            message = await socket.arecv()
            try:
                data = json.loads(message)
                if(data["message"] == "vision_infer"):
                    imgData = data["content"]
                    await image_handler(imgData, inferer)

            except json.JSONDecodeError:
                logger.info("Not a JSON string (probably frame UUID), continuing...")
                continue

if __name__ == "__main__":
    logger.info("Zeus System | TAP Robotics")
    # get per frame camera
    asyncio.run(main())
