import zmq
import asyncio
import json
import asyncio
import logging

from inference import FrameInference
from live_vision_handler import image_handler

context = zmq.Context()
socket = context.socket(zmq.ROUTER)
socket.bind("tcp://localhost:7207")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, handlers=[
    logging.FileHandler("zeusy.log"),
    logging.StreamHandler()
])

inferer = FrameInference()

async def main():
    while True:
        message = socket.recv_string()
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
