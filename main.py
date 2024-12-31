import zmq
import asyncio
import json
import asyncio

from inference import FrameInference
from live_vision_handler import image_handler

context = zmq.Context()
socket = context.socket(zmq.ROUTER)
socket.bind("tcp://localhost:7207")

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
            print("JSON decode error, possibly uuid please ignore for now.")
            continue

        socket.send(b"Cam Recieved")

if __name__ == "__main__":
    print("TAP Vision System")
    # get per frame camera
    asyncio.run(main())
