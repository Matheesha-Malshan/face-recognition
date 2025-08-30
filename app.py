# app.py
import asyncio
import base64
import json
from io import BytesIO
from typing import Optional
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI()

# Allow CORS for testing (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def process_frame(cv_img: np.ndarray) -> dict:
    """
    Process the incoming frame (cv_img: BGR numpy array).
    Replace or extend this function to run face detection, inference, etc.

    Return a dict that will be JSON-serialized and sent back to the client.
    """
    # Example: compute simple stats (frame shape) and return
    h, w = cv_img.shape[:2]
    # placeholder: count "non-black" pixels as a naive activity metric
    non_black = int(np.count_nonzero(cv_img))
    await asyncio.sleep(0)  # keep function async-friendly
    return {"height": h, "width": w, "non_black_pixels": non_black}


def b64_to_cv_image(b64_str: str) -> np.ndarray:
    """
    Convert a base64 image string (data URL or raw base64) to OpenCV BGR image.
    Accepts strings like "data:image/png;base64,AAAA..." or just the base64 payload.
    """
    # strip data url prefix if present
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return cv_img


@app.websocket("/ws/frames")
async def websocket_frames(ws: WebSocket):
    """
    WebSocket endpoint that receives base64 frames from clients.
    Expected incoming messages:
      - JSON string: {"type": "frame", "data": "<base64ImageData>"}
      - or plain base64 string (less preferred)
    Server replies with JSON messages (stringified).
    """
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            # try parse JSON; fallback to raw base64
            try:
                payload = json.loads(msg)
                if isinstance(payload, dict) and payload.get("type") == "frame":
                    b64 = payload.get("data", "")
                else:
                    # if it's some other JSON, ignore or handle accordingly
                    b64 = payload.get("data", "") if isinstance(payload, dict) else ""
            except json.JSONDecodeError:
                # msg is not JSON — treat as bare base64
                b64 = msg

            if not b64:
                await ws.send_text(json.dumps({"error": "no frame data received"}))
                continue

            # convert base64 -> cv image
            try:
                cv_img = b64_to_cv_image(b64)
            except Exception as e:
                await ws.send_text(json.dumps({"error": f"invalid image data: {e}"}))
                continue

            # process the frame (user-defined)
            result = await process_frame(cv_img)

            # optionally: send a small processed/annotated image back to client
            # (here we send no image, just JSON result)
            await ws.send_text(json.dumps({"type": "result", "result": result}))

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as ex:
        print("Websocket error:", ex)
        try:
            await ws.close()
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
