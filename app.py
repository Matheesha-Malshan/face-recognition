import asyncio
import json
import time
import base64
import cv2
import numpy as np
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from sklearn.preprocessing import normalize
from PIL import Image
from torchvision import transforms
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Face Recognition API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize face recognition models
face_detector = MTCNN(keep_all=True)
face_recognizer = InceptionResnetV1(pretrained='vggface2').eval()

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load embeddings and labels (update paths as needed)
try:
    embeddings = np.load("/home/matheesha/Documents/se/embeddings .npy")
    labels = np.load("/home/matheesha/Documents/se/labels .npy")
    logger.info(f"Loaded {len(embeddings)} embeddings and {len(labels)} labels")
except FileNotFoundError:
    logger.error("Could not load embeddings or labels files")
    embeddings = None
    labels = None

# Configuration
THRESHOLD = 0.5
ATTENDANCE_API_URL = "http://192.168.166.68:8080/attendance/mark"

# Global state management
class FaceRecognitionState:
    def __init__(self):
        self.previous_label = None
        self.recognized_people = []
        self.active_connections: List[WebSocket] = []
    
    def reset_recognition(self):
        """Reset recognition state when person with ID '4' is detected"""
        self.recognized_people = []
        self.previous_label = None
        logger.info("Recognition state reset")

# Global state instance
recognition_state = FaceRecognitionState()

def get_embedding(face_image):
    """Extract face embedding from image"""
    try:
        img_tensor = transform(face_image).unsqueeze(0)
        with torch.no_grad():
            embedding = face_recognizer(img_tensor).cpu().numpy().squeeze()
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return None

def predict_person(face_image):
    """Predict person identity from face image"""
    if embeddings is None or labels is None:
        return "Unknown", 1.0
    
    new_embedding = get_embedding(face_image)
    if new_embedding is None:
        return "Unknown", 1.0
    
    new_embedding = normalize(new_embedding.reshape(1, -1))
    distances = np.linalg.norm(embeddings - new_embedding, axis=1)
    
    min_distance = np.min(distances)
    min_index = np.argmin(distances)
    
    if min_distance > THRESHOLD:
        return "Unknown", min_distance
    else:
        return labels[min_index], min_distance

async def mark_attendance(student_id: str):
    """Mark attendance for recognized person"""
    try:
        payload = {
            "studentId": student_id,
            "attendanceDate": time.strftime("%Y-%m-%d"),
            "attendanceTime": time.strftime("%I:%M %p"),
            "attendanceStatus": True
        }
        
        headers = {'Content-Type': 'application/json'}
        
        # Use async request in production
        response = requests.post(
            ATTENDANCE_API_URL,
            headers=headers,
            json=payload,
            timeout=10
        )
        
        logger.info(f"Attendance marked for {student_id}: {response.status_code}")
        return response.status_code == 200
        
    except Exception as e:
        logger.error(f"Error marking attendance for {student_id}: {e}")
        return False

def process_frame(frame_data: str):
    """Process base64 encoded frame and perform face recognition"""
    try:
        # Decode base64 image
        img_data = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        boxes, _ = face_detector.detect(rgb_frame)
        
        results = []
        
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                face = frame[y1:y2, x1:x2]
                if face.shape[0] > 0 and face.shape[1] > 0:
                    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    predicted_person, distance = predict_person(face_pil)
                    
                    # Handle recognition logic
                    new_person_detected = False
                    if predicted_person != "Unknown" and predicted_person != recognition_state.previous_label:
                        recognition_state.previous_label = predicted_person
                        
                        # Reset state if person with ID '4' is detected
                        if predicted_person == "4":
                            recognition_state.reset_recognition()
                        
                        # Add new person if not already recognized and not the reset person
                        elif predicted_person not in recognition_state.recognized_people:
                            recognition_state.recognized_people.append(predicted_person)
                            new_person_detected = True
                            logger.info(f"Recognized: {predicted_person} at {time.strftime('%H:%M:%S')}")
                    
                    results.append({
                        'bbox': [x1, y1, x2, y2],
                        'person': predicted_person,
                        'distance': float(distance),
                        'new_detection': new_person_detected,
                        'timestamp': time.strftime('%H:%M:%S')
                    })
        
        return {
            'faces': results,
            'recognized_people': recognition_state.recognized_people.copy(),
            'timestamp': time.time()
        }
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return None

@app.websocket("/ws/face-recognition")
async def websocket_face_recognition(websocket: WebSocket):
    """WebSocket endpoint for real-time face recognition"""
    await websocket.accept()
    recognition_state.active_connections.append(websocket)
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get('type') == 'frame':
                frame_data = message.get('data')
                if frame_data:
                    # Process frame
                    result = process_frame(frame_data)
                    
                    if result:
                        # Mark attendance for new detections
                        for face in result['faces']:
                            if face['new_detection'] and face['person'] != "Unknown":
                                await mark_attendance(face['person'])
                        
                        # Send results back
                        await websocket.send_text(json.dumps({
                            'type': 'recognition_result',
                            'data': result
                        }))
            
            elif message.get('type') == 'reset':
                # Manual reset
                recognition_state.reset_recognition()
                await websocket.send_text(json.dumps({
                    'type': 'reset_confirmed',
                    'data': {'recognized_people': []}
                }))
                
    except WebSocketDisconnect:
        recognition_state.active_connections.remove(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in recognition_state.active_connections:
            recognition_state.active_connections.remove(websocket)

@app.post("/api/process-image")
async def process_single_image(image_data: dict):
    """REST endpoint to process a single image"""
    try:
        frame_data = image_data.get('image')
        if not frame_data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        result = process_frame(frame_data)
        if result is None:
            raise HTTPException(status_code=400, detail="Failed to process image")
        
        # Mark attendance for new detections
        for face in result['faces']:
            if face['new_detection'] and face['person'] != "Unknown":
                await mark_attendance(face['person'])
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing single image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status():
    """Get current recognition status"""
    return {
        'recognized_people': recognition_state.recognized_people,
        'active_connections': len(recognition_state.active_connections),
        'embeddings_loaded': embeddings is not None,
        'labels_loaded': labels is not None,
        'threshold': THRESHOLD
    }

@app.post("/api/reset")
async def reset_recognition():
    """Reset recognition state"""
    recognition_state.reset_recognition()
    
    # Notify all connected clients
    for connection in recognition_state.active_connections:
        try:
            await connection.send_text(json.dumps({
                'type': 'reset_notification',
                'data': {'recognized_people': []}
            }))
        except:
            pass
    
    return {'message': 'Recognition state reset', 'recognized_people': []}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        'message': 'Face Recognition API',
        'version': '1.0.0',
        'endpoints': {
            'websocket': '/ws/face-recognition',
            'process_image': '/api/process-image',
            'status': '/api/status',
            'reset': '/api/reset'
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)