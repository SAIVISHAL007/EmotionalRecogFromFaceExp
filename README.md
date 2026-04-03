# Facial Emotion Recognition

A full-stack facial emotion recognition project with a FastAPI backend and a React frontend. The current default runtime detects multiple faces from a webcam, draws bounding boxes, and predicts one of seven emotion classes in real time.

## Why This Project Stands Out

This is not just a single-model demo. It shows an end-to-end application flow:

- Browser webcam capture
- Backend API inference
- Multi-face tracking in one frame
- Live bounding-box overlays and emotion labels
- A clean separation between the default runtime and optional alternate workflows

That makes it easier to explain in interviews as a real product-shaped system rather than a notebook-only ML experiment.

## Practical Use Cases

The project can be positioned honestly for scenarios like:

- classroom engagement monitoring
- telemedicine or remote support demos
- customer interaction analytics
- general live video emotion analysis research

The code in this repository is a working prototype for those kinds of workflows, not a finished commercial product.

## What Is Implemented

- Real-time webcam capture in the browser
- FastAPI backend with `/api/health`, `/api/model-info`, `/api/predict`, and `/api/predict-base64`
- React frontend that connects to the backend and displays live predictions
- YuNet DNN face detection as the primary runtime detector
- CNN-based emotion prediction on 48×48 grayscale face crops
- Multi-face support in a single frame
- Per-face smoothing to reduce flicker across frames
- Backend and frontend proxy wiring for local development

## What Runs By Default

The live application currently uses:

- Face detection: YuNet DNN
- Emotion classifier: CNN
- Frontend: React + Vite
- Backend: FastAPI

If the YuNet detector is unavailable, the backend falls back to a legacy detector path. The MediaPipe + Random Forest workflow is optional and not the default application path.

## What To Emphasize In A Recruiter Interview

- You built a real-time full-stack ML application, not only a trained model
- The backend and frontend are connected through a defined API contract
- The system supports multiple faces in a single frame
- The default runtime path is simple and stable, which helps avoid unnecessary prediction changes
- Optional workflows are isolated so they do not affect the main app unless explicitly enabled

## Optional or Legacy Paths

These are included in the repository, but they are not the primary runtime path:

- Haar Cascade fallback detector
- MediaPipe + Random Forest alternate workflow
- Standalone OpenCV webcam app

## Tech Stack

- Python 3.11 in the current workspace
- FastAPI
- TensorFlow / Keras
- OpenCV
- React 18
- Vite
- Axios
- scikit-learn
- MediaPipe for the optional alternate workflow

## Repository Structure

- `backend/` - API and inference service
- `frontend/` - React UI
- `realtime/` - detectors, predictors, and standalone webcam code
- `model/` - CNN training and evaluation scripts
- `data/` - dataset helpers and preprocessing
- `trained_models/` - saved model artifacts

## Run Locally

Backend:

```bash
cd backend
python main.py
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Or from the project root, run the backend with uvicorn:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The frontend dev server proxies API requests to `http://127.0.0.1:8000`.

## Current Behavior

- Multiple faces are detected in a single frame
- Each face gets its own bounding box and emotion prediction
- The backend health endpoint returns `model_loaded: true` when the CNN model is available
- The frontend only shows live predictions when the backend is reachable

## Honest Limitations

- The local CNN model is the default runtime model and is the one currently wired into the main app
- The optional MediaPipe + Random Forest workflow is included for experimentation and alternate training, but it is not the main production path in this repository
- The Haar detector and standalone webcam app are legacy fallbacks, not the primary experience
- Accuracy numbers can vary depending on dataset split, image quality, lighting, and face pose, so they should be discussed as measured results rather than guarantees

## Notes

If you are reviewing this repo as a recruiter, the main things to look at are the end-to-end app flow, the multi-face runtime support, the FastAPI backend, and the browser-based webcam UI.

## License

MIT
