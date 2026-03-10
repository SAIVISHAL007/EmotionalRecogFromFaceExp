# Frontend - React Emotion Recognition Interface

Modern React-based web interface for real-time facial emotion recognition.

## Features

- 📹 Real-time webcam capture
- 🎯 Live emotion detection
- 📊 Detailed probability breakdowns
- 🎨 Color-coded emotion visualization
- 📱 Responsive design
- ⚡ Real-time FPS monitoring

## Technology Stack

- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **Axios** - HTTP client
- **MediaDevices API** - Webcam access
- **Canvas API** - Frame capture

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── WebcamFeed.jsx       # Webcam capture component
│   │   ├── WebcamFeed.css
│   │   ├── EmotionDisplay.jsx   # Results display component
│   │   └── EmotionDisplay.css
│   ├── services/
│   │   └── api.js               # Backend API integration
│   ├── App.jsx                  # Main application component
│   ├── App.css
│   ├── main.jsx                 # Entry point
│   └── index.css                # Global styles
├── index.html                   # HTML template
├── package.json                 # Dependencies
└── vite.config.js               # Vite configuration
```

## Installation

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Install Dependencies

```bash
cd frontend
npm install
```

## Development

### Start Dev Server

```bash
npm run dev
```

The application will be available at: http://localhost:3000

### Build for Production

```bash
npm run build
```

Output will be in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Configuration

### Backend URL

Set backend URL in `.env` file:

```env
VITE_API_URL=http://localhost:8000
```

Or it will default to `http://localhost:8000`.

## Components

### WebcamFeed

Handles webcam capture and frame processing:
- Accesses user's webcam
- Captures frames at regular intervals
- Sends frames to backend API
- Displays bounding boxes and labels
- Shows FPS and statistics

**Props:**
- `isActive` (boolean) - Whether detection is active
- `captureInterval` (number) - Milliseconds between captures (default: 500)

### EmotionDisplay

Displays detailed emotion results:
- Shows all detected faces
- Displays primary emotion with confidence
- Shows probability bars for all 7 emotions
- Color-coded by emotion type
- Animated transitions

**Props:**
- `faces` (array) - Array of face detection results

## API Integration

The application communicates with the FastAPI backend through the `api.js` service:

### Available Functions

```javascript
// Check backend health
checkHealth()

// Get model information
getModelInfo()

// Predict emotion from base64 image
predictEmotion(base64Image)

// Predict emotion from file
predictEmotionFromFile(file)
```

## Features Explained

### Real-time Detection

1. Webcam captures video stream
2. Canvas extracts frame as base64
3. Frame sent to backend via POST request
4. Backend returns emotion predictions
5. Results displayed with overlay
6. Process repeats every 500ms

### Emotion Visualization

Each emotion has a unique color:
- 🔴 **Angry**: Red
- 🟢 **Disgust**: Dark Green
- 🟣 **Fear**: Purple
- 🟡 **Happy**: Yellow
- 🔵 **Sad**: Blue
- 🟠 **Surprise**: Orange
- ⚪ **Neutral**: Gray

### Performance Monitoring

- FPS counter shows frames processed per second
- Statistics track:
  - Total frames processed
  - Total faces detected
  - Currently visible faces

## Browser Compatibility

Requires modern browser with:
- MediaDevices API (webcam access)
- Canvas API (frame capture)
- ES6+ JavaScript support

**Tested on:**
- Chrome 90+
- Firefox 88+
- Edge 90+
- Safari 14+

## Camera Permissions

The application requires camera access. Users will see a browser permission prompt on first use.

**Troubleshooting:**
- Ensure HTTPS (or localhost for development)
- Check browser camera permissions
- Only one application can use camera at a time
- Some browsers require user gesture (button click) before camera access

## Responsive Design

The interface adapts to different screen sizes:
- Desktop: Full layout with side-by-side components
- Tablet: Stacked layout
- Mobile: Single column, optimized controls

## Error Handling

The application handles various error states:

- Backend disconnected
- Model not loaded
- Webcam access denied
- Prediction failures
- Network errors

Each error shows a user-friendly message with troubleshooting steps.

## Performance Optimization

- Frame capture throttled to 500ms
- Canvas reused for frame extraction
- Predictions processed asynchronously
- Minimal re-renders with React hooks
- Lazy loading of components

## Customization

### Adjust Capture Rate

In `App.jsx`:
```jsx
<WebcamFeed 
  isActive={isActive}
  captureInterval={1000}  // Capture every 1 second
/>
```

### Change Video Resolution

In `WebcamFeed.jsx`:
```javascript
const stream = await navigator.mediaDevices.getUserMedia({
  video: {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    facingMode: 'user'
  }
});
```

## Deployment

### Static Hosting

Build and deploy to services like:
- Netlify
- Vercel
- GitHub Pages
- AWS S3 + CloudFront

### Update Backend URL

For production, update API URL:
```env
VITE_API_URL=https://your-api-domain.com
```

### Build

```bash
npm run build
```

Upload `dist/` contents to your hosting service.

## Troubleshooting

**Webcam not working?**
- Check browser permissions
- Ensure HTTPS (required for camera access)
- Try different browser
- Check if camera is available

**Backend connection failed?**
- Ensure backend is running on port 8000
- Check CORS configuration
- Verify API URL in .env

**Low FPS?**
- Increase `captureInterval`
- Reduce video resolution
- Check network latency
- Ensure backend has sufficient resources

**Predictions not showing?**
- Check browser console for errors
- Verify backend is responding
- Ensure model is loaded in backend
- Check network tab for API responses

## Academic Discussion

For your presentation:

1. **Browser APIs**
   - MediaDevices for webcam
   - Canvas for frame extraction
   - WebRTC fundamentals

2. **React Architecture**
   - Component-based design
   - State management with hooks
   - Props and data flow

3. **Real-time Processing**
   - Frame capture optimization
   - Network communication
   - Asynchronous programming

4. **User Experience**
   - Responsive design
   - Error handling
   - Visual feedback

## Next Steps

- Add face tracking across frames
- Implement emotion history graphs
- Add screenshot/recording features
- Multi-language support
- Accessibility improvements

## Credits

Built with React + Vite for Neural Networks & Deep Learning course.
