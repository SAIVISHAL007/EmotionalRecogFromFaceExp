/**
 * Main App Component
 * 
 * Root component for the Facial Emotion Recognition web application.
 */

import React, { useState, useEffect } from 'react';
import WebcamFeed from './components/WebcamFeed';
import EmotionDisplay from './components/EmotionDisplay';
import { checkHealth, getModelInfo } from './services/api';
import './App.css';

function App() {
  const [isActive, setIsActive] = useState(false);
  const [currentFaces, setCurrentFaces] = useState([]);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [modelInfo, setModelInfo] = useState(null);
  const [showInfo, setShowInfo] = useState(false);

  // Check backend health on mount
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const health = await checkHealth();
        if (health.model_loaded) {
          setBackendStatus('connected');
          
          // Get model info
          const info = await getModelInfo();
          setModelInfo(info);
        } else {
          setBackendStatus('no-model');
        }
      } catch (error) {
        console.error('Backend check failed:', error);
        setBackendStatus('disconnected');
      }
    };

    checkBackend();
  }, []);

  const handleToggle = () => {
    if (backendStatus === 'connected') {
      setIsActive(!isActive);
    }
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <h1 className="app-title">
            <span className="title-emoji">🎭</span>
            Facial Emotion Recognition
          </h1>
          <p className="app-subtitle">
            Real-time emotion detection powered by CNN deep learning
          </p>
        </div>
        
        <div className="header-actions">
          <button 
            className="info-button"
            onClick={() => setShowInfo(!showInfo)}
            title="About"
          >
            ℹ️
          </button>
          
          <div className={`status-badge status-${backendStatus}`}>
            {backendStatus === 'checking' && '⏳ Checking...'}
            {backendStatus === 'connected' && '✅ Connected'}
            {backendStatus === 'no-model' && '⚠️ No Model'}
            {backendStatus === 'disconnected' && '❌ Disconnected'}
          </div>
        </div>
      </header>

      {/* Info Panel */}
      {showInfo && (
        <div className="info-panel">
          <div className="info-content">
            <h3>About This System</h3>
            <p>
              This is a real-time facial emotion recognition system built for academic purposes.
              It uses a Convolutional Neural Network (CNN) trained on the FER-2013 dataset to classify
              seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
            </p>
            
            {modelInfo && (
              <div className="model-details">
                <h4>Model Information:</h4>
                <ul>
                  <li>Classes: {modelInfo.num_classes}</li>
                  <li>Parameters: {modelInfo.total_parameters.toLocaleString()}</li>
                  <li>Input Size: 48x48 grayscale</li>
                  <li>Emotions: {modelInfo.emotion_labels.join(', ')}</li>
                </ul>
              </div>
            )}
            
            <h4>How to Use:</h4>
            <ol>
              <li>Ensure your webcam is connected and permissions are granted</li>
              <li>Click the "Start Detection" button</li>
              <li>Face the camera and watch as emotions are detected in real-time</li>
              <li>View detailed probability breakdowns for each detected face</li>
            </ol>
            
            <p className="project-credit">
              <strong>Academic Project:</strong> Neural Networks & Deep Learning
            </p>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="app-main">
        {backendStatus === 'disconnected' && (
          <div className="error-card">
            <h2>❌ Backend Not Available</h2>
            <p>Cannot connect to the emotion recognition backend.</p>
            <p>Please ensure the backend server is running:</p>
            <code>python backend/main.py</code>
          </div>
        )}

        {backendStatus === 'no-model' && (
          <div className="error-card">
            <h2>⚠️ Model Not Loaded</h2>
            <p>The backend is running but the CNN model is not loaded.</p>
            <p>Please train the model first:</p>
            <code>python model/train.py</code>
          </div>
        )}

        {backendStatus === 'connected' && (
          <>
            {/* Control Panel */}
            <div className="control-panel">
              <button 
                className={`control-button ${isActive ? 'stop' : 'start'}`}
                onClick={handleToggle}
              >
                {isActive ? (
                  <>
                    <span className="button-icon">⏸️</span>
                    Stop Detection
                  </>
                ) : (
                  <>
                    <span className="button-icon">▶️</span>
                    Start Detection
                  </>
                )}
              </button>
              
              {isActive && (
                <p className="control-hint">
                  Detection is active. Face the camera to see your emotions!
                </p>
              )}
            </div>

            {/* Webcam Feed */}
            <WebcamFeed 
              isActive={isActive}
              captureInterval={200}
              onFacesDetected={setCurrentFaces}
            />

            {/* Emotion Display */}
            {isActive && (
              <EmotionDisplay faces={currentFaces} />
            )}
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>
          Built with React + FastAPI + TensorFlow/Keras
        </p>
        <p className="footer-tech">
          🔬 CNN • 📹 OpenCV • ⚡ Real-time Processing
        </p>
      </footer>
    </div>
  );
}

export default App;
