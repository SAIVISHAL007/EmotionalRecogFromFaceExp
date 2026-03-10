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
          <div className="header-icon">◈</div>
          <div>
            <h1 className="app-title">Facial Emotion Recognition</h1>
            <p className="app-subtitle">Real-time CNN-based emotion detection</p>
          </div>
        </div>
        
        <div className="header-actions">
          <button 
            className="info-button"
            onClick={() => setShowInfo(!showInfo)}
            title="About"
          >
            ℹ
          </button>
          
          <div className={`status-badge ${backendStatus === 'connected' ? 'connected' : 'disconnected'}`}>
            {backendStatus === 'checking' && 'Connecting...'}
            {backendStatus === 'connected' && 'Connected'}
            {backendStatus === 'no-model' && 'No Model'}
            {backendStatus === 'disconnected' && 'Offline'}
          </div>
          
          <button 
            className={`toggle-button ${isActive ? 'stop' : 'start'}`}
            onClick={handleToggle}
            disabled={backendStatus !== 'connected'}
          >
            {isActive ? 'Stop' : 'Start'}
          </button>
        </div>
      </header>

      {/* Info Panel */}
      {showInfo && (
        <div className="info-panel">
          <h3>System Information</h3>
          <div className="info-content">
            <p>
              Facial emotion recognition system using deep convolutional neural networks.
              Detects seven emotion classes in real-time from webcam input.
            </p>
            
            {modelInfo && (
              <div className="model-details">
                <strong>Model Details:</strong>
                <ul>
                  <li>Classes: {modelInfo.num_classes}</li>
                  <li>Input: 48×48 grayscale images</li>
                  <li>Emotions: {modelInfo.emotion_labels.join(', ')}</li>
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="app-content">
        {backendStatus === 'disconnected' && (
          <div style={{
            padding: '40px 20px',
            textAlign: 'center',
            color: '#c53030',
            maxWidth: '600px',
            margin: '0 auto'
          }}>
            <h2 style={{ marginBottom: '16px', fontSize: '24px' }}>Backend Offline</h2>
            <p style={{ marginBottom: '12px' }}>Cannot connect to emotion recognition backend.</p>
            <code style={{ background: '#f7fafc', padding: '12px', borderRadius: '6px', display: 'block', marginTop: '12px' }}>
              uvicorn backend.main:app --host 0.0.0.0 --port 8000
            </code>
          </div>
        )}

        {backendStatus === 'no-model' && (
          <div style={{
            padding: '40px 20px',
            textAlign: 'center',
            color: '#c05621',
            maxWidth: '600px',
            margin: '0 auto'
          }}>
            <h2 style={{ marginBottom: '16px', fontSize: '24px' }}>Model Not Found</h2>
            <p style={{ marginBottom: '12px' }}>Backend is running but CNN model is not loaded.</p>
            <code style={{ background: '#f7fafc', padding: '12px', borderRadius: '6px', display: 'block', marginTop: '12px' }}>
              python model/train.py
            </code>
          </div>
        )}

        {backendStatus === 'connected' && (
          <>
            {/* Webcam Feed */}
            <WebcamFeed 
              isActive={isActive}
              captureInterval={200}
              onFacesDetected={setCurrentFaces}
            />

            {/* Emotion Display */}
            <EmotionDisplay faces={isActive ? currentFaces : []} />
          </>
        )}
      </div>
    </div>
  );
}

export default App;
