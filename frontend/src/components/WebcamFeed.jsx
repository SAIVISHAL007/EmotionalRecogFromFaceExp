/**
 * WebcamFeed Component
 * 
 * Captures video from user's webcam, sends frames to backend,
 * and displays real-time emotion predictions.
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { predictEmotion } from '../services/api';
import './WebcamFeed.css';

const WebcamFeed = ({ isActive, captureInterval = 200, onFacesDetected }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [predictions, setPredictions] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [fps, setFps] = useState(0);
  const [stats, setStats] = useState({ processed: 0, detected: 0 });
  
  const intervalRef = useRef(null);
  const lastFrameTimeRef = useRef(Date.now());

  // Start webcam
  useEffect(() => {
    const startWebcam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user'
          }
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        
        setError(null);
      } catch (err) {
        console.error('Webcam error:', err);
        setError('Failed to access webcam. Please grant camera permissions.');
      }
    };

    startWebcam();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // Capture and process frames
  const captureFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || isProcessing) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    // Resize for faster processing - use smaller resolution
    const targetWidth = 384;
    const targetHeight = 288;
    canvas.width = targetWidth;
    canvas.height = targetHeight;

    // Draw current frame to canvas (scaled down)
    context.drawImage(video, 0, 0, targetWidth, targetHeight);

    // Convert canvas to base64 with lower quality for speed
    const base64Image = canvas.toDataURL('image/jpeg', 0.75);

    setIsProcessing(true);

    try {
      // Send to backend
      const response = await predictEmotion(base64Image);
      
      // Update predictions
      setPredictions(response.faces || []);
      
      // Notify parent component
      if (onFacesDetected) {
        onFacesDetected(response.faces || []);
      }
      
      // Update stats
      setStats(prev => ({
        processed: prev.processed + 1,
        detected: prev.detected + response.num_faces
      }));
      
      // Calculate FPS
      const now = Date.now();
      const timeDiff = now - lastFrameTimeRef.current;
      setFps(Math.round(1000 / timeDiff));
      lastFrameTimeRef.current = now;
      
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  }, [isProcessing]);

  // Start/stop frame capture
  useEffect(() => {
    if (isActive) {
      intervalRef.current = setInterval(captureFrame, captureInterval);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isActive, captureFrame, captureInterval]);

  // Get emotion color
  const getEmotionColor = (emotion) => {
    const colors = {
      angry: '#ff0000',
      disgust: '#008000',
      fear: '#800080',
      happy: '#ffff00',
      sad: '#0000ff',
      surprise: '#ffa500',
      neutral: '#808080'
    };
    return colors[emotion.toLowerCase()] || '#ffffff';
  };

  return (
    <div className="webcam-container">
      <div className="video-wrapper">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="webcam-video"
        />
        <canvas ref={canvasRef} style={{ display: 'none' }} />
        
        {/* Overlay predictions */}
        <svg className="predictions-overlay">
          {predictions.map((face, index) => {
            // Scale coordinates from processed image (320x240) to video display size
            const video = videoRef.current;
            const scaleX = video ? video.videoWidth / 384 : 2;
            const scaleY = video ? video.videoHeight / 288 : 2;
            
            const [origX, origY, origW, origH] = face.bbox;
            const x = origX * scaleX;
            const y = origY * scaleY;
            const w = origW * scaleX;
            const h = origH * scaleY;
            
            const color = getEmotionColor(face.emotion);
            
            return (
              <g key={index}>
                {/* Bounding box */}
                <rect
                  x={x}
                  y={y}
                  width={w}
                  height={h}
                  fill="none"
                  stroke={color}
                  strokeWidth="3"
                />
                
                {/* Emotion label */}
                <rect
                  x={x}
                  y={y - 30}
                  width={w}
                  height="30"
                  fill={color}
                  opacity="0.8"
                />
                <text
                  x={x + 5}
                  y={y - 10}
                  fill="white"
                  fontSize="16"
                  fontWeight="bold"
                >
                  {face.emotion}: {(face.confidence * 100).toFixed(1)}%
                </text>
              </g>
            );
          })}
        </svg>
        
        {/* FPS Counter */}
        <div className="fps-counter">
          FPS: {fps}
        </div>
        
        {/* Status indicator */}
        <div className={`status-indicator ${isActive ? 'active' : 'inactive'}`}>
          {isActive ? '● REC' : '○ PAUSED'}
        </div>
      </div>

      {/* Error message */}
      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}

      {/* Stats */}
      <div className="stats">
        <div className="stat-item">
          <span className="stat-label">Frames:</span>
          <span className="stat-value">{stats.processed}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Faces:</span>
          <span className="stat-value">{stats.detected}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Current:</span>
          <span className="stat-value">{predictions.length}</span>
        </div>
      </div>
    </div>
  );
};

export default WebcamFeed;
