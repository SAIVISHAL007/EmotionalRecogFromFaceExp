/**
 * EmotionDisplay Component
 * 
 * Professional 3D emotion detection display.
 * Displays emotion predictions with modern, clean design.
 */

import React from 'react';
import './EmotionDisplay.css';

const EmotionDisplay = ({ faces }) => {
  if (!faces || faces.length === 0) {
    return (
      <div className="emotion-display empty">
        <p className="no-faces-message">No Faces Detected</p>
      </div>
    );
  }

  return (
    <div className="emotion-display">
      <h3 className="display-title">Detected Emotions ({faces.length} Face{faces.length !== 1 ? 's' : ''})</h3>
      
      <div className="faces-grid">
        {faces.map((face, idx) => (
          <FaceCard key={idx} face={face} index={idx} />
        ))}
      </div>
    </div>
  );
};

const FaceCard = ({ face, index }) => {
  const { emotion, confidence, probabilities } = face;
  const sortedEmotions = Object.entries(probabilities).sort(([,a], [,b]) => b - a);
  
  const emotionIcons = {
    angry: 'A',
    disgust: 'D',
    fear: 'F',
    happy: 'H',
    sad: 'S',
    surprise: 'P',
    neutral: 'N'
  };
  
  const emotionColors = {
    angry: '#dc2626',
    disgust: '#059669',
    fear: '#7c3aed',
    happy: '#f59e0b',
    sad: '#2563eb',
    surprise: '#ec4899',
    neutral: '#6b7280'
  };
  
  const getEmotionIcon = (name) => emotionIcons[name.toLowerCase()] || '?';
  const getAccentColor = (name) => emotionColors[name.toLowerCase()] || '#888';
  
  const accentColor = getAccentColor(emotion);
  const confPercent = (confidence * 100).toFixed(1);

  return (
    <div className="face-card">
      <div className="face-header">
        <span className="face-number">Face {index + 1}</span>
      </div>
      
      <div className="primary-emotion">
        <span className="emotion-icon" style={{ background: accentColor }}>
          {getEmotionIcon(emotion)}
        </span>
        <div>
          <span className="emotion-name">{emotion.toUpperCase()}</span>
          <span className="emotion-confidence">{confPercent}% confidence</span>
        </div>
      </div>
      
      <div className="probabilities">
        <div className="prob-title">Emotion Breakdown</div>
        {sortedEmotions.map(([emotion, prob], idx) => (
          <div key={idx} className="prob-bar-container">
            <div className="prob-label">
              <div className="prob-icon" style={{ background: getAccentColor(emotion) }}>
                {getEmotionIcon(emotion)}
              </div>
              <span className="prob-name">{emotion}</span>
              <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
            </div>
            <div className="prob-bar-bg">
              <div className="prob-bar-fill" style={{ width: `${prob * 100}%`, background: getAccentColor(emotion) }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default EmotionDisplay;
