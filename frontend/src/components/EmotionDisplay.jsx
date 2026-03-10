/**
 * EmotionDisplay Component
 * 
 * Displays detailed emotion prediction resultsfor detected faces.
 */

import React from 'react';
import './EmotionDisplay.css';

const EmotionDisplay = ({ faces }) => {
  if (!faces || faces.length === 0) {
    return (
      <div className="emotion-display empty">
        <p className="no-faces-message">
          No faces detected. Please face the camera.
        </p>
      </div>
    );
  }

  return (
    <div className="emotion-display">
      <h3 className="display-title">
        Detected Emotions ({faces.length} {faces.length === 1 ? 'Face' : 'Faces'})
      </h3>
      
      <div className="faces-grid">
        {faces.map((face, index) => (
          <FaceCard key={index} face={face} index={index} />
        ))}
      </div>
    </div>
  );
};

const FaceCard = ({ face, index }) => {
  const { emotion, confidence, probabilities } = face;
  
  // Sort emotions by probability
  const sortedEmotions = Object.entries(probabilities)
    .sort(([, a], [, b]) => b - a);
  
  // Get emotion color
  const getEmotionColor = (emotionName) => {
    const colors = {
      angry: '#ff0000',
      disgust: '#008000',
      fear: '#800080',
      happy: '#ffff00',
      sad: '#0000ff',
      surprise: '#ffa500',
      neutral: '#808080'
    };
    return colors[emotionName.toLowerCase()] || '#ffffff';
  };
  
  // Get emotion emoji
  const getEmotionEmoji = (emotionName) => {
    const emojis = {
      angry: '😠',
      disgust: '🤢',
      fear: '😨',
      happy: '😊',
      sad: '😢',
      surprise: '😲',
      neutral: '😐'
    };
    return emojis[emotionName.toLowerCase()] || '😶';
  };

  return (
    <div className="face-card">
      <div 
        className="face-header"
        style={{ borderLeftColor: getEmotionColor(emotion) }}
      >
        <span className="face-number">Face {index + 1}</span>
        <div className="primary-emotion">
          <span className="emotion-emoji">{getEmotionEmoji(emotion)}</span>
          <span className="emotion-name">{emotion.toUpperCase()}</span>
          <span className="emotion-confidence">
            {(confidence * 100).toFixed(1)}%
          </span>
        </div>
      </div>
      
      <div className="probabilities">
        <h4 className="prob-title">All Emotions:</h4>
        {sortedEmotions.map(([emotionName, prob]) => (
          <div key={emotionName} className="prob-bar-container">
            <div className="prob-label">
              <span className="prob-emoji">{getEmotionEmoji(emotionName)}</span>
              <span className="prob-name">{emotionName}</span>
              <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
            </div>
            <div className="prob-bar-bg">
              <div 
                className="prob-bar-fill"
                style={{ 
                  width: `${prob * 100}%`,
                  backgroundColor: getEmotionColor(emotionName)
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default EmotionDisplay;
