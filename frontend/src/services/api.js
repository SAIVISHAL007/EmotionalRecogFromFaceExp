/**
 * API Service for Emotion Recognition Backend
 * 
 * This service handles all HTTP requests to the FastAPI backend.
 */

import axios from 'axios';

// Use relative paths so Vite's dev-server proxy forwards /api/* to port 8000.
// In production, set VITE_API_URL to the full backend URL.
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Check backend health status
 */
export const checkHealth = async () => {
  try {
    const response = await api.get('/api/health');
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
};

/**
 * Get model information
 */
export const getModelInfo = async () => {
  try {
    const response = await api.get('/api/model-info');
    return response.data;
  } catch (error) {
    console.error('Failed to get model info:', error);
    throw error;
  }
};

/**
 * Predict emotions from base64 encoded image
 * 
 * @param {string} base64Image - Base64 encoded image string
 * @returns {Promise} Prediction response
 */
export const predictEmotion = async (base64Image) => {
  try {
    const response = await api.post('/api/predict-base64', {
      image: base64Image
    });
    return response.data;
  } catch (error) {
    console.error('Prediction failed:', error);
    if (error.response) {
      throw new Error(error.response.data.detail || 'Prediction failed');
    }
    throw new Error('Failed to connect to backend');
  }
};

/**
 * Predict emotions from file upload
 * 
 * @param {File} file - Image file
 * @returns {Promise} Prediction response
 */
export const predictEmotionFromFile = async (file) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post(
      '/api/predict',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    
    return response.data;
  } catch (error) {
    console.error('File prediction failed:', error);
    if (error.response) {
      throw new Error(error.response.data.detail || 'Prediction failed');
    }
    throw new Error('Failed to connect to backend');
  }
};

export default {
  checkHealth,
  getModelInfo,
  predictEmotion,
  predictEmotionFromFile,
};
