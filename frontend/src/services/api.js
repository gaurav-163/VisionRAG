/**
 * API service — single source of truth for all backend calls.
 * Every component uses these functions instead of calling axios directly.
 */

import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000, // VLM analysis can take time
})

// ── Upload ──

export async function uploadImage(file, sessionId = 'default', onProgress) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('session_id', sessionId)

  const response = await api.post('/upload', formData, {
    onUploadProgress: onProgress
      ? (e) => onProgress(Math.round((e.loaded * 100) / e.total))
      : undefined,
  })
  return response.data
}

export async function generateSceneGraph(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await api.post('/v1/vision/visualize/scene-graph', formData, {
    responseType: 'blob',
  })

  return {
    blob: response.data,
    nodeCount: response.headers?.['x-node-count'],
  }
}

// ── Chat ──

export async function sendMessage(message, sessionId = 'default') {
  const response = await api.post('/chat', {
    message,
    session_id: sessionId,
  })
  return response.data
}

// ── Session Data ──

export async function getSessionInfo(sessionId = 'default') {
  const response = await api.get(`/session/${sessionId}`)
  return response.data
}

export async function getSessionImages(sessionId = 'default') {
  const response = await api.get(`/images/${sessionId}`)
  return response.data
}

// ── Health ──

export async function healthCheck() {
  const response = await api.get('/health')
  return response.data
}