/**
 * Chat hook — manages chat state, message sending, and history.
 * Reused by any component that needs chat functionality.
 */

import { useState, useCallback, useRef } from 'react'
import { sendMessage } from '../services/api'

export function useChat(sessionId = 'default') {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const abortRef = useRef(null)

  const addMessage = useCallback((role, content, extra = {}) => {
    const msg = {
      id: Date.now() + Math.random(),
      role,
      content,
      timestamp: new Date().toISOString(),
      ...extra,
    }
    setMessages((prev) => [...prev, msg])
    return msg
  }, [])

  const send = useCallback(
    async (text) => {
      if (!text.trim() || isLoading) return

      setError(null)
      addMessage('user', text)
      setIsLoading(true)

      try {
        const response = await sendMessage(text, sessionId)
        addMessage('assistant', response.answer, {
          referencedImages: response.referenced_images || [],
          correlations: response.correlations_used || [],
        })
      } catch (err) {
        const errorMsg =
          err.response?.data?.detail || 'Failed to get response. Please try again.'
        setError(errorMsg)
        addMessage('assistant', `Error: ${errorMsg}`)
      } finally {
        setIsLoading(false)
      }
    },
    [sessionId, isLoading, addMessage]
  )

  const clearChat = useCallback(() => {
    setMessages([])
    setError(null)
  }, [])

  return { messages, isLoading, error, send, clearChat }
}