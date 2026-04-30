import React, { useState, useRef, useEffect } from 'react'
import { Send, Loader2, Trash2, ImageIcon } from 'lucide-react'
import { useChat } from '../hooks/useChat'
import ChatMessage from './ChatMessage'

export default function ChatInterface({ sessionId, imageCount }) {
  const { messages, isLoading, error, send, clearChat } = useChat(sessionId)
  const [input, setInput] = useState('')
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Auto-focus
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!input.trim()) return
    send(input)
    setInput('')
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <div className="w-16 h-16 rounded-2xl bg-gray-800 flex items-center justify-center mb-4">
              <ImageIcon size={32} className="text-gray-600" />
            </div>
            <h2 className="text-xl font-medium text-gray-300 mb-2">
              Vision RAG Assistant
            </h2>
            <p className="text-sm text-center max-w-md">
              Upload images using the button above, then ask questions about them.
              I'll analyze objects, find correlations, and answer your queries.
            </p>
            {imageCount > 0 && (
              <p className="text-sm mt-4 text-purple-400">
                {imageCount} image{imageCount !== 1 ? 's' : ''} ready — ask me anything!
              </p>
            )}
          </div>
        )}

        {messages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} />
        ))}

        {isLoading && (
          <div className="flex items-center gap-2 text-gray-400 text-sm">
            <Loader2 size={16} className="animate-spin" />
            Analyzing and generating response...
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-gray-800 p-4">
        {error && (
          <p className="text-red-400 text-xs mb-2 px-2">{error}</p>
        )}
        <form onSubmit={handleSubmit} className="flex items-end gap-3">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                imageCount === 0
                  ? 'Upload an image first, then ask a question...'
                  : 'Ask about your images...'
              }
              rows={1}
              className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 pr-12
                text-sm text-white placeholder-gray-500 resize-none
                focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50
                transition-colors"
              style={{ minHeight: '44px', maxHeight: '120px' }}
              disabled={isLoading}
            />
          </div>

          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="p-3 rounded-xl bg-purple-600 hover:bg-purple-500 disabled:opacity-40
              disabled:cursor-not-allowed transition-colors"
          >
            <Send size={18} className="text-white" />
          </button>

          {messages.length > 0 && (
            <button
              type="button"
              onClick={clearChat}
              className="p-3 rounded-xl bg-gray-800 hover:bg-gray-700 transition-colors"
              title="Clear chat"
            >
              <Trash2 size={18} className="text-gray-400" />
            </button>
          )}
        </form>
      </div>
    </div>
  )
}