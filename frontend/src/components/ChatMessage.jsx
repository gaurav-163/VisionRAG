import React from 'react'
import { User, Bot, Image as ImageIcon } from 'lucide-react'

/**
 * Single chat message — reused for both user and assistant messages.
 */
export default function ChatMessage({ message }) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-600 to-blue-600 flex items-center justify-center flex-shrink-0 mt-1">
          <Bot size={16} className="text-white" />
        </div>
      )}

      <div className={`max-w-[75%] space-y-2`}>
        {/* Message bubble */}
        <div
          className={`px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap
            ${isUser
              ? 'bg-blue-600 text-white rounded-tr-md'
              : 'bg-gray-800 text-gray-200 rounded-tl-md'
            }`}
        >
          {message.content}
        </div>

        {/* Referenced images */}
        {message.referencedImages?.length > 0 && (
          <div className="flex flex-wrap gap-1.5 px-1">
            {message.referencedImages.map((imgId) => (
              <span
                key={imgId}
                className="inline-flex items-center gap-1 text-xs bg-gray-800 text-gray-400 px-2 py-1 rounded-md"
              >
                <ImageIcon size={10} />
                {imgId}
              </span>
            ))}
          </div>
        )}

        {/* Correlations */}
        {message.correlations?.length > 0 && (
          <div className="px-1">
            <details className="text-xs text-gray-500">
              <summary className="cursor-pointer hover:text-gray-300">
                {message.correlations.length} correlation(s) used
              </summary>
              <div className="mt-1 space-y-1 pl-2 border-l border-gray-700">
                {message.correlations.map((c, i) => (
                  <p key={i}>
                    {c.image_id_a} ↔ {c.image_id_b}: {c.shared_objects?.join(', ') || 'N/A'}
                    <span className="text-gray-600 ml-1">
                      ({(c.similarity_score * 100).toFixed(0)}%)
                    </span>
                  </p>
                ))}
              </div>
            </details>
          </div>
        )}

        {/* Timestamp */}
        <p className={`text-xs text-gray-600 px-1 ${isUser ? 'text-right' : ''}`}>
          {new Date(message.timestamp).toLocaleTimeString()}
        </p>
      </div>

      {isUser && (
        <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center flex-shrink-0 mt-1">
          <User size={16} className="text-gray-300" />
        </div>
      )}
    </div>
  )
}