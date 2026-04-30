import React, { useState, useCallback } from 'react'
import Sidebar from './Sidebar'
import ImageUploader from './ImageUploader'
import ImageGallery from './ImageGallery'
import ChatInterface from './ChatInterface'

export default function App() {
  const [sessionId] = useState('default')
  const [images, setImages] = useState([])
  const [correlations, setCorrelations] = useState([])
  const [activeTab, setActiveTab] = useState('chat') // 'chat' | 'gallery'

  const handleImageProcessed = useCallback((result) => {
    setImages((prev) => [
      ...prev,
      {
        id: result.image_id,
        filename: result.filename,
        analysis: result.analysis,
        correlations: result.correlations || [],
      },
    ])
    // Merge new correlations
    if (result.correlations?.length) {
      setCorrelations((prev) => [...prev, ...result.correlations])
    }
  }, [])

  return (
    <div className="flex h-screen bg-gray-950">
      {/* Sidebar */}
      <Sidebar
        images={images}
        activeTab={activeTab}
        onTabChange={setActiveTab}
      />

      {/* Main Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="h-14 border-b border-gray-800 flex items-center justify-between px-6">
          <h1 className="text-lg font-semibold text-white">
            Vision RAG
            <span className="text-xs text-gray-500 ml-2 font-normal">
              SmolVLM2 + Cohere
            </span>
          </h1>
          <ImageUploader
            sessionId={sessionId}
            onImageProcessed={handleImageProcessed}
          />
        </header>

        {/* Content */}
        <main className="flex-1 overflow-hidden">
          {activeTab === 'chat' ? (
            <ChatInterface sessionId={sessionId} imageCount={images.length} />
          ) : (
            <ImageGallery images={images} correlations={correlations} />
          )}
        </main>
      </div>
    </div>
  )
}