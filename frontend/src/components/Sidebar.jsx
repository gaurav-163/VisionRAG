import React from 'react'
import { MessageSquare, Images, Eye } from 'lucide-react'

export default function Sidebar({ images, activeTab, onTabChange }) {
  const tabs = [
    { id: 'chat', label: 'Chat', icon: MessageSquare },
    { id: 'gallery', label: 'Gallery', icon: Images },
  ]

  return (
    <aside className="w-64 border-r border-gray-800 flex flex-col bg-gray-900/50">
      {/* Logo */}
      <div className="h-14 border-b border-gray-800 flex items-center px-4 gap-3">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center">
          <Eye size={18} className="text-white" />
        </div>
        <span className="font-bold text-white">VisionRAG</span>
      </div>

      {/* Tabs */}
      <nav className="p-3 space-y-1">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => onTabChange(id)}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors
              ${activeTab === id
                ? 'bg-gray-800 text-white'
                : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
              }`}
          >
            <Icon size={18} />
            {label}
            {id === 'gallery' && images.length > 0 && (
              <span className="ml-auto text-xs bg-gray-700 px-2 py-0.5 rounded-full">
                {images.length}
              </span>
            )}
          </button>
        ))}
      </nav>

      {/* Image List */}
      <div className="flex-1 overflow-y-auto scrollbar-thin px-3 mt-2">
        <p className="text-xs text-gray-500 uppercase tracking-wider mb-2 px-1">
          Uploaded Images
        </p>
        {images.length === 0 ? (
          <p className="text-xs text-gray-600 px-1">No images yet</p>
        ) : (
          <div className="space-y-1">
            {images.map((img) => (
              <div
                key={img.id}
                className="flex items-center gap-2 px-2 py-2 rounded-md hover:bg-gray-800/50 text-sm"
              >
                <div className="w-2 h-2 rounded-full bg-green-500 flex-shrink-0" />
                <span className="truncate text-gray-300">{img.filename}</span>
                <span className="text-xs text-gray-600 flex-shrink-0">
                  {img.analysis?.objects?.length || 0} obj
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-gray-800">
        <p className="text-xs text-gray-600 text-center">
          {images.length} images • {images.reduce((sum, img) => sum + (img.correlations?.length || 0), 0)} correlations
        </p>
      </div>
    </aside>
  )
}