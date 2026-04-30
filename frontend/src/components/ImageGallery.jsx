import React from 'react'
import { Link2, Tag, FileText } from 'lucide-react'

export default function ImageGallery({ images, correlations }) {
  if (images.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        <div className="text-center">
          <p className="text-lg mb-2">No images uploaded yet</p>
          <p className="text-sm">Upload images to see analysis and correlations</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto scrollbar-thin p-6">
      {/* Image Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 mb-8">
        {images.map((img) => (
          <div
            key={img.id}
            className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden hover:border-gray-700 transition-colors"
          >
            {/* Image preview */}
            <div className="h-48 bg-gray-800 flex items-center justify-center overflow-hidden">
              <img
                src={`/uploads/${img.id}${getExtension(img.filename)}`}
                alt={img.filename}
                className="w-full h-full object-cover"
                onError={(e) => {
                  e.target.style.display = 'none'
                  e.target.parentElement.innerHTML = `
                    <div class="flex flex-col items-center text-gray-600">
                      <span class="text-3xl mb-1">🖼️</span>
                      <span class="text-xs">${img.filename}</span>
                    </div>
                  `
                }}
              />
            </div>

            {/* Info */}
            <div className="p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium text-white truncate">
                  {img.filename}
                </h3>
                <span className="text-xs text-gray-500 font-mono">{img.id}</span>
              </div>

              {/* Description */}
              {img.analysis?.description && (
                <div className="flex gap-2 text-xs">
                  <FileText size={12} className="text-gray-500 flex-shrink-0 mt-0.5" />
                  <p className="text-gray-400 line-clamp-3">
                    {img.analysis.description}
                  </p>
                </div>
              )}

              {/* Objects */}
              {img.analysis?.objects?.length > 0 && (
                <div className="flex gap-2 items-start">
                  <Tag size={12} className="text-gray-500 flex-shrink-0 mt-1" />
                  <div className="flex flex-wrap gap-1">
                    {img.analysis.objects.slice(0, 8).map((obj, i) => (
                      <span
                        key={i}
                        className="px-2 py-0.5 bg-gray-800 text-gray-400 text-xs rounded-md"
                      >
                        {obj}
                      </span>
                    ))}
                    {img.analysis.objects.length > 8 && (
                      <span className="text-xs text-gray-600">
                        +{img.analysis.objects.length - 8} more
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Correlations */}
      {correlations.length > 0 && (
        <div>
          <h2 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <Link2 size={16} className="text-purple-400" />
            Correlations ({correlations.length})
          </h2>
          <div className="space-y-2">
            {correlations.map((corr, i) => (
              <div
                key={i}
                className="bg-gray-900 border border-gray-800 rounded-lg p-3 flex items-center gap-4 text-sm"
              >
                <div className="flex items-center gap-2 text-gray-300">
                  <span className="font-mono text-xs bg-gray-800 px-2 py-0.5 rounded">
                    {corr.image_id_a}
                  </span>
                  <span className="text-gray-600">↔</span>
                  <span className="font-mono text-xs bg-gray-800 px-2 py-0.5 rounded">
                    {corr.image_id_b}
                  </span>
                </div>

                <div className="flex-1 text-xs text-gray-500 truncate">
                  {corr.shared_objects?.join(', ') || 'No shared objects'}
                </div>

                <div className="flex items-center gap-1 flex-shrink-0">
                  <div className="w-16 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-purple-500 rounded-full"
                      style={{ width: `${(corr.similarity_score || 0) * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-500 w-8 text-right">
                    {((corr.similarity_score || 0) * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function getExtension(filename) {
  if (!filename) return '.jpg'
  const ext = filename.split('.').pop()
  return ext ? `.${ext}` : '.jpg'
}