import React, { useState } from 'react'
import {
  Link2,
  Tag,
  FileText,
  Network,
  Maximize2,
  Download,
  X,
  Image as ImageIcon,
} from 'lucide-react'

export default function ImageGallery({ images, correlations }) {
  const [viewer, setViewer] = useState(null)

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
    <div className="relative h-full overflow-y-auto scrollbar-thin p-6">
      <div className="mb-5 flex items-center justify-between gap-4 rounded-2xl border border-gray-800 bg-gradient-to-r from-gray-950 to-gray-900 px-4 py-3 shadow-lg shadow-black/20">
        <div>
          <h2 className="text-base font-semibold text-white">Uploaded Images</h2>
          <p className="text-xs text-gray-400">
            Each upload now includes analysis, correlations, and a generated scene graph preview.
          </p>
        </div>
        <div className="rounded-full border border-gray-700 bg-gray-900 px-3 py-1 text-xs text-gray-400">
          {images.length} image{images.length !== 1 ? 's' : ''}
        </div>
      </div>

      {/* Image Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 mb-8">
        {images.map((img) => {
          const imageId = img.id || img.image_id
          const filename = img.filename || 'image.jpg'
          const imageSrc = imageId ? `/uploads/${imageId}${getExtension(filename)}` : ''

          return (
          <div
            key={imageId || filename}
            className="group bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden hover:border-purple-500/30 transition-all duration-200 shadow-lg shadow-black/20"
          >
            {/* Image preview */}
            <div className="relative h-48 bg-gray-800 flex items-center justify-center overflow-hidden">
              {imageSrc ? (
                <img
                  src={imageSrc}
                  alt={filename}
                  className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
                  onError={(e) => {
                    e.target.style.display = 'none'
                    e.target.parentElement.innerHTML = `
                      <div class="flex flex-col items-center text-gray-600">
                        <span class="text-3xl mb-1">🖼️</span>
                        <span class="text-xs">${filename}</span>
                      </div>
                    `
                  }}
                />
              ) : (
                <div className="flex flex-col items-center text-gray-600">
                  <span className="text-3xl mb-1">🖼️</span>
                  <span className="text-xs">{filename}</span>
                </div>
              )}
            </div>

            {/* Info */}
            <div className="p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium text-white truncate">
                  {filename}
                </h3>
                <span className="text-xs text-gray-500 font-mono">{imageId}</span>
              </div>

              {/* Description */}
              {img.analysis?.description && (
                <div className="flex gap-2 text-xs">
                  <FileText size={12} className="text-gray-500 flex-shrink-0 mt-0.5" />
                  <p className="text-gray-400 line-clamp-3">{img.analysis.description}</p>
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

              {/* Scene graph preview */}
              {img.sceneGraphUrl && (
                <div className="space-y-3 pt-3 border-t border-gray-800/80">
                  <div className="flex items-center justify-between gap-2 text-xs text-gray-400">
                    <div className="flex items-center gap-2">
                      <Network size={12} className="text-purple-400" />
                      <span>
                        Scene graph{img.sceneGraphNodeCount ? ` (${img.sceneGraphNodeCount} nodes)` : ''}
                      </span>
                    </div>
                    <span className="rounded-full border border-gray-700 bg-gray-950 px-2 py-0.5 text-[10px] uppercase tracking-wider text-gray-500">
                      Generated
                    </span>
                  </div>

                  <button
                    type="button"
                    onClick={() => setViewer({
                      src: img.sceneGraphUrl,
                      title: filename,
                      nodes: img.sceneGraphNodeCount || 0,
                    })}
                    className="group/preview block w-full overflow-hidden rounded-xl border border-gray-800 bg-white/5 hover:border-purple-500/40 transition-all"
                    title="Open scene graph preview"
                  >
                    <div className="relative bg-white">
                      <img
                        src={img.sceneGraphUrl}
                        alt={`Scene graph for ${filename}`}
                        className="w-full h-44 object-contain bg-white"
                      />
                      <div className="absolute inset-0 flex items-end justify-between p-3 opacity-0 transition-opacity group-hover/preview:opacity-100 bg-gradient-to-t from-black/70 via-black/5 to-transparent">
                        <span className="text-xs text-white/90 font-medium flex items-center gap-2">
                          <Maximize2 size={12} />
                          Open larger view
                        </span>
                        <span className="text-[10px] text-white/70">
                          Tap to zoom
                        </span>
                      </div>
                    </div>
                  </button>

                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setViewer({
                        src: img.sceneGraphUrl,
                        title: filename,
                        nodes: img.sceneGraphNodeCount || 0,
                      })}
                      className="flex-1 inline-flex items-center justify-center gap-2 rounded-lg border border-gray-800 bg-gray-950 px-3 py-2 text-xs text-gray-300 hover:border-purple-500/40 hover:text-white transition-colors"
                    >
                      <Maximize2 size={12} />
                      Zoom
                    </button>
                    <a
                      href={img.sceneGraphUrl}
                      download={`${filename}-scene-graph.png`}
                      className="inline-flex items-center justify-center gap-2 rounded-lg border border-gray-800 bg-gray-950 px-3 py-2 text-xs text-gray-300 hover:border-purple-500/40 hover:text-white transition-colors"
                    >
                      <Download size={12} />
                      Download
                    </a>
                  </div>
                </div>
              )}
            </div>
          </div>
        )})}
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

      {/* Scene graph viewer modal */}
      {viewer && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 px-4 py-6 backdrop-blur-sm">
          <div className="relative w-full max-w-6xl overflow-hidden rounded-3xl border border-gray-700 bg-gray-950 shadow-2xl shadow-black/50">
            <div className="flex items-center justify-between gap-3 border-b border-gray-800 px-5 py-4">
              <div>
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                  <ImageIcon size={18} className="text-purple-400" />
                  {viewer.title}
                </h3>
                <p className="text-xs text-gray-400">
                  Scene graph preview{viewer.nodes ? ` • ${viewer.nodes} nodes` : ''}
                </p>
              </div>
              <button
                type="button"
                onClick={() => setViewer(null)}
                className="rounded-full border border-gray-700 bg-gray-900 p-2 text-gray-300 hover:text-white hover:border-gray-500 transition-colors"
                aria-label="Close scene graph preview"
              >
                <X size={16} />
              </button>
            </div>

            <div className="bg-white p-4">
              <img
                src={viewer.src}
                alt={`${viewer.title} scene graph preview`}
                className="mx-auto max-h-[75vh] w-full object-contain"
              />
            </div>
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