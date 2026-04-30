import React, { useState, useRef } from 'react'
import { Upload, Loader2, CheckCircle, AlertCircle } from 'lucide-react'
import { uploadImage } from '../services/api'

export default function ImageUploader({ sessionId, onImageProcessed }) {
  const [status, setStatus] = useState('idle') // idle | uploading | analyzing | done | error
  const [progress, setProgress] = useState(0)
  const [errorMsg, setErrorMsg] = useState('')
  const fileInputRef = useRef(null)

  const handleFiles = async (files) => {
    for (const file of files) {
      if (!file.type.startsWith('image/')) continue

      setStatus('uploading')
      setProgress(0)
      setErrorMsg('')

      try {
        const result = await uploadImage(file, sessionId, (pct) => {
          setProgress(pct)
          if (pct >= 100) setStatus('analyzing')
        })

        setStatus('done')
        onImageProcessed(result)

        // Reset after 2s
        setTimeout(() => setStatus('idle'), 2000)
      } catch (err) {
        setStatus('error')
        setErrorMsg(err.response?.data?.detail || 'Upload failed')
        setTimeout(() => setStatus('idle'), 3000)
      }
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    handleFiles(e.dataTransfer.files)
  }

  const statusConfig = {
    idle: { icon: Upload, text: 'Upload Image', color: 'text-gray-400' },
    uploading: { icon: Loader2, text: `Uploading ${progress}%`, color: 'text-blue-400' },
    analyzing: { icon: Loader2, text: 'Analyzing...', color: 'text-purple-400' },
    done: { icon: CheckCircle, text: 'Done!', color: 'text-green-400' },
    error: { icon: AlertCircle, text: errorMsg, color: 'text-red-400' },
  }

  const { icon: Icon, text, color } = statusConfig[status]
  const isProcessing = status === 'uploading' || status === 'analyzing'

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
        disabled={isProcessing}
      />
      <button
        onClick={() => fileInputRef.current?.click()}
        disabled={isProcessing}
        className={`flex items-center gap-2 px-4 py-2 rounded-lg border border-gray-700 
          hover:border-gray-500 transition-colors text-sm ${color}
          ${isProcessing ? 'opacity-70 cursor-wait' : 'cursor-pointer hover:bg-gray-800'}`}
      >
        <Icon size={16} className={isProcessing ? 'animate-spin' : ''} />
        <span className="max-w-[200px] truncate">{text}</span>
      </button>
    </div>
  )
}