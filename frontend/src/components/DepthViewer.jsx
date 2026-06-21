import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { Layers, Eye, Download } from 'lucide-react'

/**
 * DepthViewer — shows:
 *  1. Coloured INFERNO depth map
 *  2. Per-object depth table (foreground / mid / background)
 *  3. Interactive Three.js 3D parallax viewer
 */
export default function DepthViewer({ imageId, filename, objectDepths, depthMapPath, depthMapGrayPath }) {
  const [activeTab, setActiveTab] = useState('map')   // 'map' | '3d' | 'objects'
  const canvasRef = useRef(null)
  const rendererRef = useRef(null)
  const animFrameRef = useRef(null)

  // Use relative paths — these go through Vite's dev server proxy
  // (see vite.config.js: '/uploads' → http://localhost:8000), which keeps
  // everything same-origin. Hardcoding http://localhost:8000 here breaks
  // WebGL texture loading because FastAPI's StaticFiles doesn't send
  // Access-Control-Allow-Origin headers, so the browser blocks the image.
  const depthColorUrl = depthMapPath || null
  const depthGrayUrl = depthMapGrayPath || null
  const originalUrl = imageId
    ? `/uploads/${imageId}${getExt(filename)}`
    : null

  // ── Three.js setup ──────────────────────────────────────────────────────
  useEffect(() => {
    if (activeTab !== '3d' || !canvasRef.current || !originalUrl || !depthGrayUrl) return

    const canvas = canvasRef.current
    const W = canvas.clientWidth || 600
    const H = canvas.clientHeight || 400

    // Scene
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x111827)

    // Camera
    const camera = new THREE.PerspectiveCamera(60, W / H, 0.1, 100)
    camera.position.set(0, 0, 1.5)

    // Renderer
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true })
    renderer.setSize(W, H)
    renderer.setPixelRatio(window.devicePixelRatio)
    rendererRef.current = renderer

    // Plane geometry with enough segments for smooth displacement
    const geometry = new THREE.PlaneGeometry(2, 1.5, 128, 96)

    // Load textures, with error logging so failures are visible in console
    const loader = new THREE.TextureLoader()
    const colorTex = loader.load(
      originalUrl,
      () => { renderer.render(scene, camera) },
      undefined,
      (err) => console.error('DepthViewer: failed to load color texture', originalUrl, err)
    )
    colorTex.colorSpace = THREE.SRGBColorSpace

    const depthTex = loader.load(
      depthGrayUrl,
      () => { renderer.render(scene, camera) },
      undefined,
      (err) => console.error('DepthViewer: failed to load depth texture', depthGrayUrl, err)
    )

    const material = new THREE.MeshStandardMaterial({
      map: colorTex,
      displacementMap: depthTex,
      displacementScale: 0.3,    // positive = closer objects (higher depth value) pop toward camera
      displacementBias: -0.15,
    })

    const mesh = new THREE.Mesh(geometry, material)
    scene.add(mesh)

    // Lights
    scene.add(new THREE.AmbientLight(0xffffff, 1.5))
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8)
    dirLight.position.set(1, 1, 2)
    scene.add(dirLight)

    // Mouse-driven orbit
    let mouseX = 0
    let mouseY = 0

    const onMouseMove = (e) => {
      const rect = canvas.getBoundingClientRect()
      mouseX = ((e.clientX - rect.left) / rect.width - 0.5) * 2
      mouseY = -((e.clientY - rect.top) / rect.height - 0.5) * 2
    }
    canvas.addEventListener('mousemove', onMouseMove)

    // Animation loop
    const animate = () => {
      animFrameRef.current = requestAnimationFrame(animate)
      mesh.rotation.y = mouseX * 0.25
      mesh.rotation.x = mouseY * 0.15
      renderer.render(scene, camera)
    }
    animate()

    // Cleanup
    return () => {
      cancelAnimationFrame(animFrameRef.current)
      canvas.removeEventListener('mousemove', onMouseMove)
      renderer.dispose()
      geometry.dispose()
      material.dispose()
      colorTex.dispose()
      depthTex.dispose()
    }
  }, [activeTab, originalUrl, depthGrayUrl])

  // ── Depth label colour ──────────────────────────────────────────────────
  const labelColor = (label) => {
    if (label === 'foreground') return 'text-green-400 bg-green-400/10'
    if (label === 'mid-ground') return 'text-yellow-400 bg-yellow-400/10'
    return 'text-blue-400 bg-blue-400/10'
  }

  if (!depthMapPath) {
    return (
      <div className="p-6 text-center text-gray-500">
        <Layers size={32} className="mx-auto mb-2 text-gray-600" />
        <p className="text-sm">Depth map not available for this image.</p>
      </div>
    )
  }

  return (
    <div className="space-y-4 p-4">
      {/* Tab switcher */}
      <div className="flex gap-2 border-b border-gray-700 pb-2">
        {[
          { id: 'map', label: '🎨 Depth Map' },
          { id: 'objects', label: '📊 Objects by Depth' },
          { id: '3d', label: '🧊 3D Viewer' },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-3 py-1.5 text-xs rounded-lg transition-colors ${
              activeTab === tab.id
                ? 'bg-purple-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Depth Map Tab */}
      {activeTab === 'map' && depthColorUrl && (
        <div className="space-y-3">
          <div className="rounded-xl overflow-hidden border border-gray-700">
            <img
              src={depthColorUrl}
              alt="INFERNO depth map"
              className="w-full h-auto"
            />
          </div>
          <div className="flex items-center justify-between text-xs text-gray-500">
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1">
                <span className="w-3 h-3 rounded-full bg-yellow-300 inline-block" />
                Closest
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-3 rounded-full bg-purple-700 inline-block" />
                Farthest
              </span>
            </div>
            <a
              href={depthColorUrl}
              download={`${imageId}_depth.png`}
              className="flex items-center gap-1 text-purple-400 hover:text-purple-300"
            >
              <Download size={12} /> Download
            </a>
          </div>
        </div>
      )}

      {/* Objects by Depth Tab */}
      {activeTab === 'objects' && (
        <div className="space-y-2">
          {objectDepths && objectDepths.length > 0 ? (
            <>
              <p className="text-xs text-gray-500 mb-3">
                Objects sorted nearest → farthest from camera
              </p>
              {objectDepths.map((obj, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between bg-gray-800 rounded-lg px-3 py-2"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-gray-600 text-xs w-4">{i + 1}</span>
                    <span className="text-white text-sm font-medium capitalize">
                      {obj.class_name}
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    {/* Depth bar */}
                    <div className="w-24 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-purple-500 rounded-full"
                        style={{ width: `${(obj.depth_score || 0) * 100}%` }}
                      />
                    </div>
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${labelColor(obj.depth_label)}`}>
                      {obj.depth_label}
                    </span>
                  </div>
                </div>
              ))}
            </>
          ) : (
            <p className="text-sm text-gray-500 text-center py-4">
              No object depth data available.
            </p>
          )}
        </div>
      )}

      {/* 3D Viewer Tab */}
      {activeTab === '3d' && (
        <div className="space-y-2">
          <p className="text-xs text-gray-500 flex items-center gap-1">
            <Eye size={12} /> Move your mouse over the viewer to orbit in 3D
          </p>
          <div className="rounded-xl overflow-hidden border border-gray-700 bg-gray-900"
               style={{ height: '360px' }}>
            <canvas
              ref={canvasRef}
              style={{ width: '100%', height: '100%', display: 'block' }}
            />
          </div>
          <p className="text-xs text-gray-600">
            Depth-Anything-V2 displacement mapping — closer objects appear raised
          </p>
        </div>
      )}
    </div>
  )
}

function getExt(filename) {
  if (!filename) return '.jpg'
  const ext = filename.split('.').pop()
  return ext ? `.${ext}` : '.jpg'
}