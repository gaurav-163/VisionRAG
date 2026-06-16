import React, { useMemo } from 'react'
import Plot from 'react-plotly.js'
import { BarChart3, PieChart, Network, Image as ImageIcon } from 'lucide-react'

export default function AnalysisDiagram({ analysis, filename }) {
  const {
    object_counts: objectCounts = {},
    objects = [],
    attributes = {},
    relations = [],
    people_count: peopleCount = null,
    annotated_image_path: annotatedImagePath = null
  } = analysis || {}

  // Prepare data for visualizations
  const objectData = useMemo(() => {
    // Use object_counts directly if available, otherwise count from objects array
    const counts = Object.keys(objectCounts).length > 0
      ? objectCounts
      : (() => {
          const c = {}
          objects.forEach(obj => {
            c[obj] = (c[obj] || 0) + 1
          })
          return c
        })()

    if (!counts || Object.keys(counts).length === 0) return null

    const labels = Object.keys(counts)
    const values = Object.values(counts)

    return {
      labels,
      values,
      x: labels,
      y: values
    }
  }, [objectCounts, objects])

  // Prepare attributes data
  const attributesData = useMemo(() => {
    if (!attributes || Object.keys(attributes).length === 0) return null

    const keys = Object.keys(attributes).slice(0, 10)
    const vals = keys.map(k => {
      const v = attributes[k]
      return typeof v === 'object' ? JSON.stringify(v) : String(v)
    })

    return { keys, values: vals }
  }, [attributes])

  // Prepare relations data
  const relationsData = useMemo(() => {
    if (!relations || relations.length === 0) return null
    return relations.slice(0, 5)
  }, [relations])

  if (!analysis) {
    return (
      <div className="p-4 text-center text-gray-500">
        <p>No analysis data available</p>
      </div>
    )
  }

  return (
    <div className="space-y-6 p-4">
      {/* Objects Distribution */}
      {objectData && objectData.labels.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 size={16} className="text-blue-400" />
            <h3 className="text-sm font-semibold text-white">Objects Detected</h3>
            <span className="text-xs text-gray-500">({objectData.labels.length} unique)</span>
          </div>
          <Plot
            data={[
              {
                x: objectData.x,
                y: objectData.y,
                type: 'bar',
                marker: { color: '#3b82f6' },
                hovertemplate: '<b>%{x}</b><br>Count: %{y}<extra></extra>'
              }
            ]}
            layout={{
              margin: { l: 50, r: 20, t: 0, b: 40 },
              paper_bgcolor: 'rgba(17,24,39,1)',
              plot_bgcolor: 'rgba(31,41,55,1)',
              font: { color: '#9ca3af', size: 11 },
              xaxis: {
                showgrid: false,
                tickangle: -45
              },
              yaxis: {
                showgrid: true,
                gridcolor: 'rgba(55,65,81,0.5)'
              },
              height: 300
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>
      )}

      {/* Attributes */}
      {attributesData && attributesData.keys.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-4">
            <PieChart size={16} className="text-purple-400" />
            <h3 className="text-sm font-semibold text-white">Key Attributes</h3>
          </div>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {attributesData.keys.map((key, i) => (
              <div key={i} className="flex justify-between text-xs">
                <span className="text-gray-400 font-medium">{key}</span>
                <span className="text-gray-500 truncate max-w-xs ml-2">
                  {attributesData.values[i]}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Relations Network */}
      {relationsData && relationsData.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-4">
            <Network size={16} className="text-green-400" />
            <h3 className="text-sm font-semibold text-white">Object Relations</h3>
          </div>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {relationsData.map((relation, i) => (
              <div
                key={i}
                className="text-xs text-gray-400 p-2 bg-gray-800 rounded-md"
              >
                {relation}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Summary Stats */}
      <div className="grid grid-cols-4 gap-3">
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-3 text-center">
          <p className="text-xs text-gray-500">Objects</p>
          <p className="text-lg font-bold text-blue-400">{Object.keys(objectCounts).length}</p>
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-3 text-center">
          <p className="text-xs text-gray-500">Attributes</p>
          <p className="text-lg font-bold text-purple-400">
            {Object.keys(attributes || {}).length}
          </p>
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-3 text-center">
          <p className="text-xs text-gray-500">People</p>
          <p className="text-lg font-bold text-orange-400">
            {peopleCount === null ? 'N/A' : peopleCount}
          </p>
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-3 text-center">
          <p className="text-xs text-gray-500">Relations</p>
          <p className="text-lg font-bold text-green-400">{relations.length}</p>
        </div>
      </div>

      {/* Annotated Image with Bounding Boxes */}
      {annotatedImagePath && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-4">
            <ImageIcon size={16} className="text-cyan-400" />
            <h3 className="text-sm font-semibold text-white">Detected Objects Visualization</h3>
          </div>
          <div className="rounded-lg overflow-hidden border border-gray-700">
            <img
              src={annotatedImagePath}
              alt="Annotated detection results"
              className="w-full h-auto bg-gray-800"
            />
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Colored bounding boxes show detected objects with confidence scores
          </p>
        </div>
      )}
    </div>
  )
}
