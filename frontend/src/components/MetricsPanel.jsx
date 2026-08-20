import { useEffect, useState } from 'react'
import { Activity, LayoutGrid } from 'lucide-react'

const METRICS = [
  { key: 'stt_ms',        label: 'STT (Sarvam)',    icon: '🎙', color: 'text-blue-400',  bg: 'bg-blue-400', max: 500 },
  { key: 'retrieval_ms',  label: 'Retrieval',       icon: '🔍', color: 'text-brand-purple', bg: 'bg-brand-purple', max: 100 },
  { key: 'llm_ttft_ms',   label: 'First Token',     icon: '⚡', color: 'text-brand-amber', bg: 'bg-brand-amber', max: 150 },
  { key: 'llm_total_ms',  label: 'LLM Gen (Groq)',  icon: '🧠', color: 'text-pink-400', bg: 'bg-pink-400', max: 200 },
]

function MetricCard({ config, value }) {
  const [barW, setBarW] = useState(0)
  const pct = value !== undefined ? Math.min((value / config.max) * 100, 100) : 0

  useEffect(() => {
    if (value === undefined) return
    const t = setTimeout(() => setBarW(pct), 100)
    return () => clearTimeout(t)
  }, [pct, value])

  return (
    <div className="flex flex-col gap-2 p-3 rounded-xl bg-white/5 border border-white/5 hover:border-white/10 transition-colors">
      <div className="flex items-center gap-2">
        <span className="text-sm opacity-80">{config.icon}</span>
        <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">{config.label}</span>
      </div>
      <div className={`text-lg font-mono font-bold ${config.color}`}>
        {value !== undefined ? `${value}ms` : '—'}
      </div>
      <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden mt-1">
        <div 
          className={`h-full ${config.bg} rounded-full transition-all duration-1000 ease-out`}
          style={{ width: `${barW}%` }}
        />
      </div>
    </div>
  )
}

export default function MetricsPanel({ metrics, chunkSources, loading }) {
  const [benchmark, setBenchmark] = useState(null)
  const [benchmarking, setBenchmarking] = useState(false)

  const fetchBenchmark = async () => {
    setBenchmarking(true)
    try {
      const res = await fetch('/api/benchmark')
      if (res.ok) {
        const data = await res.json()
        setBenchmark(data)
      }
    } catch (e) {
      console.error(e)
    } finally {
      setBenchmarking(false)
    }
  }

  return (
    <div className="glass-card overflow-hidden">
      <div className="p-5 flex items-center justify-between border-b border-white/5">
        <div className="flex items-center gap-2 text-sm text-gray-400 font-medium">
          <Activity size={16} className="text-success" />
          Pipeline Trace & Latency
        </div>
        <button
          onClick={fetchBenchmark}
          disabled={benchmarking}
          className="text-xs px-3 py-1.5 rounded-lg bg-brand-purple/10 text-brand-purple border border-brand-purple/20 hover:bg-brand-purple/20 transition-colors font-medium"
        >
          {benchmarking ? 'Running...' : 'Run Benchmark'}
        </button>
      </div>

      <div className="p-5 flex flex-col gap-5">
        {loading && !metrics && (
          <div className="grid grid-cols-2 gap-4 animate-pulse">
            {[1,2,3,4].map(i => (
              <div key={i} className="h-24 bg-white/5 rounded-xl"></div>
            ))}
          </div>
        )}

        {metrics && (
          <>
            <div className="grid grid-cols-2 gap-4">
              {METRICS.map(cfg => {
                if (cfg.key === 'stt_ms' && !metrics.stt_ms) return null
                return <MetricCard key={cfg.key} config={cfg} value={metrics[cfg.key]} />
              })}
            </div>

            {chunkSources && chunkSources.length > 0 && (
              <div className="mt-2 p-3 rounded-xl bg-white/5 border border-white/5">
                <div className="flex items-center gap-2 mb-2 text-xs text-gray-400 font-medium">
                  <LayoutGrid size={14} /> Chunking Strategies Used
                </div>
                <div className="flex flex-wrap gap-2">
                  {Array.from(new Set(chunkSources.map(s => s.chunk_strategy))).map(strat => (
                    <span key={strat} className="text-xs px-2 py-1 rounded bg-brand-purple/20 text-brand-purple-light border border-brand-purple/30">
                      {strat}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
