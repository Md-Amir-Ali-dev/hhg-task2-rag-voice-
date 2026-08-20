import { useState, useEffect } from 'react'
import QueryInput from './components/QueryInput'
import AnswerCard from './components/AnswerCard'
import MetricsPanel from './components/MetricsPanel'
import Header from './components/Header'
import History from './components/History'

// In production (Vercel), set VITE_API_BASE_URL to your Railway backend URL.
// Locally, the Vite proxy handles /api → localhost:8000 so this stays empty.
const API_BASE = import.meta.env.VITE_API_BASE_URL || ''


export default function App() {
  const [query, setQuery] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [useMock, setUseMock] = useState(false)
  const [error, setError] = useState(null)
  const [stage, setStage] = useState(null)
  const [history, setHistory] = useState([])
  const [mounted, setMounted] = useState(false)

  useEffect(() => { setMounted(true) }, [])

  const handleSubmit = async (q, prefetchedResult = null) => {
    const queryText = (typeof q === 'string' && q.trim()) ? q.trim() : query.trim()
    if (!queryText) return

    setLoading(true)
    setResult(null)
    setError(null)
    setStage('retrieving')

    if (prefetchedResult) {
      setResult(prefetchedResult)
      setStage('done')
      setHistory(h => [
        { query: queryText, answer: prefetchedResult.answer, metrics: prefetchedResult.metrics, ts: Date.now() },
        ...h.slice(0, 9),
      ])
      setLoading(false)
      return
    }

    try {
      setTimeout(() => setStage('generating'), 400)

      const res = await fetch(`${API_BASE}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: queryText, mock: useMock }),
      })
      if (!res.ok) {
        let errMsg = 'Server error'
        try {
          const err = await res.json()
          errMsg = err.detail || 'Server error'
        } catch (e) {
          errMsg = `Server error (${res.status}): The backend is likely still starting up.`
        }
        throw new Error(errMsg)
      }
      const data = await res.json()
      setResult(data)
      setStage('done')
      setHistory(h => [
        { query: queryText, answer: data.answer, metrics: data.metrics, ts: Date.now() },
        ...h.slice(0, 9),
      ])
    } catch (e) {
      setError(e.message)
      setStage(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={`min-h-screen relative overflow-hidden transition-opacity duration-500 ${mounted ? 'opacity-100' : 'opacity-0'}`}>
      <div className="grid-bg" />
      <div className="glow-effect" />

      <div className="max-w-7xl mx-auto px-4 py-8 relative z-10 flex flex-col gap-10">
        <Header useMock={useMock} onToggleMock={() => setUseMock(m => !m)} />
        
        {/* Hero Section */}
        <div className="text-center space-y-4 py-6">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-xs font-medium text-gray-300">
            <span className="w-2 h-2 rounded-full bg-success"></span>
            All Systems Operational
          </div>
          <h1 className="text-5xl font-extrabold tracking-tight text-white">
            Speak. <span className="text-gray-400">We Retrieve.</span> We Ground. <span className="text-brand-purple">We Answer.</span>
          </h1>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Experience ultra-low latency Voice RAG using Sarvam Saaras v3 and Llama 3.1 70B, combined with real-time vector search.
          </p>
        </div>

        {error && (
          <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 flex items-center gap-3">
            <span>⚠</span> {error}
          </div>
        )}

        {/* Main Grid Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* Left Column (Voice Console & History) */}
          <div className="lg:col-span-5 flex flex-col gap-6">
            <QueryInput
              query={query}
              setQuery={setQuery}
              onSubmit={handleSubmit}
              loading={loading}
              useMock={useMock}
              apiBase={API_BASE}
            />
            <History history={history} onSelect={(q) => { setQuery(q); handleSubmit(q) }} />
          </div>

          {/* Right Column (Answer & Metrics) */}
          <div className="lg:col-span-7 flex flex-col gap-6">
            <AnswerCard result={result} loading={loading} stage={stage} query={query} />
            {(loading || result) && (
              <MetricsPanel metrics={result?.metrics} chunkSources={result?.chunk_sources} loading={loading} />
            )}
            
            {!result && !loading && (
              <div className="glass-card flex flex-col items-center justify-center p-12 text-center h-[300px]">
                <div className="w-16 h-16 rounded-2xl bg-white/5 flex items-center justify-center mb-4 border border-white/10">
                  <span className="text-2xl opacity-50">🎙</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-200">Waiting for query...</h3>
                <p className="text-gray-500 mt-2">Use the voice console to ask a question</p>
              </div>
            )}
          </div>
          
        </div>
      </div>
    </div>
  )
}
