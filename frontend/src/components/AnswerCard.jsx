import { useEffect, useState } from 'react'
import { Sparkles, MessageSquare, CheckCircle2, ShieldAlert } from 'lucide-react'

const STAGES = [
  { key: 'retrieving', icon: '🔍', label: 'Searching knowledge base (FAISS)', color: 'text-brand-purple' },
  { key: 'generating', icon: '🧠', label: 'Generating with LLM harness', color: 'text-brand-amber' },
  { key: 'done',       icon: '✅', label: 'Complete (<200ms target)',        color: 'text-success' },
]

export default function AnswerCard({ result, loading, stage, query }) {
  const [displayed, setDisplayed] = useState('')
  const [showCursor, setShowCursor] = useState(false)

  useEffect(() => {
    if (!result?.answer) { setDisplayed(''); setShowCursor(false); return }
    setDisplayed('')
    setShowCursor(true)
    let i = 0
    const text = result.answer
    const interval = setInterval(() => {
      i += 2
      setDisplayed(text.slice(0, i))
      if (i >= text.length) { clearInterval(interval); setShowCursor(false) }
    }, 12)
    return () => clearInterval(interval)
  }, [result?.answer])

  const currentStage = STAGES.find(s => s.key === stage)

  return (
    <div className="glass-card overflow-hidden h-full flex flex-col">
      <div className="h-1 bg-gradient-to-r from-brand-purple to-success opacity-80" />
      
      <div className="p-5 flex items-center justify-between border-b border-white/5">
        <div className="flex items-center gap-2 text-sm text-gray-400 font-medium">
          <MessageSquare size={16} className="text-brand-purple" />
          Answer & Guardrails
        </div>
        
        {currentStage && (
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-xs font-medium">
            <span className={`w-2 h-2 rounded-full ${currentStage.color === 'text-success' ? 'bg-success' : 'bg-brand-purple animate-pulse'}`} />
            <span className={currentStage.color}>{currentStage.label}</span>
          </div>
        )}
      </div>

      <div className="p-6 flex-1 flex flex-col gap-6">
        {loading && !result && (
          <div className="space-y-4 w-full animate-pulse">
            <div className="h-4 bg-white/5 rounded w-3/4"></div>
            <div className="h-4 bg-white/5 rounded w-1/2"></div>
            <div className="h-4 bg-white/5 rounded w-5/6"></div>
          </div>
        )}

        {result && (
          <>
            <div className="flex gap-4 p-4 rounded-xl bg-white/5 border border-white/5">
              <div className="w-8 h-8 rounded-full bg-white/10 flex items-center justify-center shrink-0">
                <span className="text-sm">👤</span>
              </div>
              <p className="text-gray-300 text-sm pt-1 leading-relaxed">{result.query}</p>
            </div>

            {result.off_topic && (
              <div className="flex gap-3 p-3 rounded-lg bg-brand-amber/10 border border-brand-amber/20 text-brand-amber text-sm items-start">
                <ShieldAlert size={18} className="shrink-0 mt-0.5" />
                <p><strong>Guardrail triggered</strong> — Query is off-topic, low confidence, or filtered by safety policies.</p>
              </div>
            )}

            <div className="flex gap-4">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-purple to-brand-amber flex items-center justify-center shrink-0 shadow-lg shadow-brand-purple/20">
                <Sparkles size={16} className="text-white" />
              </div>
              <div className="text-gray-200 text-sm leading-relaxed pt-1 whitespace-pre-wrap">
                {displayed}
                {showCursor && <span className="inline-block w-1.5 h-4 ml-1 align-middle bg-brand-purple animate-pulse" />}
              </div>
            </div>
          </>
        )}
      </div>

      {result && (
        <div className="p-4 bg-white/5 border-t border-white/5 flex flex-wrap gap-2">
          <span className={`px-2.5 py-1 rounded-md text-xs font-medium border ${result.metrics?.total_e2e_ms <= 200 ? 'bg-success/10 text-success border-success/20' : 'bg-brand-amber/10 text-brand-amber border-brand-amber/20'}`}>
            ⚡ {result.metrics?.total_e2e_ms} ms {result.metrics?.total_e2e_ms <= 200 ? '(Target Met)' : ''}
          </span>
          <span className={`px-2.5 py-1 rounded-md text-xs font-medium border ${result.metrics?.best_sim_score >= 0.35 ? 'bg-brand-purple/10 text-brand-purple border-brand-purple/20' : 'bg-red-500/10 text-red-400 border-red-500/20'}`}>
            🎯 {result.metrics?.best_sim_score} similarity
          </span>
          {result.grounded !== undefined && (
            <span className={`px-2.5 py-1 rounded-md text-xs font-medium border flex items-center gap-1 ${result.grounded ? 'bg-success/10 text-success border-success/20' : 'bg-red-500/10 text-red-400 border-red-500/20'}`}>
              <CheckCircle2 size={12} /> {result.grounded ? 'Grounded (No Hallucination)' : 'Ungrounded'}
            </span>
          )}
        </div>
      )}
    </div>
  )
}
