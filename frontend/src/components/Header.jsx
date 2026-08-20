import { Code, Activity, Waves } from 'lucide-react'

export default function Header({ useMock, onToggleMock }) {
  return (
    <header className="flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className="w-11 h-11 rounded-2xl bg-gradient-to-br from-brand-amber via-orange-400 to-brand-purple flex items-center justify-center shadow-lg shadow-brand-amber/30 border border-white/20">
          <Waves size={24} className="text-white drop-shadow-md" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-white tracking-tight">Voice Rag App</h1>
          <p className="text-xs text-gray-400 font-medium">Research Preview v0.1</p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <button 
          onClick={onToggleMock}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            useMock ? 'bg-amber-500/10 text-amber-500 border border-amber-500/20 hover:bg-amber-500/20' 
                   : 'bg-white/5 text-gray-300 border border-white/10 hover:bg-white/10'
          }`}
        >
          <Activity size={16} />
          {useMock ? 'Mocking Backend' : 'Live Mode'}
        </button>

        <a href="#" className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-white/5 text-gray-300 border border-white/10 hover:bg-white/10 transition-colors">
          <Code size={16} />
          View Source
        </a>
      </div>
    </header>
  )
}
