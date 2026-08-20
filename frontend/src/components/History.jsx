import { History as HistoryIcon, Clock } from 'lucide-react'

export default function History({ history, onSelect }) {
  return (
    <div className="glass-card overflow-hidden">
      <div className="p-4 flex items-center justify-between border-b border-white/5">
        <div className="flex items-center gap-2 text-sm text-gray-400 font-medium">
          <HistoryIcon size={16} className="text-gray-400" />
          Recent Queries
        </div>
        {history.length > 0 && (
          <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-white/10 text-gray-300">
            {history.length}
          </span>
        )}
      </div>

      <div className="p-2 max-h-[300px] overflow-y-auto">
        {history.length === 0 ? (
          <div className="p-8 flex flex-col items-center justify-center text-gray-500">
            <Clock size={32} className="mb-2 opacity-50" />
            <p className="text-sm">No queries yet</p>
          </div>
        ) : (
          <div className="flex flex-col gap-1">
            {history.map((item, i) => (
              <button
                key={item.ts}
                className="w-full text-left p-3 rounded-lg hover:bg-white/5 border border-transparent hover:border-white/10 transition-all group"
                onClick={() => onSelect(item.query)}
              >
                <div className="flex gap-2 items-start mb-1">
                  <span className="text-sm mt-0.5 opacity-50 group-hover:opacity-100 transition-opacity">👤</span>
                  <p className="text-sm font-medium text-gray-300 group-hover:text-white transition-colors line-clamp-2 leading-tight">
                    {item.query}
                  </p>
                </div>
                <p className="text-xs text-gray-500 line-clamp-1 pl-6 mb-2">
                  {item.answer?.replace(/\n/g, ' ')}
                </p>
                <div className="flex items-center justify-between pl-6">
                  <span className="text-xs text-brand-purple font-medium bg-brand-purple/10 px-1.5 py-0.5 rounded">
                    ⚡ {item.metrics?.total_e2e_ms || '?'}ms
                  </span>
                  <span className="text-[10px] text-gray-600">
                    {new Date(item.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </span>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
