import { useState, useRef, useEffect } from 'react'
import { Mic, Square, ArrowUp, Zap } from 'lucide-react'

const SAMPLE_QUERIES = [
  'What is the capital of India?',
  'Where is the Taj Mahal located?',
  'What is machine learning?',
  'What is another name for India?',
]

export default function QueryInput({ query, setQuery, onSubmit, loading, useMock = false, apiBase = '' }) {
  const [listening, setListening] = useState(false)
  const [voiceError, setVoiceError] = useState(null)
  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])
  const speechRecognitionRef = useRef(null)
  const fallbackTranscriptRef = useRef('')

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition()
      recognition.continuous = true
      recognition.interimResults = true
      recognition.lang = 'en-IN'

      recognition.onresult = (event) => {
        let interim = ''
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          interim += event.results[i][0].transcript
        }
        if (interim) {
          fallbackTranscriptRef.current = interim
          setQuery(interim)
        }
      }

      recognition.onerror = (e) => {
        console.warn('SpeechRecognition notice:', e.error)
      }

      speechRecognitionRef.current = recognition
    }
  }, [setQuery])

  const startRecording = async () => {
    setVoiceError(null)
    fallbackTranscriptRef.current = ''
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
          ? 'audio/webm;codecs=opus'
          : 'audio/webm',
      })
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop())
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        await sendAudioToBackend(blob)
      }

      mediaRecorder.start(100)
      setListening(true)

      if (speechRecognitionRef.current) {
        try { speechRecognitionRef.current.start() } catch (e) {}
      }

      setTimeout(() => {
        if (mediaRecorderRef.current?.state === 'recording') stopRecording()
      }, 8000)
    } catch (err) {
      setVoiceError('Microphone access denied.')
    }
  }

  const stopRecording = () => {
    if (speechRecognitionRef.current) {
      try { speechRecognitionRef.current.stop() } catch (e) {}
    }
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop()
    }
    setListening(false)
  }

  const sendAudioToBackend = async (blob) => {
    try {
      setListening(false)
      const formData = new FormData()
      formData.append('file', blob, 'recording.webm')

      const res = await fetch(`${apiBase}/api/voice?mock=${useMock}`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        if (fallbackTranscriptRef.current && fallbackTranscriptRef.current.trim()) {
          const fallbackText = fallbackTranscriptRef.current.trim()
          console.warn('Voice failed, falling back to client transcript:', fallbackText)
          setQuery(fallbackText)
          onSubmit(fallbackText)
          return
        }
        let errMsg = 'Voice processing failed'
        try {
          const err = await res.json()
          errMsg = err.detail || 'Voice processing failed'
        } catch (e) {
          errMsg = `Voice processing failed (${res.status}): The backend is likely still starting up.`
        }
        throw new Error(errMsg)
      }

      const data = await res.json()
      setQuery(data.query)
      onSubmit(data.query, data)
    } catch (err) {
      if (fallbackTranscriptRef.current && fallbackTranscriptRef.current.trim()) {
        const fallbackText = fallbackTranscriptRef.current.trim()
        setQuery(fallbackText)
        onSubmit(fallbackText)
      } else {
        setVoiceError(err.message)
      }
    }
  }

  const handleVoiceClick = () => {
    if (listening) stopRecording()
    else startRecording()
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      onSubmit()
    }
  }

  return (
    <div className="glass-card overflow-hidden">
      <div className="h-1 bg-gradient-to-r from-success-light via-brand-purple to-brand-amber opacity-80" />
      
      <div className="p-6 flex flex-col items-center gap-6">
        <div className="flex items-center gap-2 text-sm text-gray-400 font-medium">
          <Zap size={16} className="text-brand-amber" />
          Voice Console
        </div>

        <button
          className={`relative flex items-center justify-center w-24 h-24 rounded-2xl transition-all duration-300 shadow-2xl group ${
            listening ? 'bg-red-500 hover:bg-red-600 shadow-red-500/20' : 'bg-success hover:bg-success-light shadow-success/20'
          }`}
          onClick={handleVoiceClick}
          disabled={loading}
        >
          {listening ? (
            <Square fill="currentColor" size={32} className="text-white animate-pulse" />
          ) : (
            <Mic size={36} className="text-white group-hover:scale-110 transition-transform" />
          )}
          {listening && (
            <div className="absolute -inset-4 border-2 border-red-500/30 rounded-3xl animate-ping" />
          )}
        </button>

        <p className="text-sm font-medium text-gray-400">
          {listening ? 'Listening...' : loading ? 'Processing voice...' : 'Tap to speak'}
        </p>

        {voiceError && (
          <p className="text-xs text-amber-500 bg-amber-500/10 px-3 py-1 rounded-full">
            {voiceError}
          </p>
        )}

        <div className="w-full relative">
          <textarea
            className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 pr-12 text-white placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-brand-purple/50 resize-none transition-all"
            placeholder="Or type your question here..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKey}
            disabled={loading || listening}
            rows={2}
          />
          <button
            onClick={() => onSubmit()}
            disabled={loading || !query.trim()}
            className="absolute right-3 bottom-3 p-2 bg-brand-purple hover:bg-brand-purple/80 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ArrowUp size={18} strokeWidth={3} />
          </button>
        </div>

        <div className="w-full">
          <p className="text-xs text-gray-500 font-medium mb-3">Try asking:</p>
          <div className="flex flex-wrap gap-2">
            {SAMPLE_QUERIES.map(q => (
              <button
                key={q}
                onClick={() => { setQuery(q); onSubmit(q) }}
                disabled={loading}
                className="text-xs px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-gray-300 hover:bg-success/10 hover:border-success/30 hover:text-success transition-all text-left"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
