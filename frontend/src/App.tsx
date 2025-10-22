import React, { useEffect, useState } from 'react'
import axios from 'axios'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

export default function App() {
  const [theme, setTheme] = useState<'light' | 'dark'>(() => (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'))
  const [input, setInput] = useState('Explain Binary technique with examples.')
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Apply theme to document
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  const ask = async () => {
    if (!input.trim()) return
    setError(null)
    setLoading(true)
    const userMsg: Message = { role: 'user', content: input }
    setMessages(prev => [...prev, userMsg])
    setInput('')

    try {
      const res = await axios.post('http://localhost:8000/ask', {
        user_id: 'Charan',
        question: userMsg.content,
      })
      const answer: string = res.data?.answer ?? 'No answer.'
      setMessages(prev => [...prev, { role: 'assistant', content: answer }])
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  const onKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      ask()
    }
  }

  return (
    <div className="container">
      <header>
        <h1>DSA-Sensei</h1>
        <div className="controls">
          <button className="toggle" onClick={() => setTheme(prev => (prev === 'dark' ? 'light' : 'dark'))}>
            {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
          </button>
        </div>
      </header>

      <div className="chat">
        {messages.map((m, i) => (
          <div key={i} className={`bubble ${m.role}`}>
            <div className="role">{m.role === 'user' ? 'You' : 'DSA-Sensei'}</div>
            <div className="content">{m.content}</div>
          </div>
        ))}
      </div>

      {error && <div className="error">{error}</div>}

      <div className="composer">
        <input
          placeholder="Ask a DSA question..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={onKey}
          disabled={loading}
        />
        <button onClick={ask} disabled={loading}>
          {loading ? 'Asking…' : 'Ask'}
        </button>
      </div>
    </div>
  )
}
