import React, { useState } from 'react'
import axios from 'axios'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

export default function App() {
  const [userId, setUserId] = useState('user-1')
  const [input, setInput] = useState('Explain two pointers technique with examples.')
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const ask = async () => {
    if (!input.trim()) return
    setError(null)
    setLoading(true)
    const userMsg: Message = { role: 'user', content: input }
    setMessages(prev => [...prev, userMsg])
    setInput('')

    try {
      const res = await axios.post('http://localhost:8000/ask', {
        user_id: userId,
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
        <div className="user">
          <label>User ID</label>
          <input value={userId} onChange={e => setUserId(e.target.value)} />
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
