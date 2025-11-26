"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"
import { Loader2, Send, FileText, ChevronRight, User, Bot } from "lucide-react"

interface QueryResponse {
  answer: string
  sources: Array<{
    candidate_name?: string
    score?: number
    content?: string
  }>
}

interface Message {
  id: string
  type: "user" | "assistant"
  content: string
  sources?: QueryResponse["sources"]
  timestamp: Date
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

export default function CVScreener() {
  const [question, setQuestion] = useState("")
  const [loading, setLoading] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [error, setError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!question.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: question.trim(),
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, userMessage])

    const currentQuestion = question.trim()
    setQuestion("")
    setLoading(true)
    setError(null)

    try {
      const res = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: currentQuestion,
          top_k: 5,
        }),
      })

      if (!res.ok) {
        throw new Error(`API error: ${res.status}`)
      }

      const data: QueryResponse = await res.json()

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "assistant",
        content: data.answer,
        sources: data.sources,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, assistantMessage])
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to query CVs")
      console.error("[v0] Error querying CV API:", err)
    } finally {
      setLoading(false)
    }
  }

  const exampleQuestions = [
    "What AWS experience does Evelyn Hamilton have?",
    "Which candidates have React experience?",
    "Who has worked with Python?",
    "Find candidates with project management skills",
  ]

  const handleExampleClick = (q: string) => {
    setQuestion(q)
  }

  return (
    <div className="flex h-screen flex-col bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex size-10 items-center justify-center rounded-lg bg-primary">
                <FileText className="size-5 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-foreground">CV Screener</h1>
                <p className="text-sm text-muted-foreground">AI-powered candidate search</p>
              </div>
            </div>
            {messages.length > 0 && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setMessages([])
                  setError(null)
                }}
              >
                Clear Chat
              </Button>
            )}
          </div>
        </div>
      </header>

      {/* Main Chat Area */}
      <main className="flex-1 overflow-hidden">
        <div className="h-full overflow-y-auto">
          <div className="container mx-auto px-4 py-6">
            <div className="mx-auto max-w-4xl space-y-6">
              {messages.length === 0 && (
                <div className="space-y-6 py-12">
                  <div className="text-center space-y-2">
                    <h2 className="text-3xl font-bold text-foreground text-balance">Find the perfect candidate</h2>
                    <p className="text-muted-foreground text-lg">
                      Ask questions about candidates and get instant answers from their CVs
                    </p>
                  </div>

                  {/* Example Questions */}
                  <div className="space-y-3">
                    <p className="text-sm font-medium text-muted-foreground text-center">Try asking:</p>
                    <div className="grid gap-2 sm:grid-cols-2">
                      {exampleQuestions.map((q, idx) => (
                        <button
                          key={idx}
                          onClick={() => handleExampleClick(q)}
                          className="group flex items-center gap-2 rounded-lg border border-border bg-card p-3 text-left text-sm transition-colors hover:bg-accent hover:text-accent-foreground"
                          disabled={loading}
                        >
                          <ChevronRight className="size-4 text-muted-foreground group-hover:text-accent-foreground transition-colors" />
                          <span className="text-pretty">{q}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex gap-4 ${message.type === "user" ? "justify-end" : "justify-start"}`}
                >
                  {message.type === "assistant" && (
                    <div className="flex size-8 shrink-0 items-center justify-center rounded-full bg-primary">
                      <Bot className="size-4 text-primary-foreground" />
                    </div>
                  )}

                  <div className={`flex-1 space-y-2 ${message.type === "user" ? "max-w-[80%]" : "max-w-full"}`}>
                    <div
                      className={`rounded-lg p-4 ${
                        message.type === "user"
                          ? "bg-primary text-primary-foreground ml-auto"
                          : "bg-card border border-border"
                      }`}
                    >
                      <p className="leading-relaxed">{message.content}</p>
                    </div>

                    {/* Sources for assistant messages */}
                    {message.type === "assistant" && message.sources && message.sources.length > 0 && (
                      <div className="space-y-2 pl-4">
                        <p className="text-xs font-medium text-muted-foreground">
                          Relevant CV Sections ({message.sources.length})
                        </p>
                        <div className="space-y-2">
                          {message.sources.map((source, idx) => (
                            <Card key={idx} className="bg-muted/50 border-border p-3">
                              <div className="space-y-1">
                                <div className="flex items-center justify-between">
                                  {source.candidate_name && (
                                    <span className="text-xs font-medium text-foreground">{source.candidate_name}</span>
                                  )}
                                  {source.score && (
                                    <span className="text-xs text-muted-foreground">
                                      {(source.score * 100).toFixed(1)}%
                                    </span>
                                  )}
                                </div>
                                {source.content && (
                                  <p className="text-xs text-muted-foreground leading-relaxed line-clamp-2">
                                    {source.content}
                                  </p>
                                )}
                              </div>
                            </Card>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {message.type === "user" && (
                    <div className="flex size-8 shrink-0 items-center justify-center rounded-full bg-muted">
                      <User className="size-4 text-muted-foreground" />
                    </div>
                  )}
                </div>
              ))}

              {loading && (
                <div className="flex gap-4">
                  <div className="flex size-8 shrink-0 items-center justify-center rounded-full bg-primary">
                    <Bot className="size-4 text-primary-foreground" />
                  </div>
                  <div className="rounded-lg border border-border bg-card p-4">
                    <div className="flex items-center gap-2">
                      <Loader2 className="size-4 animate-spin text-muted-foreground" />
                      <span className="text-sm text-muted-foreground">Searching CVs...</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Error Display */}
              {error && (
                <Card className="border-destructive/50 bg-destructive/10 p-4">
                  <div className="space-y-1">
                    <h3 className="text-sm font-semibold text-destructive">Error</h3>
                    <p className="text-xs text-destructive/90">{error}</p>
                    <p className="text-xs text-muted-foreground">
                      Make sure the CV Screener API is running on {API_BASE_URL}
                    </p>
                  </div>
                </Card>
              )}

              <div ref={messagesEndRef} />
            </div>
          </div>
        </div>
      </main>

      <div className="border-t border-border bg-card">
        <div className="container mx-auto px-4 py-4">
          <div className="mx-auto max-w-4xl">
            <form onSubmit={handleQuery} className="flex gap-2">
              <Input
                type="text"
                placeholder="Ask a question about candidates..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                className="flex-1 bg-background"
                disabled={loading}
              />
              <Button type="submit" disabled={loading || !question.trim()}>
                {loading ? <Loader2 className="size-4 animate-spin" /> : <Send className="size-4" />}
              </Button>
            </form>
          </div>
        </div>
      </div>
    </div>
  )
}
