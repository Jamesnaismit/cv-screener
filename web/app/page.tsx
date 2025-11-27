"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import ReactMarkdown from "react-markdown"
import { ThemeToggle } from "@/components/theme-toggle"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"
import {
  Loader2,
  Send,
  FileText,
  ChevronRight,
  ChevronsUpDown,
  User,
  Bot,
  ExternalLink,
  Sparkles,
  ShieldCheck,
} from "lucide-react"

interface QueryResponse {
  answer: string
  sources: Array<{
    title?: string
    url?: string
    content?: string
    similarity?: number
    relevance_score?: number
    metadata?: Record<string, unknown>
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
  const [openSources, setOpenSources] = useState<Record<string, boolean>>({})

  const markdownComponents = {
    ul: (props: React.HTMLAttributes<HTMLUListElement>) => (
      <ul className="list-disc space-y-1 pl-5" {...props} />
    ),
    ol: (props: React.HTMLAttributes<HTMLOListElement>) => (
      <ol className="list-decimal space-y-1 pl-5" {...props} />
    ),
    li: (props: React.HTMLAttributes<HTMLLIElement>) => <li className="leading-relaxed" {...props} />,
  }

  const toSourceLink = (url?: string) => {
    if (!url) return null
    if (url.startsWith("http://") || url.startsWith("https://")) return url
    const cleaned = url.replace("cv://", "").replace(/^\/+/, "")
    return `${API_BASE_URL}/cv/${encodeURIComponent(cleaned)}`
  }

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const runQuery = async (prompt: string) => {
    const trimmed = prompt.trim()
    if (!trimmed) return

    const userMessage: Message = {
      id: crypto.randomUUID(),
      type: "user",
      content: trimmed,
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, userMessage])

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
          question: trimmed,
          top_k: 5,
        }),
      })

      if (!res.ok) {
        throw new Error(`API error: ${res.status}`)
      }

      const data: QueryResponse = await res.json()

      const assistantMessage: Message = {
        id: crypto.randomUUID(),
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

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault()
    if (loading) return
    await runQuery(question)
  }

  const exampleQuestions = [
    "What AWS experience does Evelyn Hamilton have?",
    "Which candidates have React experience?",
    "Who has worked with Python?",
    "Find candidates with project management skills",
  ]

  const handleExampleClick = (q: string) => {
    if (loading) return
    setQuestion(q)
    void runQuery(q)
  }

  return (
    <div className="relative min-h-screen overflow-hidden bg-slate-50 text-slate-900 transition-colors duration-300 dark:bg-slate-950 dark:text-slate-50">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(56,189,248,0.18),transparent_35%),radial-gradient(circle_at_80%_0%,rgba(59,130,246,0.12),transparent_32%),radial-gradient(circle_at_50%_80%,rgba(16,185,129,0.12),transparent_30%)] dark:bg-[radial-gradient(circle_at_20%_20%,rgba(56,189,248,0.12),transparent_35%),radial-gradient(circle_at_80%_0%,rgba(14,165,233,0.1),transparent_32%),radial-gradient(circle_at_50%_80%,rgba(74,222,128,0.08),transparent_30%)]" />

      <div className="relative mx-auto flex min-h-screen max-w-6xl flex-col gap-8 px-4 py-10 lg:px-6">
        <header className="flex items-center justify-between rounded-2xl border border-slate-200/80 bg-white/90 px-5 py-4 shadow-lg shadow-sky-100/60 backdrop-blur dark:border-white/10 dark:bg-white/5 dark:shadow-sky-900/20">
          <div className="flex items-center gap-3">
            <div className="flex size-12 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-500 to-emerald-400 text-slate-950 shadow-lg shadow-cyan-500/30">
              <FileText className="size-6" />
            </div>
            <div>
              <p className="text-sm uppercase tracking-[0.18em] text-sky-700 dark:text-cyan-200">CV Screener</p>
              <h1 className="text-xl font-semibold text-slate-900 dark:text-white">AI recruiter workspace</h1>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-sky-700 shadow-sm dark:border-white/10 dark:bg-white/10 dark:text-cyan-100">
              <span className="relative flex size-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400/70 opacity-75" />
                <span className="relative inline-flex size-2 rounded-full bg-emerald-500 dark:bg-emerald-400" />
              </span>
              Live CV knowledgebase
            </div>
            <div className="hidden items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600 shadow-sm sm:flex dark:border-white/10 dark:bg-white/5 dark:text-slate-200">
              <ShieldCheck className="size-3.5 text-emerald-300" />
              Source-backed answers
            </div>
            <ThemeToggle />
          </div>
        </header>

        <div className="grid gap-6 lg:grid-cols-[1fr,1.35fr]">
          <div className="space-y-4">
            <div className="rounded-3xl border border-slate-200 bg-white/90 p-6 shadow-xl shadow-sky-100/80 backdrop-blur dark:border-white/10 dark:bg-white/5 dark:shadow-sky-900/20">
              <div className="flex items-start justify-between gap-3">
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-sm font-medium text-sky-700 dark:text-cyan-200">
                    <Sparkles className="size-4" />
                    Precision search, grounded in CVs
                  </div>
                  <h2 className="text-2xl font-semibold leading-tight text-slate-900 dark:text-white">
                    Ask nuanced questions. Get concise, sourced answers.
                  </h2>
                  <p className="text-sm text-slate-600 dark:text-slate-300">
                    Every response cites the underlying CV snippets, so you can move fast without guessing.
                  </p>
                </div>
                <div className="hidden rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600 shadow-sm sm:block dark:border-white/10 dark:bg-white/10 dark:text-slate-200">
                  {API_BASE_URL.replace("http://", "")}
                </div>
              </div>

              <div className="mt-5 grid gap-2 sm:grid-cols-2">
                {exampleQuestions.map((q, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleExampleClick(q)}
                    className="group flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-3 text-left text-sm text-slate-700 transition hover:-translate-y-0.5 hover:border-sky-200 hover:bg-sky-50 hover:text-slate-900 dark:border-white/10 dark:bg-slate-900/50 dark:text-slate-200 dark:hover:border-cyan-400/60 dark:hover:bg-slate-900 dark:hover:text-white"
                    disabled={loading}
                  >
                    <ChevronRight className="size-4 text-sky-600 transition group-hover:translate-x-0.5 dark:text-cyan-300" />
                    <span className="text-pretty">{q}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="rounded-3xl border border-slate-200 bg-white/90 shadow-2xl shadow-sky-100/90 backdrop-blur dark:border-white/10 dark:bg-white/5 dark:shadow-sky-900/30">
            <div className="flex items-center justify-between border-b border-slate-200 px-5 py-4 dark:border-white/10">
              <div>
                <p className="text-xs uppercase tracking-[0.22em] text-sky-700 dark:text-cyan-200">Conversation</p>
                <h3 className="text-lg font-semibold text-slate-900 dark:text-white">Ask about any candidate</h3>
                <p className="text-xs text-slate-600 dark:text-slate-300">
                  Responses include citations and stay within your CV data.
                </p>
              </div>
              {messages.length > 0 && (
                <Button
                  variant="outline"
                  size="sm"
                  className="border-slate-200 bg-white text-slate-700 hover:bg-sky-50 dark:border-white/20 dark:bg-white/10 dark:text-slate-100 dark:hover:bg-white/20"
                  onClick={() => {
                    setMessages([])
                    setError(null)
                  }}
                >
                  Clear chat
                </Button>
              )}
            </div>

            <div className="flex h-[65vh] flex-col">
              <div className="flex-1 space-y-4 overflow-y-auto px-5 py-4">
                {messages.length === 0 && !loading && !error && (
                  <div className="flex flex-col items-start gap-3 rounded-2xl border border-dashed border-slate-200 bg-slate-50 p-5 text-left shadow-sm dark:border-white/10 dark:bg-slate-900/40">
                    <div className="flex items-center gap-2 text-sm font-semibold text-slate-900 dark:text-white">
                      <Sparkles className="size-4 text-sky-600 dark:text-cyan-300" />
                      Ready when you are
                    </div>
                    <p className="text-sm text-slate-600 dark:text-slate-300">
                      Ask about skills, projects, or who fits a role. I&apos;ll return a concise, source-backed summary.
                    </p>
                  </div>
                )}

                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex gap-3 ${message.type === "user" ? "justify-end" : "justify-start"}`}
                  >
                    {message.type === "assistant" && (
                      <div className="flex size-9 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-cyan-500 to-emerald-400 text-slate-950 shadow-lg shadow-cyan-500/30">
                        <Bot className="size-4" />
                      </div>
                    )}

                    <div className={`flex-1 space-y-2 ${message.type === "user" ? "max-w-[82%]" : "max-w-full"}`}>
                      <div
                        className={`rounded-2xl px-4 py-3 text-sm ${
                          message.type === "user"
                            ? "ml-auto border border-emerald-200 bg-gradient-to-br from-sky-100 to-emerald-50 text-slate-900 shadow-lg shadow-emerald-200/60 dark:border-emerald-300/30 dark:from-cyan-500/20 dark:to-emerald-500/20 dark:text-slate-50 dark:shadow-emerald-900/30"
                            : "border border-slate-200 bg-slate-100 text-slate-900 shadow-lg shadow-sky-100/70 dark:border-white/10 dark:bg-slate-900/70 dark:text-slate-50 dark:shadow-slate-900/40"
                        }`}
                      >
                        {message.type === "assistant" ? (
                          <div className="prose prose-sm max-w-none prose-p:leading-relaxed prose-ul:my-2 prose-li:my-0 dark:prose-invert">
                            <ReactMarkdown components={markdownComponents}>{message.content}</ReactMarkdown>
                          </div>
                        ) : (
                          <p className="leading-relaxed">{message.content}</p>
                        )}
                      </div>

                      {message.type === "assistant" && message.sources && message.sources.length > 0 && (
                        <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 shadow-sm dark:border-white/10 dark:bg-slate-900/60 dark:shadow-lg dark:shadow-slate-900/30">
                          <button
                            type="button"
                            className="flex w-full items-center justify-between rounded-lg px-2 py-1 text-left text-[11px] font-semibold uppercase tracking-[0.18em] text-sky-700 hover:text-slate-900 dark:text-cyan-200 dark:hover:text-white"
                            onClick={() =>
                              setOpenSources((prev) => ({
                                ...prev,
                                [message.id]: !prev[message.id],
                              }))
                            }
                          >
                            Sources
                            <ChevronsUpDown className="size-3.5 text-sky-700 dark:text-cyan-200" />
                          </button>
                          {openSources[message.id] && (
                            <div className="mt-2 grid gap-2 sm:grid-cols-2">
                              {message.sources.map((source, idx) => {
                                const title = source.title || source.url || `Source ${idx + 1}`
                                const linkUrl = toSourceLink(source.url)
                                const citeNumber = idx + 1
                                const CardInner = () => (
                                  <div className="rounded-lg border border-slate-200 bg-white p-3 text-xs text-slate-800 shadow-sm transition hover:border-sky-200 hover:bg-sky-50 dark:border-white/10 dark:bg-slate-950/50 dark:text-slate-100 dark:hover:border-cyan-400/50 dark:hover:bg-slate-900/70">
                                    <div className="flex items-start justify-between gap-2">
                                      <span className="font-semibold text-slate-900 line-clamp-1 dark:text-white">
                                        {title}
                                      </span>
                                      <span className="rounded-full border border-slate-200 bg-slate-100 px-2 py-[2px] text-[11px] font-semibold text-sky-700 dark:border-white/10 dark:bg-slate-900/70 dark:text-cyan-200">
                                        {citeNumber}
                                      </span>
                                    </div>
                                    {source.url && (
                                      <p className="mt-1 text-[11px] text-slate-500 line-clamp-1" title={source.url}>
                                        {source.url}
                                      </p>
                                    )}
                                    {source.content && (
                                      <p className="mt-2 text-[11px] text-slate-700 leading-relaxed line-clamp-3 dark:text-slate-200">
                                        {source.content}
                                      </p>
                                    )}
                                    {linkUrl ? (
                                      <div className="mt-2 inline-flex items-center gap-1 text-[11px] font-semibold text-sky-700 dark:text-cyan-200">
                                        <ExternalLink className="size-3" />
                                        View CV
                                      </div>
                                    ) : (
                                      <p className="mt-2 text-[11px] text-slate-500 dark:text-slate-400">No direct link available</p>
                                    )}
                                  </div>
                                )

                                return (
                                  <div key={idx}>
                                    {linkUrl ? (
                                      <a href={linkUrl} target="_blank" rel="noreferrer" className="block">
                                        <CardInner />
                                      </a>
                                    ) : (
                                      <CardInner />
                                    )}
                                  </div>
                                )
                              })}
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    {message.type === "user" && (
                      <div className="flex size-9 shrink-0 items-center justify-center rounded-full border border-slate-200 bg-slate-100 text-slate-700 shadow-sm dark:border-white/10 dark:bg-slate-900/70 dark:text-slate-200">
                        <User className="size-4" />
                      </div>
                    )}
                  </div>
                ))}

                {loading && (
                  <div className="flex items-center gap-3">
                    <div className="flex size-9 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-cyan-500 to-emerald-400 text-slate-950 shadow-lg shadow-cyan-500/30">
                      <Bot className="size-4" />
                    </div>
                    <div className="rounded-2xl border border-slate-200 bg-slate-100 px-4 py-3 text-sm text-slate-800 shadow-lg shadow-sky-100/70 dark:border-white/10 dark:bg-slate-900/60 dark:text-slate-200 dark:shadow-slate-900/40">
                      <div className="flex items-center gap-2">
                        <Loader2 className="size-4 animate-spin text-sky-600 dark:text-cyan-300" />
                        Searching CVs...
                      </div>
                    </div>
                  </div>
                )}

                {error && (
                  <Card className="border-destructive/40 bg-destructive/15 text-destructive-foreground">
                    <div className="space-y-1 p-4">
                      <h3 className="text-sm font-semibold text-slate-900 dark:text-white">Error</h3>
                      <p className="text-xs text-red-700 dark:text-red-100/90">{error}</p>
                      <p className="text-xs text-slate-600 dark:text-slate-200">
                        Make sure the CV Screener API is running on {API_BASE_URL}
                      </p>
                    </div>
                  </Card>
                )}

                <div ref={messagesEndRef} />
              </div>

              <div className="border-t border-slate-200 bg-white/80 px-5 py-4 backdrop-blur dark:border-white/10 dark:bg-slate-900/60">
                <form onSubmit={handleQuery} className="flex gap-2">
                  <Input
                    type="text"
                    placeholder="Ask about a skill, candidate, or role..."
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    className="flex-1 border-slate-200 bg-white text-slate-900 placeholder:text-slate-500 dark:border-white/10 dark:bg-slate-950/60 dark:text-slate-50 dark:placeholder:text-slate-400"
                    disabled={loading}
                  />
                  <Button
                    type="submit"
                    className="bg-gradient-to-br from-cyan-500 to-emerald-400 text-slate-950 shadow-lg shadow-cyan-500/30 hover:from-cyan-400 hover:to-emerald-300"
                    disabled={loading || !question.trim()}
                  >
                    {loading ? <Loader2 className="size-4 animate-spin" /> : <Send className="size-4" />}
                  </Button>
                </form>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
