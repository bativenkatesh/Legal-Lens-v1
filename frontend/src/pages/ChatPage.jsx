import React, { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import { Sparkles, Send, Paperclip, Image, File, PiggyBank, Calculator, TrendingUp, Settings, HelpCircle, LogOut, ChevronDown, Bell, Grid3x3, MoreVertical, MessageSquare, Plus, Trash2 } from 'lucide-react'
import '../App.css'

const API_BASE_URL = 'http://localhost:8000'

function ChatPage() {
    const [messages, setMessages] = useState([])
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const [conversationHistory, setConversationHistory] = useState([])
    const [chatHistory, setChatHistory] = useState([])
    const [currentChatId, setCurrentChatId] = useState(null)
    const messagesEndRef = useRef(null)

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }

    useEffect(() => {
        scrollToBottom()
    }, [messages])

    // Load chat history from localStorage on mount
    useEffect(() => {
        const savedHistory = localStorage.getItem('chatHistory')
        if (savedHistory) {
            try {
                setChatHistory(JSON.parse(savedHistory))
            } catch (e) {
                console.error('Error loading chat history:', e)
            }
        }
    }, [])

    // Save chat history to localStorage whenever it changes
    useEffect(() => {
        if (chatHistory.length > 0) {
            localStorage.setItem('chatHistory', JSON.stringify(chatHistory))
        }
    }, [chatHistory])

    const sendMessage = async (e) => {
        e.preventDefault()
        if (!input.trim() || loading) return

        const userMessage = input.trim()
        setInput('')
        setLoading(true)

        const newUserMessage = {
            role: 'user',
            content: userMessage,
            timestamp: new Date()
        }
        setMessages(prev => [...prev, newUserMessage])

        const updatedHistory = [
            ...conversationHistory,
            { role: 'user', content: userMessage }
        ]

        try {
            const response = await axios.post(`${API_BASE_URL}/chat`, {
                message: userMessage,
                conversation_history: updatedHistory
            })

            const botMessage = {
                role: 'assistant',
                content: response.data.response,
                relevantSections: response.data.relevant_sections,
                relevantArticles: response.data.relevant_articles,
                confidence: response.data.confidence,
                timestamp: new Date()
            }

            const updatedMessages = [...messages, newUserMessage, botMessage]
            setMessages(updatedMessages)

            const finalHistory = [
                ...updatedHistory,
                { role: 'assistant', content: response.data.response }
            ]
            setConversationHistory(finalHistory)

            // Save to chat history if this is a new conversation
            if (!currentChatId) {
                const newChatId = Date.now().toString()
                setCurrentChatId(newChatId)
                const newChat = {
                    id: newChatId,
                    title: userMessage.substring(0, 50) + (userMessage.length > 50 ? '...' : ''),
                    messages: updatedMessages,
                    timestamp: new Date()
                }
                setChatHistory(prev => [newChat, ...prev])
            } else {
                // Update existing chat
                setChatHistory(prev => prev.map(chat =>
                    chat.id === currentChatId
                        ? { ...chat, messages: updatedMessages }
                        : chat
                ))
            }
        } catch (error) {
            console.error('Error:', error)
            const errorMessage = {
                role: 'assistant',
                content: 'Sorry, I encountered an error. Please try again.',
                error: true,
                timestamp: new Date()
            }
            setMessages(prev => [...prev, errorMessage])
        } finally {
            setLoading(false)
        }
    }

    const clearChat = () => {
        setMessages([])
        setConversationHistory([])
        setCurrentChatId(null)
    }

    const loadChat = (chatId) => {
        const chat = chatHistory.find(c => c.id === chatId)
        if (chat) {
            setMessages(chat.messages)
            setCurrentChatId(chatId)
            // Rebuild conversation history from messages
            const history = chat.messages.map(msg => ({
                role: msg.role,
                content: msg.content
            }))
            setConversationHistory(history)
        }
    }

    const deleteChat = (chatId, e) => {
        e.stopPropagation()
        setChatHistory(prev => prev.filter(chat => chat.id !== chatId))
        if (currentChatId === chatId) {
            clearChat()
        }
    }

    const startNewChat = () => {
        clearChat()
    }

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            sendMessage(e)
        }
    }

    const promptSuggestions = [
        {
            title: "Smart Budget",
            description: "Create a budget that adapts to your lifestyle and goals.",
            icon: PiggyBank
        },
        {
            title: "Calculation",
            description: "Easily crunch the numbers for clearer money choices.",
            icon: Calculator
        },
        {
            title: "Spending",
            description: "See your spending habits and spot useful patterns.",
            icon: TrendingUp
        }
    ]

    const handlePromptClick = (prompt) => {
        setInput(prompt.description)
    }

    const footerItems = [
        { title: "Setting", icon: Settings },
        { title: "Help Center", icon: HelpCircle },
        { title: "Sign Out", icon: LogOut },
    ]

    return (
        <div className="flex min-h-screen w-full bg-background">
            {/* Sidebar */}
            <aside className="sidebar">
                <div className="sidebar-header">
                    <button className="sidebar-dropdown">
                        <div className="flex items-center gap-2">
                            <div className="w-6 h-6 rounded bg-primary flex items-center justify-center">
                                <span className="text-xs font-semibold text-primary-foreground">P</span>
                            </div>
                            <span className="text-sm font-medium text-sidebar-foreground">Personal</span>
                        </div>
                        <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    </button>
                </div>

                <div className="sidebar-content">
                    {/* New Chat Button */}
                    <button onClick={startNewChat} className="new-chat-button">
                        <Plus className="h-4 w-4" />
                        <span>New Chat</span>
                    </button>

                    {/* Chat History */}
                    <div className="chat-history-section">
                        <h3 className="chat-history-title">Chat History</h3>
                        <div className="chat-history-list">
                            {chatHistory.length === 0 ? (
                                <div className="no-chats-message">
                                    <MessageSquare className="h-8 w-8 text-muted-foreground" />
                                    <p>No previous chats</p>
                                    <p className="text-xs">Start a conversation to see it here</p>
                                </div>
                            ) : (
                                chatHistory.map((chat) => (
                                    <div
                                        key={chat.id}
                                        className={`chat-history-item ${currentChatId === chat.id ? 'active' : ''}`}
                                        onClick={() => loadChat(chat.id)}
                                    >
                                        <MessageSquare className="h-4 w-4" />
                                        <span className="chat-history-item-title">{chat.title}</span>
                                        <button
                                            className="delete-chat-button"
                                            onClick={(e) => deleteChat(chat.id, e)}
                                            title="Delete chat"
                                        >
                                            <Trash2 className="h-3 w-3" />
                                        </button>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                </div>

                <div className="sidebar-footer">
                    {footerItems.map((item) => (
                        <a key={item.title} href="#" className="nav-item">
                            <item.icon className="h-4 w-4" />
                            <span>{item.title}</span>
                        </a>
                    ))}
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1">
                {/* Header */}
                <header className="header">
                    <h1 className="header-title">Legal Lens</h1>
                </header>

                <div className="main-content-wrapper">
                    {messages.length === 0 ? (
                        <>
                            {/* Welcome Section */}
                            <div className="welcome-section">
                                <div className="welcome-icon">
                                    <Sparkles className="h-10 w-10 text-white" />
                                </div>
                                <h2 className="welcome-title">Welcome to Legal Lens!</h2>
                                <p className="welcome-description">
                                    Your tax law companion — clear, accurate, and designed to help you navigate the Income Tax Act, 1961.
                                </p>
                            </div>

                            {/* AI Model Card */}
                            <div className="ai-model-card">
                                <div className="flex items-center gap-3">
                                    <div className="ai-model-icon">
                                        <Sparkles className="h-5 w-5 text-primary-foreground" />
                                    </div>
                                    <div>
                                        <h3 className="ai-model-title">Powered by Advanced AI</h3>
                                        <p className="ai-model-subtitle">Expert tax law analysis and intelligent document retrieval</p>
                                    </div>
                                </div>
                            </div>

                            {/* Chat Input */}
                            <div className="chat-input-card">
                                <div className="chat-input-wrapper">
                                    <input
                                        type="text"
                                        value={input}
                                        onChange={(e) => setInput(e.target.value)}
                                        onKeyPress={handleKeyPress}
                                        placeholder="Ask Legal Lens..."
                                        className="chat-input"
                                        disabled={loading}
                                    />
                                </div>
                                <div className="chat-input-actions">
                                    <div className="chat-input-icons">
                                        <button className="icon-button">
                                            <Paperclip className="h-4 w-4" />
                                        </button>
                                        <button className="icon-button">
                                            <Image className="h-4 w-4" />
                                        </button>
                                        <button className="icon-button">
                                            <File className="h-4 w-4" />
                                        </button>
                                    </div>
                                    <button
                                        onClick={sendMessage}
                                        className="send-button"
                                        disabled={loading || !input.trim()}
                                    >
                                        <Send className="h-4 w-4 text-primary-foreground" />
                                    </button>
                                </div>
                            </div>
                        </>
                    ) : (
                        <>
                            {/* Messages Area */}
                            <div className="messages-area">
                                <div className="messages-header">
                                    <h3>Conversation</h3>
                                    <button onClick={clearChat} className="clear-button">Clear</button>
                                </div>
                                <div className="messages-list">
                                    {messages.map((msg, idx) => (
                                        <div
                                            key={idx}
                                            className={`message ${msg.role === 'user' ? 'user-message' : 'bot-message'}`}
                                        >
                                            <div className="message-content">
                                                <div className="message-text">{msg.content}</div>
                                                {msg.relevantSections && msg.relevantSections.length > 0 && (
                                                    <div className="relevant-sections">
                                                        <h4 className="relevant-sections-title">📚 Relevant Sections:</h4>
                                                        {msg.relevantSections.map((section, secIdx) => (
                                                            <div key={secIdx} className="section-card">
                                                                <div className="section-header">
                                                                    <strong>Section {section.section}</strong>
                                                                    <span className="similarity-score">
                                                                        {(section.similarity_score * 100).toFixed(1)}%
                                                                    </span>
                                                                </div>
                                                                <div className="section-title">{section.title}</div>
                                                                <div className="section-summary">{section.summary}</div>
                                                            </div>
                                                        ))}
                                                    </div>
                                                )}
                                                {msg.relevantArticles && msg.relevantArticles.length > 0 && (
                                                    <div className="relevant-articles" style={{ marginTop: '1.5rem', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '1rem' }}>
                                                        <h4 className="relevant-sections-title" style={{ color: '#60a5fa' }}>📰 Related Articles:</h4>
                                                        {msg.relevantArticles.map((art, artIdx) => (
                                                            <div key={artIdx} className="section-card" style={{ borderLeft: '3px solid #60a5fa' }}>
                                                                <div className="section-header">
                                                                    <strong style={{ fontSize: '0.9rem' }}>{art.title}</strong>
                                                                </div>
                                                                <div className="section-date" style={{ fontSize: '0.75rem', color: '#9ca3af', marginBottom: '0.25rem' }}>
                                                                    {art.date} • {art.author}
                                                                </div>
                                                                <div className="section-summary" style={{ fontSize: '0.85rem', marginBottom: '0.5rem' }}>
                                                                    {art.snippet}
                                                                </div>
                                                                <div className="section-footer" style={{ fontSize: '0.75rem', color: '#60a5fa' }}>
                                                                    Refers to Section {art.related_section}
                                                                </div>
                                                            </div>
                                                        ))}
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    ))}
                                    {loading && (
                                        <div className="message bot-message">
                                            <div className="message-content">
                                                <div className="loading-dots">
                                                    <span></span>
                                                    <span></span>
                                                    <span></span>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                    <div ref={messagesEndRef} />
                                </div>
                            </div>

                            {/* Input Card at Bottom for Chat */}
                            <div className="chat-input-card" style={{ marginTop: '20px' }}>
                                <div className="chat-input-wrapper">
                                    <input
                                        type="text"
                                        value={input}
                                        onChange={(e) => setInput(e.target.value)}
                                        onKeyPress={handleKeyPress}
                                        placeholder="Ask Legal Lens..."
                                        className="chat-input"
                                        disabled={loading}
                                    />
                                </div>
                                <div className="chat-input-actions">
                                    <div className="chat-input-icons">
                                        <button className="icon-button">
                                            <Paperclip className="h-4 w-4" />
                                        </button>
                                        <button className="icon-button">
                                            <Image className="h-4 w-4" />
                                        </button>
                                        <button className="icon-button">
                                            <File className="h-4 w-4" />
                                        </button>
                                    </div>
                                    <button
                                        onClick={sendMessage}
                                        className="send-button"
                                        disabled={loading || !input.trim()}
                                    >
                                        <Send className="h-4 w-4 text-primary-foreground" />
                                    </button>
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </main>
        </div>
    )
}

export default ChatPage
