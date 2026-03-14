import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Sparkles, ArrowRight, Play, Check, ChevronDown, Plus, Search, Shield, Zap, FileText, Star } from 'lucide-react';
import heroAITax from '../assets/hero_ai_tax.png';
import featureHighlightDashboard from '../assets/feature_highlight_dashboard.png';

const LandingPage = () => {
    const [openFaq, setOpenFaq] = useState(0);

    const faqs = [
        { q: "What is Legal Lens AI?", a: "Legal Lens AI is an advanced retrieval-augmented generation (RAG) tool designed to help tax professionals navigate the complex Income Tax Act effortlessly." },
        { q: "How accurate are the answers?", a: "Our AI processes official documents and citations to ensure 99.2% accuracy, always grounding its responses in the actual text of the law." },
        { q: "Can it extract data from my bills?", a: "Yes, our OCR Dashboard allows you to upload medical bills or GST invoices and automatically extracts relevant tax fields for you." },
        { q: "Does it support the latest amendments?", a: "Absolutely. Our database is continuously updated with the latest circulars, notifications, and amendments from the CBDT." }
    ];

    return (
        <div className="min-h-screen font-sans bg-[#FAF7F2] text-[#111111] overflow-hidden selection:bg-[#F5B027] selection:text-white">

            {/* 1. Navigation */}
            <nav className="fixed top-6 left-6 right-6 lg:left-1/2 lg:-translate-x-1/2 lg:w-full lg:max-w-6xl z-50 flex items-center justify-between px-4 py-3 bg-white/90 backdrop-blur-md rounded-full shadow-lg shadow-black/5 border border-white">
                <div className="flex items-center gap-2">
                    <div className="flex items-center justify-center w-8 h-8 rounded-full bg-gradient-to-tr from-gray-900 to-gray-700">
                        <Sparkles className="w-4 h-4 text-white" />
                    </div>
                    <span className="text-xl font-bold tracking-tight">Legal Lens</span>
                </div>

                <div className="hidden gap-8 text-sm font-semibold text-gray-600 md:flex">
                    <a href="#" className="text-gray-900 transition-colors">Product</a>
                    <a href="#" className="hover:text-gray-900 transition-colors">Workflow</a>
                    <a href="#" className="hover:text-gray-900 transition-colors">Pricing</a>
                    <a href="#" className="hover:text-gray-900 transition-colors">Blog</a>
                    <a href="#" className="hover:text-gray-900 transition-colors">Contact</a>
                </div>

                <div className="flex items-center gap-3">
                    <Link to="/chat" className="hidden px-5 py-2.5 text-sm font-bold text-gray-700 transition-colors border-2 border-transparent rounded-full sm:block hover:border-gray-200">
                        Log in
                    </Link>
                    <Link to="/chat" className="px-5 py-2.5 text-sm font-bold text-white transition-all bg-gray-900 rounded-full hover:bg-gray-800 hover:shadow-lg hover:-translate-y-0.5">
                        Start for free
                    </Link>
                </div>
            </nav>

            {/* 2. Hero Section */}
            <section className="relative pt-32 pb-20 overflow-hidden text-center md:pt-40 lg:pt-48">
                {/* Background Gradient Blob */}
                <div className="absolute top-10 left-1/2 -translate-x-1/2 w-[800px] h-[600px] bg-gradient-to-b from-[#FAD9B8] via-[#F2E5D5] to-transparent rounded-full blur-3xl opacity-60 -z-10" />

                <h1 className="text-5xl md:text-7xl lg:text-[84px] font-black tracking-tighter leading-[0.9] text-gray-900 mb-6 relative z-10 flex flex-col items-center justify-center gap-4">
                    <div className="flex items-center gap-4">
                        <span>AI Tax Assistants</span>
                        {/* Soundwave graphic mimicking the reference */}
                        <div className="flex items-center gap-1.5 h-12 md:h-16">
                            {[1, 3, 5, 4, 6, 4, 3, 5, 2, 4, 5, 3].map((h, i) => (
                                <div key={i} className="w-1.5 md:w-2 bg-[#F5B027] rounded-full" style={{ height: `${h * 15}%`, opacity: 0.6 + (h * 0.05) }} />
                            ))}
                        </div>
                        <span>That</span>
                    </div>
                    <span className="text-gray-400 font-medium tracking-tight">Master The Law</span>
                </h1>

                {/* Floating Elements Container */}
                <div className="relative max-w-6xl mx-auto mt-16 md:mt-24 h-[400px] md:h-[500px]">

                    {/* Left Column Text & CTA */}
                    <div className="absolute -left-4 lg:-left-20 xl:-left-32 top-1/2 -translate-y-1/2 z-30 max-w-[280px] hidden lg:block text-left pt-20">
                        <p className="text-gray-600 font-medium mb-6 text-[15px] leading-relaxed">
                            GenFuse automates complex operations through AI agents that think, learn, and act seamlessly.
                        </p>
                        <div className="flex items-center gap-3">
                            <Link to="/chat" className="px-5 py-3 text-sm font-bold text-white bg-[#111111] rounded-full hover:bg-gray-800 transition-all">
                                Try for Free
                            </Link>
                            <Link to="/chat" className="flex items-center justify-center w-11 h-11 bg-[#F5B027] rounded-full text-white hover:bg-[#EAA115] transition-colors">
                                <ArrowRight className="w-5 h-5 -rotate-45" />
                            </Link>
                        </div>
                    </div>

                    {/* Central Image Wrapper */}
                    <div className="absolute inset-0 z-10 mx-auto w-full max-w-4xl">

                        {/* Central Image */}
                        <div className="absolute inset-x-8 inset-y-0 mx-auto overflow-hidden shadow-none lg:shadow-2xl rounded-none lg:rounded-[2.5rem] bg-transparent text-center z-10 flex justify-center">
                            <img
                                src={heroAITax}
                                alt="Legal Lens AI Tax Assistant Concept"
                                className="object-cover h-full xl:w-full rounded-[2.5rem]"
                            />
                        </div>

                        {/* Badge 1: 500+ (Top Left) */}
                        <div className="absolute top-10 lg:-top-4 -left-4 md:left-8 z-30 flex flex-col p-5 bg-white rounded-[1.5rem] shadow-[0_8px_30px_rgb(0,0,0,0.05)] text-left w-[180px]">
                            <div className="flex items-baseline mb-1">
                                <span className="text-[40px] leading-none font-black text-gray-900 tracking-tight">200</span>
                                <span className="text-3xl font-bold text-[#F5B027]">+</span>
                            </div>
                            <span className="text-xs font-medium text-gray-500 leading-snug tracking-tight">Systems Connected and<br />Automated</span>
                        </div>

                        {/* Badge 2: 12K+ (Top Right) */}
                        <div className="absolute top-36 md:top-24 -right-2 md:right-10 z-30 flex items-center gap-3 bg-white rounded-full shadow-[0_8px_30px_rgb(0,0,0,0.05)] py-2.5 px-3 pr-2 border border-gray-50">
                            <Star className="w-5 h-5 text-[#F5B027] fill-current ml-1" />
                            <div className="flex flex-col text-left pr-2">
                                <span className="font-black text-[15px] leading-none mb-0.5 text-gray-900">12K+</span>
                                <span className="text-[10px] text-gray-400 font-medium tracking-wide">Automate Daily</span>
                            </div>
                            <div className="flex -space-x-2">
                                <img src="https://i.pravatar.cc/100?img=5" className="w-8 h-8 rounded-full border-2 border-white relative z-10" alt="user" />
                                <img src="https://i.pravatar.cc/100?img=11" className="w-8 h-8 rounded-full border-2 border-white relative z-0" alt="user" />
                            </div>
                        </div>

                        {/* Badge 3: Smart Decision (Bottom Left Center) */}
                        <div className="absolute -bottom-12 md:-bottom-8 left-1/4 lg:left-[22%] z-30 flex flex-col items-center justify-center p-4 bg-white rounded-[1.5rem] shadow-[0_8px_30px_rgb(0,0,0,0.05)] w-[130px]">
                            <div className="relative mb-2 text-[#F5B027] w-12 h-12">
                                <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M50 0 L58 35 L96 35 L66 56 L76 92 L50 72 L24 92 L34 56 L4 35 L42 35 Z" />
                                </svg>
                                <Sparkles className="w-4 h-4 text-white absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
                            </div>
                            <span className="text-[10px] font-bold text-gray-900 text-center leading-tight">Smart Decision<br />Assistance</span>
                        </div>

                        {/* Badge 4: 0.25s (Bottom Right) */}
                        <div className="absolute -bottom-8 md:bottom-8 -right-4 md:-right-4 lg:-right-12 z-30 flex flex-col p-5 bg-white rounded-[1.5rem] shadow-[0_8px_30px_rgb(0,0,0,0.05)] text-left w-[180px]">
                            <div className="flex items-baseline mb-1">
                                <span className="text-[40px] leading-none font-black text-gray-900 tracking-tight">0.25</span>
                                <span className="text-3xl font-bold text-[#F5B027]">s</span>
                            </div>
                            <span className="text-xs font-medium text-gray-500 leading-snug tracking-tight">Real-Time Intelligent<br />Responses</span>
                        </div>

                    </div>
                </div>

                {/* Trusted By text beneath hero */}
                <p className="mt-20 text-xs font-bold tracking-widest text-[#B4B0A8] uppercase">Trusted by forward-thinking tax professionals</p>
                <div className="flex items-center justify-center gap-8 mt-6 opacity-40 grayscale">
                    {/* Placeholder logos */}
                    <div className="flex items-center gap-2 font-black text-xl"><Zap className="w-6 h-6" /> Vortex</div>
                    <div className="flex items-center gap-2 font-black text-xl"><Shield className="w-6 h-6" /> SecureTax</div>
                    <div className="flex items-center gap-2 font-black text-xl"><FileText className="w-6 h-6" /> DocuFlow</div>
                </div>
            </section>

            {/* 3. Feature Highlight (Left Image, Right Text) */}
            <section className="px-6 py-24 mx-auto max-w-7xl">
                <div className="flex flex-col items-center gap-16 md:flex-row lg:gap-24">
                    <div className="w-full md:w-1/2">
                        <div className="relative aspect-square md:aspect-[4/3] rounded-[2.5rem] overflow-hidden shadow-xl">
                            <img
                                src={featureHighlightDashboard}
                                alt="Dashboard"
                                className="object-cover w-full h-full"
                            />
                        </div>
                    </div>
                    <div className="w-full md:w-1/2 flex flex-col items-start text-left">
                        <div className="px-3 py-1 mb-6 text-xs text-[#F5B027] font-bold uppercase tracking-wider bg-[#F5B027]/10 rounded-full">
                            Accuracy
                        </div>
                        <h2 className="mb-6 text-4xl md:text-5xl font-extrabold tracking-tight text-gray-900 leading-[1.1]">
                            Results that speak for themselves
                        </h2>

                        <div className="flex items-center gap-4 mb-6">
                            <div className="flex -space-x-3">
                                <img src="https://i.pravatar.cc/100?img=1" className="w-10 h-10 rounded-full border-2 border-[#FAF7F2] relative z-20" alt="user" />
                                <img src="https://i.pravatar.cc/100?img=2" className="w-10 h-10 rounded-full border-2 border-[#FAF7F2] relative z-10" alt="user" />
                                <img src="https://i.pravatar.cc/100?img=3" className="w-10 h-10 rounded-full border-2 border-[#FAF7F2] relative z-0" alt="user" />
                            </div>
                            <div className="flex flex-col">
                                <span className="text-sm font-bold text-gray-900 leading-tight">4.9/5 Rating</span>
                                <div className="flex items-center gap-0.5 mt-0.5">
                                    {[1, 2, 3, 4, 5].map(i => <Star key={i} className="w-3.5 h-3.5 text-[#F5B027] fill-current" />)}
                                </div>
                            </div>
                        </div>

                        <p className="mb-8 text-lg font-medium text-gray-500 leading-relaxed max-w-md">
                            Stop searching through endless PDFs. Ask a question and get a precise answer backed by exact citations from the 1961 Act in seconds.
                        </p>
                    </div>
                </div>
            </section>

            {/* 4. Bento Grid Layout */}
            <section className="px-6 py-24 mx-auto max-w-7xl">
                <div className="text-center mb-16">
                    <div className="px-3 py-1 mb-4 text-xs text-[#F5B027] font-bold uppercase tracking-wider bg-[#F5B027]/10 rounded-full inline-block">
                        Workflow
                    </div>
                    <h2 className="text-4xl md:text-5xl font-extrabold tracking-tight text-gray-900 mb-4">
                        Take Full Control of Your<br />Daily Workflow
                    </h2>
                    <p className="text-gray-500 font-medium max-w-xl mx-auto">
                        Automate bill scanning, GST extraction, and legal research inside one powerful, unified interface.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4 auto-rows-[240px]">

                    {/* Box 1 - Top Left Large (Gradient) */}
                    <div className="col-span-1 md:col-span-2 row-span-2 p-8 rounded-[2.5rem] bg-gradient-to-br from-[#FFF8E7] to-[#FFECA8] relative overflow-hidden group shadow-[inset_0_2px_10px_rgba(255,255,255,0.8),0_4px_15px_rgba(0,0,0,0.02)] transition-transform duration-500 hover:-translate-y-1">
                        <div className="absolute -right-20 -top-20 w-64 h-64 bg-gradient-to-bl from-white to-transparent rounded-full opacity-60 blur-2xl group-hover:scale-110 transition-transform duration-700" />
                        <div className="w-12 h-12 bg-white/80 backdrop-blur-sm rounded-full flex items-center justify-center text-[#F5B027] mb-6 shadow-sm relative z-10">
                            <Search className="w-6 h-6" />
                        </div>
                        <h3 className="text-[28px] font-extrabold text-gray-900 w-2/3 leading-tight mb-3 relative z-10 tracking-tight">Instant Legal Research</h3>
                        <p className="text-gray-600 font-medium text-[15px] w-2/3 relative z-10 leading-relaxed">
                            Type your natural language query and retrieve highly specific clauses instantly, completely backed by the 1961 Act.
                        </p>
                        {/* Decorative lines matching reference */}
                        <div className="absolute right-0 bottom-0 top-0 w-1/2 opacity-[0.15] mix-blend-overlay">
                            <div className="w-full h-full bg-[repeating-linear-gradient(45deg,transparent,transparent_12px,#000_12px,#000_13px)] group-hover:opacity-60 transition-opacity duration-700" />
                        </div>
                    </div>

                    {/* Box 2 - Top Right (Pink/Purple Gradient) */}
                    <div className="col-span-1 md:col-span-2 row-span-2 p-10 rounded-[2.5rem] bg-gradient-to-br from-[#FFA2CB] via-[#D3B4FF] to-[#8C84FF] relative overflow-hidden text-white flex flex-col justify-end group shadow-[inset_0_2px_15px_rgba(255,255,255,0.4),0_8px_20px_rgba(140,132,255,0.15)] transition-transform duration-500 hover:-translate-y-1">
                        <div className="absolute top-0 right-0 w-[150%] h-[150%] bg-[radial-gradient(circle_at_top_right,rgba(255,255,255,0.5),transparent_50%)] mix-blend-overlay group-hover:scale-110 transition-transform duration-1000 origin-top-right" />
                        <div className="absolute bottom-0 left-0 w-full h-1/2 bg-gradient-to-t from-black/20 to-transparent" />
                        <h3 className="text-4xl lg:text-[42px] font-black leading-[1.1] mb-2 max-w-sm relative z-10 tracking-tight text-white/95">
                            Your intentions become actionable <br />
                            <span className="inline-block px-4 py-1.5 mt-2 bg-white/95 text-[#8C84FF] rounded-full text-2xl align-middle shadow-xl font-black ml-1 backdrop-blur-md relative overflow-hidden group-hover:scale-105 transition-transform duration-300">
                                workflows <Sparkles className="inline w-5 h-5 ml-1 mb-1 text-[#F5B027]" />
                                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-50 translate-x-[-100%] group-hover:animate-[shimmer_1.5s_infinite]" />
                            </span>
                        </h3>
                    </div>

                    {/* Box 3 - Middle Left Small (White) */}
                    <div className="col-span-1 md:col-span-2 row-span-1 p-8 rounded-[2.5rem] bg-white border border-gray-100/60 shadow-[0_4px_20px_rgba(0,0,0,0.03)] flex items-center justify-between group transition-transform duration-500 hover:-translate-y-1 relative overflow-hidden">
                        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-[#F5B027] to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                        <div className="relative z-10">
                            <h3 className="text-2xl font-extrabold text-gray-900 mb-2 tracking-tight">OCR Bill Scanning</h3>
                            <p className="text-gray-500 text-[15px] font-medium">Auto-extract tax eligibility from any image.</p>
                        </div>
                        {/* Audio wave graphic */}
                        <div className="flex items-end gap-1.5 h-12 relative z-10">
                            {[2, 4, 8, 3, 5, 9, 6, 3].map((h, i) => (
                                <div key={i} className="w-2 bg-gradient-to-t from-[#F5B027] to-[#FFD57F] rounded-full transition-all duration-300 group-hover:h-full" style={{ height: `${h * 10}%` }} />
                            ))}
                        </div>
                    </div>

                    {/* Box 4 - Middle Right Small (White) */}
                    <div className="col-span-1 md:col-span-2 row-span-1 p-8 rounded-[2.5rem] bg-white border border-gray-100/60 shadow-[0_4px_20px_rgba(0,0,0,0.03)] flex items-center group transition-transform duration-500 hover:-translate-y-1 relative overflow-hidden">
                        <div className="absolute inset-x-0 bottom-0 h-1/2 bg-gradient-to-t from-gray-50 to-transparent" />
                        <div className="relative z-10">
                            <h3 className="text-2xl font-extrabold text-gray-900 mb-3 tracking-tight">Personalized AI assistant</h3>
                            <div className="flex items-center gap-3 mt-3">
                                <div className="flex -space-x-2">
                                    <img src="https://i.pravatar.cc/100?img=12" className="w-9 h-9 rounded-full border-2 border-white relative z-20 shadow-sm" alt="user" />
                                    <div className="w-9 h-9 rounded-full bg-gray-900 border-2 border-white relative z-10 flex items-center justify-center">
                                        <Sparkles className="w-4 h-4 text-[#F5B027]" />
                                    </div>
                                </div>
                                <div className="text-[13px] text-gray-500 font-medium leading-snug">Ask complex questions to<br />our fine-tuned tax engine.</div>
                            </div>
                        </div>
                    </div>

                    {/* Box 5 - Bottom Left (Purple abstract sphere) */}
                    <div className="col-span-1 md:col-span-2 row-span-1 p-8 rounded-[2.5rem] bg-[#F8F3FA] relative overflow-hidden flex flex-col justify-end group transition-transform duration-500 hover:-translate-y-1 shadow-[inset_0_2px_10px_rgba(255,255,255,0.8),0_4px_15px_rgba(0,0,0,0.02)]">
                        <div className="absolute -top-12 -right-12 w-56 h-56 rounded-full bg-gradient-to-tr from-[#D8B4E2] via-[#EACAFF] to-[#FFE4A0] blur-2xl opacity-70 group-hover:scale-125 transition-transform duration-1000" />
                        <div className="absolute top-8 left-8 w-12 h-12 bg-white/60 backdrop-blur-md rounded-2xl flex items-center justify-center shadow-sm">
                            <FileText className="w-6 h-6 text-[#B3A0FF]" />
                        </div>
                        <h3 className="text-xl font-black text-gray-900 relative z-10 uppercase tracking-widest mb-1.5 mt-12">GST Export</h3>
                        <p className="text-gray-500 text-[15px] font-medium relative z-10 max-w-xs">Generate strictly compliant master Excel sheets instantly.</p>
                    </div>

                    {/* Box 6 - Bottom Middle (Stripes) */}
                    <div className="col-span-1 md:col-span-1 row-span-1 rounded-[2.5rem] overflow-hidden bg-[#FAFAFA] relative border border-gray-100 group transition-transform duration-500 hover:-translate-y-1">
                        <div className="absolute inset-0 bg-[repeating-linear-gradient(45deg,transparent,transparent_6px,#EAEAEA_6px,#EAEAEA_7px)] group-hover:opacity-50 transition-opacity duration-300" />
                        <div className="absolute inset-0 bg-gradient-to-b from-transparent to-white/90" />
                        <div className="absolute bottom-6 left-6 right-6">
                            <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                                <div className="w-2/3 h-full bg-gray-900 rounded-full group-hover:w-full transition-all duration-1000 ease-out" />
                            </div>
                        </div>
                    </div>

                    {/* Box 7 - Bottom Right (Image Box) */}
                    <div className="col-span-1 md:col-span-1 row-span-1 rounded-[2.5rem] overflow-hidden relative group transition-transform duration-500 hover:-translate-y-1">
                        <img
                            src="https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=600&auto=format&fit=crop"
                            alt="Feature"
                            className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700 origin-bottom"
                        />
                        <div className="absolute inset-x-0 bottom-0 p-8 bg-gradient-to-t from-black/90 via-black/40 to-transparent text-white pt-24">
                            <h3 className="text-sm font-black uppercase tracking-widest mb-1.5 text-white/90">Scale Operations</h3>
                            <p className="text-[13px] font-medium text-white/60 leading-relaxed">Built for high-volume firms.</p>
                        </div>
                    </div>
                </div>
            </section>

            {/* 5. Gold CTA Banner */}
            <section className="px-6 py-12 mx-auto max-w-7xl">
                <div className="w-full rounded-[3rem] bg-gradient-to-r from-[#FFF5DC] via-[#FFE29F] to-[#FFCF54] p-16 md:p-24 text-center relative overflow-hidden">
                    <h2 className="text-4xl md:text-5xl font-extrabold text-[#111111] leading-tight mb-8 relative z-10 max-w-2xl mx-auto">
                        Excited to try it out? See how Legal Lens AI works.
                    </h2>
                    <Link to="/chat" className="inline-flex items-center gap-2 px-8 py-4 font-bold text-white transition-transform bg-gray-900 rounded-full hover:-translate-y-1 shadow-xl relative z-10">
                        Try It Now
                        <div className="flex items-center justify-center w-6 h-6 rounded-full bg-[#F5B027] text-white">
                            <ArrowRight className="w-3 h-3" />
                        </div>
                    </Link>
                </div>
            </section>

            {/* 6. Pricing Section */}
            <section className="px-6 py-24 mx-auto max-w-7xl">
                <div className="text-center mb-16">
                    <div className="px-3 py-1 mb-4 text-xs text-[#F5B027] font-bold uppercase tracking-wider bg-[#F5B027]/10 rounded-full inline-block">
                        Pricing
                    </div>
                    <h2 className="text-4xl md:text-5xl font-extrabold tracking-tight text-gray-900">
                        Flexible Pricing For Every<br />User Needs
                    </h2>
                </div>

                <div className="flex flex-col items-center justify-center gap-6 lg:flex-row lg:items-center">
                    {/* Basic Tier (Light) */}
                    <div className="w-full max-w-sm p-8 bg-white border border-gray-100 rounded-[2.5rem] shadow-sm">
                        <h3 className="text-xl font-bold text-gray-900 mb-2">Basic</h3>
                        <div className="flex items-baseline gap-1 mb-6">
                            <span className="text-4xl font-black text-gray-900">USD $10</span>
                            <span className="text-gray-400 font-medium">/mo</span>
                        </div>
                        <p className="text-sm font-bold text-gray-900 mb-4">What's included in Basic:</p>
                        <ul className="space-y-4 mb-8">
                            <li className="flex items-center gap-3 text-sm font-medium text-gray-600"><Check className="w-4 h-4 text-[#F5B027]" /> 100 Chat Queries</li>
                            <li className="flex items-center gap-3 text-sm font-medium text-gray-600"><Check className="w-4 h-4 text-[#F5B027]" /> 50 Bill Scans</li>
                            <li className="flex items-center gap-3 text-sm font-medium text-gray-600"><Check className="w-4 h-4 text-[#F5B027]" /> Standard Support</li>
                        </ul>
                        <button className="w-full py-3.5 rounded-xl font-bold text-gray-900 border-2 border-gray-100 hover:border-gray-200 transition-colors">
                            Get Started
                        </button>
                    </div>

                    {/* Pro Tier (Dark - Center) */}
                    <div className="w-full max-w-sm p-1 bg-gradient-to-b from-[#F5B027] to-[#FFE29F] rounded-[2.5rem] shadow-2xl relative z-10 transform lg:scale-105">
                        <div className="p-8 bg-gray-900 rounded-[2.3rem] h-full text-white">
                            <div className="flex justify-between items-start mb-6">
                                <div>
                                    <h3 className="text-xl font-bold text-white mb-2">Pro</h3>
                                    <div className="flex items-baseline gap-1">
                                        <span className="text-4xl font-black text-white">USD $29</span>
                                        <span className="text-gray-400 font-medium">/mo</span>
                                    </div>
                                </div>
                                <div className="px-3 py-1 text-[10px] font-black uppercase tracking-wider text-gray-900 bg-[#F5B027] rounded-full">
                                    Popular
                                </div>
                            </div>

                            <p className="text-sm font-bold text-white mb-4">What's included in Pro:</p>
                            <ul className="space-y-4 mb-8">
                                <li className="flex items-center gap-3 text-sm font-medium text-gray-300"><div className="w-4 h-4 rounded-full bg-[#F5B027] flex items-center justify-center"><Check className="w-3 h-3 text-gray-900" /></div> Unlimited Chats</li>
                                <li className="flex items-center gap-3 text-sm font-medium text-gray-300"><div className="w-4 h-4 rounded-full bg-[#F5B027] flex items-center justify-center"><Check className="w-3 h-3 text-gray-900" /></div> 500 GST Invoices</li>
                                <li className="flex items-center gap-3 text-sm font-medium text-gray-300"><div className="w-4 h-4 rounded-full bg-[#F5B027] flex items-center justify-center"><Check className="w-3 h-3 text-gray-900" /></div> Excel Export</li>
                                <li className="flex items-center gap-3 text-sm font-medium text-gray-300"><div className="w-4 h-4 rounded-full bg-[#F5B027] flex items-center justify-center"><Check className="w-3 h-3 text-gray-900" /></div> Priority Support</li>
                            </ul>
                            <div className="p-3 mb-6 bg-gray-800 rounded-xl">
                                <p className="text-xs text-center text-gray-400 font-medium">✨ Unlimited access to latest amendments</p>
                            </div>
                            <button className="w-full py-3.5 rounded-xl font-bold text-gray-900 bg-[#F5B027] hover:bg-[#FFC64B] transition-colors shadow-lg shadow-[#F5B027]/20">
                                Upgrade to Pro
                            </button>
                        </div>
                    </div>

                    {/* Enterprise Tier (Light) */}
                    <div className="w-full max-w-sm p-8 bg-white border border-gray-100 rounded-[2.5rem] shadow-sm">
                        <h3 className="text-xl font-bold text-gray-900 mb-2">Enterprise</h3>
                        <div className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-100 to-indigo-100 mb-6 flex items-center justify-center">
                            <Sparkles className="w-6 h-6 text-indigo-500" />
                        </div>
                        <h4 className="text-lg font-bold text-gray-900 mb-2">Custom Pricing</h4>
                        <p className="text-sm font-medium text-gray-500 mb-8 leading-relaxed">
                            For large accounting firms requiring API access, custom OCR models, and dedicated account management.
                        </p>
                        <button className="w-full py-3.5 rounded-xl font-bold text-white bg-gray-900 hover:bg-gray-800 transition-colors">
                            Contact Us
                        </button>
                    </div>
                </div>
            </section>

            {/* 8. FAQ Section */}
            <section className="px-6 py-24 mx-auto max-w-7xl">
                <div className="flex flex-col lg:flex-row gap-16">
                    <div className="w-full lg:w-1/3">
                        <div className="px-3 py-1 mb-4 text-xs text-[#F5B027] font-bold uppercase tracking-wider bg-[#F5B027]/10 rounded-full inline-block">
                            FAQ
                        </div>
                        <h2 className="text-4xl md:text-5xl font-extrabold tracking-tight text-gray-900 mb-6 leading-tight">
                            Frequently Asked Questions
                        </h2>
                        <p className="text-gray-500 font-medium mb-8">
                            Find answers to common questions about Legal Lens and our features.
                        </p>
                        <button className="px-6 py-3 font-bold text-white transition-colors bg-gray-900 rounded-full hover:bg-gray-800">
                            Drop a Message
                        </button>
                    </div>

                    <div className="w-full lg:w-2/3 flex flex-col gap-4">
                        {faqs.map((faq, idx) => (
                            <div
                                key={idx}
                                className={`p-6 rounded-[2rem] border transition-all cursor-pointer ${openFaq === idx ? 'bg-white border-gray-200 shadow-sm' : 'bg-[#FAFAFA] border-transparent hover:border-gray-200'}`}
                                onClick={() => setOpenFaq(idx === openFaq ? -1 : idx)}
                            >
                                <div className="flex items-center justify-between">
                                    <h3 className="font-bold text-gray-900">{faq.q}</h3>
                                    <div className={`flex items-center justify-center w-8 h-8 rounded-full border border-gray-200 transition-transform ${openFaq === idx ? 'rotate-180 bg-gray-900 text-white' : 'bg-white text-gray-400'}`}>
                                        <ChevronDown className="w-4 h-4" />
                                    </div>
                                </div>
                                {openFaq === idx && (
                                    <p className="mt-4 text-sm font-medium text-gray-500 leading-relaxed max-w-2xl">
                                        {faq.a}
                                    </p>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* 9. Footer CTA */}
            <section className="mt-12 bg-[#FEF3C7] rounded-t-[3rem] px-6 py-20 md:py-32 w-full">
                <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-12">
                    <div>
                        <div className="flex items-center gap-2 mb-6">
                            <div className="flex items-center justify-center w-8 h-8 rounded-full bg-gradient-to-tr from-gray-900 to-gray-700">
                                <Sparkles className="w-4 h-4 text-white" />
                            </div>
                            <span className="text-lg font-bold tracking-tight">Legal Lens</span>
                        </div>
                        <h2 className="text-5xl md:text-6xl font-black tracking-tight text-gray-900 leading-[1.1] max-w-sm mb-6">
                            Ready to see<br />Legal Lens in action?
                        </h2>
                        <Link to="/chat" className="inline-flex items-center gap-2 px-8 py-4 font-bold text-white transition-transform bg-gray-900 rounded-full hover:-translate-y-1 shadow-xl">
                            Try It Free
                        </Link>
                    </div>

                    {/* Footer Links Grid */}
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-12 md:gap-24 text-sm">
                        <div className="flex flex-col gap-4">
                            <h4 className="font-bold text-gray-900 uppercase tracking-wider mb-2">Product</h4>
                            <a href="#" className="font-semibold text-gray-500 hover:text-gray-900">Features</a>
                            <a href="#" className="font-semibold text-gray-500 hover:text-gray-900">Integrations</a>
                            <a href="#" className="font-semibold text-gray-500 hover:text-gray-900">Pricing</a>
                            <a href="#" className="font-semibold text-gray-500 hover:text-gray-900">Changelog</a>
                        </div>
                        <div className="flex flex-col gap-4">
                            <h4 className="font-bold text-gray-900 uppercase tracking-wider mb-2">Company</h4>
                            <a href="#" className="font-semibold text-gray-500 hover:text-gray-900">About Us</a>
                            <a href="#" className="font-semibold text-gray-500 hover:text-gray-900">Careers</a>
                            <a href="#" className="font-semibold text-gray-500 hover:text-gray-900">Blog</a>
                            <a href="#" className="font-semibold text-gray-500 hover:text-gray-900">Contact</a>
                        </div>
                        <div className="flex flex-col gap-4">
                            <h4 className="font-bold text-gray-900 uppercase tracking-wider mb-2">Legal</h4>
                            <a href="#" className="font-semibold text-gray-500 hover:text-gray-900">Privacy Policy</a>
                            <a href="#" className="font-semibold text-gray-500 hover:text-gray-900">Terms of Service</a>
                            <a href="#" className="font-semibold text-gray-500 hover:text-gray-900">Cookie Policy</a>
                        </div>
                    </div>
                </div>

                <div className="max-w-7xl mx-auto mt-24 pt-8 border-t border-gray-900/10 flex flex-col md:flex-row items-center justify-between gap-6">
                    <p className="font-semibold text-gray-500 text-sm">© 2026 Legal Lens Inc.</p>
                    <div className="w-12 h-12 bg-[#F5B027] rounded-full flex items-center justify-center cursor-pointer hover:scale-110 transition-transform">
                        <ArrowRight className="w-5 h-5 text-white" />
                    </div>
                </div>
            </section>

        </div>
    );
};

export default LandingPage;
