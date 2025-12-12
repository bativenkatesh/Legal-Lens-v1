import React from 'react';
import { Link } from 'react-router-dom';
import { Sparkles, ArrowRight, Shield, Zap, Search } from 'lucide-react';
import heroScreenshot from '../assets/hero-screenshot.png';

const LandingPage = () => {
    return (
        <div className="min-h-screen bg-background text-foreground flex flex-col overflow-hidden">
            {/* Background Gradients */}
            <div className="fixed inset-0 z-0 pointer-events-none">
                <div className="absolute top-[-20%] right-[-10%] w-[800px] h-[800px] rounded-full bg-primary/5 blur-[120px]" />
                <div className="absolute bottom-[-20%] left-[-10%] w-[600px] h-[600px] rounded-full bg-blue-500/5 blur-[100px]" />
            </div>

            {/* Navigation */}
            <nav className="relative z-10 w-full py-6 px-6 md:px-8 flex justify-between items-center max-w-7xl mx-auto">
                <div className="flex items-center gap-2">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-blue-600 flex items-center justify-center text-white shadow-lg shadow-primary/20">
                        <Sparkles className="w-5 h-5" />
                    </div>
                    <span className="text-xl font-bold tracking-tight">Legal Lens</span>
                </div>
            </nav>

            {/* Hero Section */}
            <main className="relative z-10 flex-1 flex flex-col items-center px-4 pt-12 md:pt-20 pb-20 max-w-7xl mx-auto w-full">

                {/* Badge */}
                <div className="mb-8 animate-fade-in-up">
                    <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-secondary/50 border border-border/50 backdrop-blur-sm text-sm font-medium text-secondary-foreground">
                        <Sparkles className="w-4 h-4 text-primary" />
                        <span>AI-Powered Tax Assistant</span>
                    </div>
                </div>

                {/* Headline (Restored) */}
                <h1 className="text-5xl md:text-7xl lg:text-7xl font-bold tracking-tight mb-8 text-center max-w-5xl bg-clip-text text-transparent bg-gradient-to-b from-foreground to-foreground/70 leading-[1.1]">
                    Master the Income Tax Act <br /> with AI Precision
                </h1>

                {/* Subheadline (Restored) */}
                <p className="text-lg md:text-xl text-muted-foreground mb-10 text-center max-w-2xl leading-relaxed">
                    Navigate complex tax laws effortlessly. Get instant, accurate answers backed by the Income Tax Act, 1961, powered by advanced RAG technology.
                </p>

                {/* CTA Buttons */}
                <div className="flex flex-col sm:flex-row gap-4 w-full justify-center mb-20">
                    <Link
                        to="/chat"
                        className="group px-8 py-4 rounded-full bg-primary text-primary-foreground text-lg font-semibold hover:bg-primary/90 transition-all flex items-center justify-center gap-2 shadow-xl shadow-primary/25 hover:shadow-2xl hover:shadow-primary/30 hover:-translate-y-0.5"
                    >
                        Get Started Now
                        <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                    </Link>
                </div>

                {/* Hero Image / Screenshot Showcase */}
                <div className="relative w-full max-w-4xl mx-auto perspective-1000 group mb-24">
                    {/* Glow effect behind image */}
                    <div className="absolute inset-0 bg-gradient-to-t from-primary/20 to-transparent blur-3xl -z-10 opacity-60 rounded-[3rem]" />

                    <div className="relative rounded-2xl overflow-hidden border border-border/50 shadow-2xl bg-card/50 backdrop-blur-xl transform transition-all duration-700 hover:scale-[1.01] hover:shadow-[0_20px_50px_rgba(0,0,0,0.12)]">
                        {/* Browser Chrome */}
                        <div className="h-10 bg-muted/50 border-b border-border/50 flex items-center px-4 gap-2">
                            <div className="flex gap-1.5">
                                <div className="w-3 h-3 rounded-full bg-red-500/20 border border-red-500/50" />
                                <div className="w-3 h-3 rounded-full bg-yellow-500/20 border border-yellow-500/50" />
                                <div className="w-3 h-3 rounded-full bg-green-500/20 border border-green-500/50" />
                            </div>
                            <div className="ml-4 flex-1 max-w-xl h-6 rounded-md bg-background/50 border border-border/30 flex items-center justify-center text-[10px] text-muted-foreground font-mono">
                                legallens.ai/chat
                            </div>
                        </div>

                        {/* Actual Screenshot */}
                        <img
                            src={heroScreenshot}
                            alt="Legal Lens Interface"
                            className="w-full h-auto object-cover"
                        />

                        {/* Overlay Gradient for depth */}
                        <div className="absolute inset-0 pointer-events-none bg-gradient-to-t from-background/20 to-transparent" />
                    </div>
                </div>

                {/* New Section: Why Choose Legal Lens */}
                <div className="w-full max-w-5xl mx-auto text-center">
                    <h2 className="text-3xl font-bold mb-12">Why Legal Lens?</h2>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
                        <div className="flex flex-col items-center">
                            <div className="w-12 h-12 rounded-full bg-blue-500/10 flex items-center justify-center text-blue-500 mb-4">
                                <Search className="w-6 h-6" />
                            </div>
                            <h3 className="text-lg font-semibold mb-2">Instant Answers</h3>
                            <p className="text-muted-foreground text-sm leading-relaxed">
                                Stop searching through endless PDFs. Ask a question and get a precise answer in seconds.
                            </p>
                        </div>
                        <div className="flex flex-col items-center">
                            <div className="w-12 h-12 rounded-full bg-purple-500/10 flex items-center justify-center text-purple-500 mb-4">
                                <Zap className="w-6 h-6" />
                            </div>
                            <h3 className="text-lg font-semibold mb-2">Always Up-to-Date</h3>
                            <p className="text-muted-foreground text-sm leading-relaxed">
                                Our database is constantly updated with the latest amendments and notifications.
                            </p>
                        </div>
                        <div className="flex flex-col items-center">
                            <div className="w-12 h-12 rounded-full bg-green-500/10 flex items-center justify-center text-green-500 mb-4">
                                <Shield className="w-6 h-6" />
                            </div>
                            <h3 className="text-lg font-semibold mb-2">Reliable Sources</h3>
                            <p className="text-muted-foreground text-sm leading-relaxed">
                                Every response is backed by citations from the official Income Tax Act, 1961.
                            </p>
                        </div>
                    </div>
                </div>

            </main>

            {/* Footer */}
            <footer className="py-12 border-t border-border/50 bg-card/30 backdrop-blur-sm">
                <div className="max-w-7xl mx-auto px-8 flex flex-col md:flex-row justify-between items-center gap-6">
                    <div className="flex items-center gap-2">
                        <div className="w-6 h-6 rounded-lg bg-primary flex items-center justify-center text-white">
                            <Sparkles className="w-3 h-3" />
                        </div>
                        <span className="font-bold">Legal Lens</span>
                    </div>
                    <div className="text-sm text-muted-foreground">
                        © 2024 Legal Lens Inc. All rights reserved.
                    </div>
                    <div className="flex gap-6 text-sm font-medium text-muted-foreground">
                        <a href="#" className="hover:text-foreground transition-colors">Privacy</a>
                        <a href="#" className="hover:text-foreground transition-colors">Terms</a>
                        <a href="#" className="hover:text-foreground transition-colors">Twitter</a>
                    </div>
                </div>
            </footer>
        </div>
    );
};

export default LandingPage;
