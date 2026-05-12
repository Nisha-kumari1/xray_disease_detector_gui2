"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, BrainCircuit, Send, User, Bot, AlertTriangle, CheckCircle, Info, Scan, Activity, FileStack, ChevronRight } from "lucide-react";

export default function Home() {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  // Chatbot states
  const [chatMessages, setChatMessages] = useState([
    { role: 'assistant', content: 'Welcome to AuraRad. Upload a Chest X-Ray and I will help you interpret the clinical findings.' }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileUpload = async (uploadedFile: File) => {
    setFile(uploadedFile);
    setAnalyzing(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", uploadedFile);

    try {
      const response = await fetch("http://localhost:8000/inference/upload_and_predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to analyze image. Please ensure the backend is running.");
      }

      const data = await response.json();
      setResult(data);
      
      setChatMessages(prev => [...prev, {
        role: 'assistant', 
        content: `Analysis complete. Diagnosis: **${data.diagnosis}**. Feel free to ask me for a detailed breakdown or recommended next steps.`
      }]);

    } catch (err: any) {
      console.error(err);
      setError(err.message || "An error occurred during analysis.");
    } finally {
      setAnalyzing(false);
    }
  };

  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim()) return;
    
    const userMsg = chatInput;
    setChatMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setChatInput('');
    setIsChatLoading(true);
    
    try {
      const response = await fetch("http://localhost:8000/chat/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMsg })
      });
      
      if (!response.ok) throw new Error("Chat request failed");
      
      const data = await response.json();
      setChatMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
    } catch (err) {
      setChatMessages(prev => [...prev, { role: 'assistant', content: "Connection to the AI brain interrupted. Please ensure the backend is running." }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full space-y-6">
      {/* Header */}
      <header className="flex justify-between items-center glass-panel rounded-3xl p-5 shrink-0 relative overflow-hidden">
        <div className="flex items-center gap-4 relative z-10">
          <div className="w-12 h-12 rounded-2xl bg-blue-600 flex items-center justify-center text-white shadow-md">
            <BrainCircuit size={28} />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-slate-900">Aura<span className="text-blue-600">Rad</span></h1>
            <p className="text-sm text-slate-500 font-medium tracking-wide uppercase">Clinical Intelligence System</p>
          </div>
        </div>
        
        <div className="flex gap-3 relative z-10">
           <button className="flex items-center gap-2 px-4 py-2 bg-white hover:bg-slate-50 border border-slate-200 rounded-full text-sm font-medium text-slate-600 transition-colors shadow-sm">
             <Activity size={16} className="text-emerald-500" /> System Online
           </button>
           <div className="w-10 h-10 rounded-full bg-slate-200 border border-slate-300 overflow-hidden ml-2 shadow-sm">
             <img src="https://i.pravatar.cc/150?img=32" alt="Doctor" className="w-full h-full object-cover" />
           </div>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col lg:flex-row gap-6 min-h-0">
        
        {/* Left Side: X-Ray & Results */}
        <div className="flex-1 flex flex-col gap-6 overflow-y-auto pr-2 custom-scrollbar">
          
          {/* Upload Section - Collapses slightly if result exists */}
          <motion.div 
             layout
             className={`glass-card shrink-0 transition-all duration-500 ${result ? 'p-4' : 'p-8'}`}
          >
            {!result && (
              <h2 className="text-xl font-semibold text-slate-800 flex items-center gap-2 mb-6">
                <Upload className="w-6 h-6 text-blue-600" /> Initialize Scan Analysis
              </h2>
            )}
            
            <div 
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-2xl flex flex-col items-center justify-center transition-all duration-300 relative overflow-hidden group ${
                result ? 'h-24 p-4' : 'p-12 h-64'
              } ${
                isDragging ? "border-blue-500 bg-blue-50" : "border-slate-300 bg-slate-50/50 hover:bg-slate-50 hover:border-blue-300"
              }`}
            >
              <input 
                type="file" 
                className="hidden" 
                id="file-upload" 
                accept="image/*"
                onChange={(e) => e.target.files && handleFileUpload(e.target.files[0])}
              />
              <label 
                htmlFor="file-upload" 
                className="flex flex-col items-center cursor-pointer w-full h-full justify-center z-10"
              >
                {!result ? (
                  <>
                    <div className="w-20 h-20 rounded-full bg-white border border-slate-200 shadow-sm flex items-center justify-center mb-5 group-hover:scale-110 group-hover:bg-blue-50 group-hover:border-blue-200 transition-all duration-300">
                      <Scan className={`w-10 h-10 ${isDragging ? "text-blue-600" : "text-slate-400 group-hover:text-blue-600"}`} />
                    </div>
                    <h3 className="text-lg font-medium text-slate-800">Drag & Drop DICOM or Image</h3>
                    <p className="text-slate-500 text-sm mt-2 font-mono">Supported formats: .dcm, .png, .jpg</p>
                  </>
                ) : (
                  <div className="flex items-center gap-4">
                     <FileStack className="w-8 h-8 text-slate-400" />
                     <div className="text-left">
                       <h3 className="text-base font-medium text-slate-700">Analyze New Scan</h3>
                       <p className="text-slate-500 text-xs">Drop file here to override current</p>
                     </div>
                  </div>
                )}
              </label>
            </div>
            
            {error && (
              <motion.div initial={{opacity:0, y:10}} animate={{opacity:1, y:0}} className="mt-4 p-4 bg-red-50 border border-red-200 text-red-700 rounded-xl flex items-center gap-3 text-sm">
                <AlertTriangle className="w-5 h-5 shrink-0" /> {error}
              </motion.div>
            )}
          </motion.div>

          {/* Analysis View */}
          <AnimatePresence mode="wait">
            {(analyzing || result) && (
              <motion.div 
                key={analyzing ? 'analyzing' : 'result'}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="glass-card p-6 flex-1 flex flex-col"
              >
                {analyzing ? (
                  <div className="flex-1 flex flex-col items-center justify-center py-20 relative">
                    <motion.div 
                      animate={{ rotate: 360 }}
                      transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                      className="w-20 h-20 border-4 border-slate-200 border-t-blue-600 rounded-full mb-6 z-10"
                    />
                    <h3 className="text-2xl font-bold text-slate-800 tracking-wide">Executing Neural Analysis</h3>
                    <p className="text-slate-500 mt-3 font-mono text-sm uppercase tracking-widest">Generating Grad-CAM Heatmaps</p>
                  </div>
                ) : result && (
                  <div className="space-y-8">
                    
                    {/* Diagnosis Banner */}
                    <div className={`relative overflow-hidden rounded-2xl p-6 border ${
                        result.diagnosis === 'Pneumonia' ? 'bg-red-50/50 border-red-200' : 'bg-emerald-50/50 border-emerald-200'
                    }`}>
                      <div className="flex items-center justify-between relative z-10">
                        <div>
                          <p className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-2">AI Diagnostic Assessment</p>
                          <h2 className={`text-4xl font-black flex items-center gap-3 tracking-tight ${
                            result.diagnosis === 'Pneumonia' ? 'text-red-700' : 'text-emerald-700'
                          }`}>
                            {result.diagnosis === 'Pneumonia' ? <AlertTriangle size={36} /> : <CheckCircle size={36} />}
                            {result.diagnosis} Detected
                          </h2>
                        </div>
                        <div className="text-right">
                          <p className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-2">Confidence Level</p>
                          <p className={`text-5xl font-black tracking-tighter ${
                            result.diagnosis === 'Pneumonia' ? 'text-red-700' : 'text-emerald-700'
                          }`}>
                            {result.probabilities[result.diagnosis]}<span className="text-2xl">%</span>
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Explainable AI Visuals */}
                    <div>
                      <div className="flex items-center justify-between mb-4">
                         <h3 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
                           <Info className="w-5 h-5 text-blue-600" /> Visual Explanation
                         </h3>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Original Image */}
                        <div className="bg-slate-900 rounded-2xl overflow-hidden aspect-square border border-slate-800 relative group shadow-md">
                          <img src={result.original_url} alt="Original X-Ray" className="w-full h-full object-contain" />
                          <div className="absolute top-4 left-4 px-3 py-1.5 bg-black/70 backdrop-blur-md text-white text-xs font-bold uppercase tracking-wider rounded-lg border border-slate-700">
                            Original X-Ray
                          </div>
                        </div>
                        
                        {/* Heatmap Image */}
                        <div className="bg-slate-900 rounded-2xl overflow-hidden aspect-square border border-slate-800 relative group shadow-md">
                          <img src={result.heatmap_url} alt="AI Heatmap" className="w-full h-full object-contain" />
                          <div className="absolute top-4 left-4 px-3 py-1.5 bg-blue-600/90 backdrop-blur-md text-white text-xs font-bold uppercase tracking-wider rounded-lg shadow-sm">
                            AI Heatmap (Grad-CAM)
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Clinical Insights */}
                    <div className="bg-blue-50/50 border border-blue-100 rounded-2xl p-6 relative overflow-hidden">
                      <h3 className="text-base font-bold text-blue-900 mb-3 uppercase tracking-wider text-sm flex items-center gap-2">
                        <ChevronRight className="w-4 h-4 text-blue-600" /> What does this mean?
                      </h3>
                      <p className="text-blue-800/80 leading-relaxed text-base whitespace-pre-wrap">
                        {result.report}
                      </p>
                    </div>

                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Right Side: Integrated Chatbot */}
        <div className="lg:w-[450px] shrink-0 glass-card flex flex-col overflow-hidden border-t-4 border-t-slate-800">
          {/* Chat Header */}
          <div className="bg-slate-900 p-5 flex items-center justify-between text-white shrink-0">
            <div className="flex items-center gap-4">
              <div className="bg-blue-600 p-2.5 rounded-xl shadow-sm">
                <Bot size={22} className="text-white" />
              </div>
              <div>
                <h3 className="font-bold text-lg tracking-tight">Medical Assistant</h3>
                <div className="flex items-center gap-2 mt-0.5">
                  <p className="text-xs text-slate-300 font-medium">Ask questions in plain English</p>
                </div>
              </div>
            </div>
          </div>
          
          {/* Chat Messages */}
          <div className="flex-1 overflow-y-auto p-5 space-y-6 bg-white custom-scrollbar">
            {chatMessages.map((msg, idx) => (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                key={idx} 
                className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
              >
                <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-1 shadow-sm ${
                  msg.role === 'user' ? 'bg-blue-600' : 'bg-slate-100 border border-slate-200'
                }`}>
                  {msg.role === 'user' ? <User size={14} className="text-white" /> : <Bot size={14} className="text-slate-600" />}
                </div>
                <div className={`px-4 py-3 rounded-2xl text-[15px] leading-relaxed max-w-[85%] shadow-sm ${
                  msg.role === 'user' 
                    ? 'bg-blue-600 text-white rounded-tr-sm' 
                    : 'bg-white border border-slate-200 text-slate-700 rounded-tl-sm'
                }`}>
                  {/* Basic markdown bold support for diagnosis */}
                  {msg.content.split('**').map((part, i) => i % 2 === 1 ? <strong key={i} className="text-slate-900">{part}</strong> : part)}
                </div>
              </motion.div>
            ))}
            
            {isChatLoading && (
              <div className="flex gap-3">
                <div className="w-8 h-8 rounded-full bg-slate-100 border border-slate-200 flex items-center justify-center shrink-0 mt-1">
                  <Bot size={14} className="text-slate-600" />
                </div>
                <div className="px-5 py-4 bg-white border border-slate-200 rounded-2xl rounded-tl-sm shadow-sm flex items-center gap-1.5">
                  <motion.div animate={{y: [0, -4, 0], opacity:[0.5,1,0.5]}} transition={{repeat: Infinity, duration: 1}} className="w-2 h-2 bg-slate-400 rounded-full" />
                  <motion.div animate={{y: [0, -4, 0], opacity:[0.5,1,0.5]}} transition={{repeat: Infinity, duration: 1, delay: 0.2}} className="w-2 h-2 bg-slate-400 rounded-full" />
                  <motion.div animate={{y: [0, -4, 0], opacity:[0.5,1,0.5]}} transition={{repeat: Infinity, duration: 1, delay: 0.4}} className="w-2 h-2 bg-slate-400 rounded-full" />
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
          
          {/* Chat Input */}
          <div className="p-4 bg-slate-50 shrink-0 border-t border-slate-200">
            <form onSubmit={handleChatSubmit} className="flex gap-2 relative">
              <input 
                type="text" 
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Ask 'What causes Pneumonia?'" 
                className="flex-1 pl-4 pr-12 py-3.5 bg-white border border-slate-200 focus:border-blue-500 rounded-xl text-sm text-slate-900 placeholder-slate-400 outline-none transition-all shadow-sm"
                disabled={isChatLoading}
              />
              <button 
                type="submit" 
                disabled={!chatInput.trim() || isChatLoading}
                className="absolute right-2 top-2 bottom-2 w-10 flex items-center justify-center bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:bg-slate-300 disabled:text-slate-500 disabled:cursor-not-allowed transition-colors"
              >
                <Send size={18} className={chatInput.trim() && !isChatLoading ? "ml-1" : ""} />
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
