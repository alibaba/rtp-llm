import React, { useEffect, useRef } from 'react'
import Header from './components/Header/index.jsx'
import Hero from './components/Hero/index.jsx'
import Features from './components/Features/index.jsx'
import AIStats from './components/AIStats/index.jsx'
import AIFeatures from './components/AIFeatures/index.jsx'
import BrandShowcase from './components/BrandShowcase/index.tsx'
import Footer from './components/Footer/index.tsx'
import { LanguageProvider } from './hooks/useLanguage.jsx'
import './styles/App.css'

function App() {
  const heroRef = useRef(null);
  const featuresRef = useRef(null);
  const isScrollingRef = useRef(false);

  useEffect(() => {
    const handleWheel = (e) => {
      // 防止重复触发滚动
      if (isScrollingRef.current) return;
      
      // 判断滚动方向
      if (e.deltaY > 0) {
        // 向下滚动，从Hero切换到Features
        const heroElement = heroRef.current;
        if (heroElement && heroElement.getBoundingClientRect().bottom > window.innerHeight * 0.5) {
          isScrollingRef.current = true;
          featuresRef.current?.scrollIntoView({ behavior: 'smooth' });
          
          // 重置滚动状态
          setTimeout(() => {
            isScrollingRef.current = false;
          }, 1000);
        }
      }
    };

    // 添加滚动事件监听器
    window.addEventListener('wheel', handleWheel, { passive: false });
    
    // 清理事件监听器
    return () => {
      window.removeEventListener('wheel', handleWheel);
    };
  }, []);

  return (
    <LanguageProvider>
      <div className="app">
        <div className="container">
          <Header />
          <div className="main-content">
            <div ref={heroRef}>
              <Hero />
            </div>
            <div ref={featuresRef}>
              <Features />
            </div>
            <AIStats />
            <AIFeatures />
            <BrandShowcase />
            <Footer />
          </div>
        </div>
      </div>
    </LanguageProvider>
  );
}

export default App;