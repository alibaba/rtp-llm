import React from 'react'
import { useLanguage } from '../../hooks/useLanguage.jsx'
import './index.css'

const Header = () => {
  const { toggleLanguage, t } = useLanguage();

  const handleGithubClick = () => {
    window.open('https://github.com/alibaba/rtp-llm', '_blank');
  };

  return (
    <div className="header">
      <div className="logo-section">
        <img
          src="https://img.alicdn.com/imgextra/i2/O1CN01f7thwU1QKoaaZnNxG_!!6000000001958-2-tps-4800-1200.png"
          className="logo-icon"
          alt="Logo"
        />
      </div>
      <div className="header-buttons">
        <div className="github-button" onClick={handleGithubClick}>
          <img 
            src="https://img.alicdn.com/imgextra/i1/6000000007725/O1CN01Mau9v526w6hT5D93O_!!6000000007725-2-gg_dtc.png" 
            className="github-icon"
            alt="GitHub"
          />
          <span className="github-text">{t('github')}</span>
        </div>
        <div className="lang-button" onClick={toggleLanguage}>
          <img 
            src="https://img.alicdn.com/imgextra/i2/6000000007361/O1CN01gSAEUV24FOX5DKqfO_!!6000000007361-2-gg_dtc.png" 
            className="lang-icon"
            alt="Language"
          />
          <span className="lang-text">{t('language')}</span>
        </div>
      </div>
    </div>
  );
};

export default Header;