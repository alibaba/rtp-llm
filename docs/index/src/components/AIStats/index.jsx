import React from 'react';
import { useLanguage } from '../../hooks/useLanguage.jsx';
import './index.css';

const AIStats = () => {
  const { t } = useLanguage();

  return (
    <div className="ai-stats-section">
      <div className="ai-stats-background">
        <video 
          src="https://cloud.video.taobao.com/vod/fETjRv18nUqPoC7DYQeAMh4zEz5vo0NkyIxmsvvfLds.mp4" 
          className="ai-stats-bg-image"
          autoPlay
          loop
          muted
          playsInline
        />
      </div>
      <div className="ai-stats-content">
        <div className="ai-stats-title-text">{t('trustedBy')}</div>
        <div className="ai-stats-grid">
          <div className="ai-stat-item">
            <span className="ai-stat-number">20+</span>
            <span className="ai-stat-label">{t('supportedModels')}</span>
          </div>
          <div className="ai-stat-item">
            <span className="ai-stat-number">Millions</span>
            <span className="ai-stat-label">{t('dailyUsers')}</span>
          </div>
          <div className="ai-stat-item">
            <span className="ai-stat-number">Production</span>
            <span className="ai-stat-label">{t('battleTested')}</span>
          </div>
          <div className="ai-stat-item">
            <span className="ai-stat-number">Open Source</span>
            <span className="ai-stat-label">{t('communityDriven')}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AIStats;