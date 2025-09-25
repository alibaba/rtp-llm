import React from 'react';
import { useLanguage } from '../../hooks/useLanguage.jsx';
import './index.css';

const AIFeatures = () => {
  const { t } = useLanguage();

  const featuresData = [
    {
      id: 'performance',
      icon: 'https://img.alicdn.com/imgextra/i4/6000000007611/O1CN01El6TPB265tY5sG9X1_!!6000000007611-2-gg_dtc.png',
      title: t('performanceFeature'),
      features: t('performanceFeatures')
    },
    {
      id: 'scalability',
      icon: 'https://img.alicdn.com/imgextra/i4/6000000005893/O1CN01egXigI1tP36l5l4ci_!!6000000005893-2-gg_dtc.png',
      title: t('scalability'),
      features: t('scalabilityFeatures')
    },
    {
      id: 'flexibility',
      icon: 'https://img.alicdn.com/imgextra/i2/6000000007471/O1CN01eAN3zM253m6ECLK8T_!!6000000007471-2-gg_dtc.png',
      title: t('flexibility'),
      features: t('flexibilityFeatures')
    }
  ];

  return (
    <div className="ai-features-section">
      <div className="ai-features-title-text">{t('advancedFeatures')}</div>
      <div className="ai-features-grid">
        {featuresData.map((feature) => (
          <div key={feature.id} className="ai-feature-card">
            <img 
              src={feature.icon}
              className="ai-feature-icon"
              alt={feature.title}
            />
            <h3 className="ai-feature-title">{feature.title}</h3>
            <div className="ai-feature-divider"></div>
            <div className="ai-feature-list">
              {feature.features.map((item, index) => (
                <div key={index} className="ai-feature-item">
                  <img 
                    src="https://img.alicdn.com/imgextra/i3/6000000005836/O1CN01wEq5BQ1sywX4GU5hZ_!!6000000005836-2-gg_dtc.png" 
                    className="ai-check-icon"
                    alt="Check"
                  />
                  <span className="ai-feature-text">{item}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AIFeatures;