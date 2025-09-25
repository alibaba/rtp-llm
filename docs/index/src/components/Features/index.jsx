import React from 'react'
import { useLanguage } from '../../hooks/useLanguage.jsx'
import FeatureCard from '../FeatureCard/index.jsx'
import './index.css'

const Features = () => {
  const { t } = useLanguage();

  const featuresData = [
    {
      id: 'production',
      title: t('productionProven'),
      description: t('productionDesc'),
      type: 'normal',
      imageUrl: 'https://img.alicdn.com/imgextra/i3/O1CN01AYzZ6Z1GUl1sW6KUp_!!6000000000626-2-tps-1200-800.png'
    },
    {
      id: 'performance',
      title: t('highPerformance'),
      description: t('performanceDesc'),
      type: 'normal',
      imageUrl: 'https://img.alicdn.com/imgextra/i4/O1CN01TCvuAd1dcl6zSgERE_!!6000000003757-2-tps-1200-800.png'
    },
    {
      id: 'developer',
      title: t('developerFriendly'),
      description: t('developerDesc'),
      type: 'normal',
      imageUrl: 'https://img.alicdn.com/imgextra/i3/O1CN01tPrJr01RTnymW5Ubx_!!6000000002113-2-tps-1200-800.png'
    },
    {
      id: 'model',
      title: t('extensiveModel'),
      description: t('extensiveDesc'),
      type: 'normal',
      imageUrl: 'https://img.alicdn.com/imgextra/i1/O1CN01jEchQf1yGPifB5MzL_!!6000000006551-2-tps-1200-800.png'
    }
  ];

  return (
    <div className="features-section">
      <div className="features-title-text">{t('whyChoose')}</div>
      <div className="features-grid">
        {featuresData.map((feature) => (
          <FeatureCard
            key={feature.id}
            title={feature.title}
            description={feature.description}
            type={feature.type}
            imageUrl={feature.imageUrl}
          />
        ))}
      </div>
    </div>
  );
};

export default Features;