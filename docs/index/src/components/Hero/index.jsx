import React from 'react'
import { useLanguage } from '../../hooks/useLanguage.jsx'
import BackgroundAnimation from '../BackgroundAnimation/index.jsx'
import './index.css'

const Hero = () => {
  const { t, isEnglish } = useLanguage();

  const handleViewGithubClick = () => {
    window.open('https://github.com/alibaba/rtp-llm', '_blank');
  };

  const handleDocsClick = () => {
    // 根据当前语言跳转到不同的页面
    const url = isEnglish 
      ? 'https://rtp-llm.ai/build/en/' 
      : 'https://rtp-llm.ai/build/zh_CN/';
    window.open(url, '_blank');
  };

  // 创建Fast标签的SVG图标组件，使用与文字相同的起始颜色
  const FastIcon = () => (
    <svg t="1758721917767" className="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="3409" width="12" height="12">
      <path d="M434.349152 63.403029c17.237432-84.480483 138.070123-84.480483 155.307555 0a475.821389 475.821389 0 0 0 370.946123 370.946123c84.480483 17.237432 84.480483 138.070123 0 155.307555a475.821389 475.821389 0 0 0-370.946123 370.946123c-17.237432 84.480483-138.070123 84.480483-155.307555 0A475.821389 475.821389 0 0 0 63.403029 589.656707c-84.480483-17.237432-84.480483-138.070123 0-155.307555A475.821389 475.821389 0 0 0 434.349152 63.403029z" p-id="3410" fill="#81F4FF"></path>
    </svg>
  );

  // 创建Flexible标签的SVG图标组件，使用与文字相同的起始颜色
  const FlexibleIcon = () => (
    <svg t="1758721917767" className="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="3409" width="12" height="12">
      <path d="M434.349152 63.403029c17.237432-84.480483 138.070123-84.480483 155.307555 0a475.821389 475.821389 0 0 0 370.946123 370.946123c84.480483 17.237432 84.480483 138.070123 0 155.307555a475.821389 475.821389 0 0 0-370.946123 370.946123c-17.237432 84.480483-138.070123 84.480483-155.307555 0A475.821389 475.821389 0 0 0 63.403029 589.656707c-84.480483-17.237432-84.480483-138.070123 0-155.307555A475.821389 475.821389 0 0 0 434.349152 63.403029z" p-id="3410" fill="#CD9CFF"></path>
    </svg>
  );

  // 创建Scalable标签的SVG图标组件，使用与文字相同的起始颜色
  const ScalableIcon = () => (
    <svg t="1758721917767" className="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="3409" width="12" height="12">
      <path d="M434.349152 63.403029c17.237432-84.480483 138.070123-84.480483 155.307555 0a475.821389 475.821389 0 0 0 370.946123 370.946123c84.480483 17.237432 84.480483 138.070123 0 155.307555a475.821389 475.821389 0 0 0-370.946123 370.946123c-17.237432 84.480483-138.070123 84.480483-155.307555 0A475.821389 475.821389 0 0 0 63.403029 589.656707c-84.480483-17.237432-84.480483-138.070123 0-155.307555A475.821389 475.821389 0 0 0 434.349152 63.403029z" p-id="3410" fill="#FFA9A9"></path>
    </svg>
  );

  // 创建Framework标签的SVG图标组件，使用与文字相同的起始颜色
  const FrameworkIcon = () => (
    <svg t="1758721929054" className="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="3554" width="12" height="12">
      <path d="M515.308629 465.678275a66.044579 66.044579 0 0 1-28.158541-7.508944L131.072534 286.146243a61.607475 61.607475 0 0 1 0-111.268903L469.487003 10.533854a104.954563 104.954563 0 0 1 91.301936 0l338.499798 164.343486a61.607475 61.607475 0 0 1 0 111.268903L543.296512 457.998673a63.996685 63.996685 0 0 1-27.987883 7.679602z m63.228724 550.542146c8.788878 5.119735 18.687032 7.764931 28.670515 7.764931a53.586557 53.586557 0 0 0 26.281305-6.655655l322.884606-154.957306a63.143396 63.143396 0 0 0 31.486369-56.317082V411.494416a63.143396 63.143396 0 0 0-30.206435-53.927873 54.439846 54.439846 0 0 0-54.269188-1.365263L580.158603 512.011875a63.740698 63.740698 0 0 0-31.571698 53.927873v396.438129a64.252671 64.252671 0 0 0 29.950448 53.927873z m-155.213292 7.679602a54.269189 54.269189 0 0 1-26.195976-6.740984l-321.690002-154.871977A62.972738 62.972738 0 0 1 42.671781 806.055309V411.323758a63.57004 63.57004 0 0 1 30.206435-53.927873 54.95182 54.95182 0 0 1 55.378464 0L449.349379 512.011875a64.338 64.338 0 0 1 32.510316 53.927873v395.328853a64.252671 64.252671 0 0 1-29.865119 53.927873 58.87695 58.87695 0 0 1-28.670515 8.788878z" p-id="3555" fill="#DBDBDB"></path>
    </svg>
  );

  return (
    <div className="hero-section">
      <BackgroundAnimation />
      <div className="hero-content">
        <span className="hero-logo">RTP-LLM</span>
        <span className="hero-subtitle">{t('heroSubtitle')}</span>
      </div>
      <div className="tag-section">
        <div className="fast-tag">
          <FastIcon />
          <span className="fast-tag-text">{t('fastTag')}</span>
        </div>
        <div className="flexible-tag">
          <FlexibleIcon />
          <span className="flexible-tag-text">{t('flexibleTag')}</span>
        </div>
        <div className="scalable-tag">
          <ScalableIcon />
          <span className="scalable-tag-text">{t('scalableTag')}</span>
        </div>
        <div className="framework-tag">
          <FrameworkIcon />
          <span className="framework-text">{t('frameworkText')}</span>
        </div>
      </div>
      <div className="action-section">
        <div className="view-github-button" onClick={handleViewGithubClick}>
          <span className="view-github-text">{t('viewGithub')}</span>
        </div>
        <div className="docs-button" onClick={handleDocsClick}>
        </div>
      </div>
      <img 
        src="https://img.alicdn.com/imgextra/i2/O1CN01ElGEgJ1JXSD0d3MtI_!!6000000001038-2-tps-5760-1920.png" 
        className="hero-bottom-image"
        alt="Hero Bottom"
      />
    </div>
  );
};

export default Hero;