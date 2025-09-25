import React from 'react';
import { useLanguage } from '../../hooks/useLanguage.jsx';
import styles from './index.module.css';

const Footer: React.FC = () => {
  const { t, isEnglish } = useLanguage();

  const handleGithubClick = () => {
    window.open('https://github.com/alibaba/rtp-llm', '_blank');
  };

  const handleDocumentationClick = () => {
    // 根据当前语言跳转到不同的文档页面
    const url = isEnglish 
      ? 'https://rtp-llm.ai/build/en/' 
      : 'https://rtp-llm.ai/build/zh_CN/';
    window.open(url, '_blank');
  };

  const handleIssuesClick = () => {
    window.open('https://github.com/alibaba/rtp-llm/issues', '_blank');
  };

  const handleReleasesClick = () => {
    window.open('https://github.com/alibaba/rtp-llm/releases', '_blank');
  };

  return (
    <div className={styles.footerSection}>
      <div className={styles.footerLinks}>
        <div className={styles.linkItem} onClick={handleGithubClick}>
          <img src="https://img.alicdn.com/imgextra/i1/6000000007725/O1CN01Mau9v526w6hT5D93O_!!6000000007725-2-gg_dtc.png" alt="GitHub图标" className={styles.linkIcon} />
          <span className={styles.linkText}>{t('github')}</span>
        </div>
        <div className={styles.linkItem} onClick={handleDocumentationClick}>
          <img src="https://img.alicdn.com/imgextra/i4/6000000005992/O1CN01rLXyj11u8OJp4wUGS_!!6000000005992-2-gg_dtc.png" alt="Documentation图标" className={styles.linkIcon} />
          <span className={styles.linkText}>{t('documentation')}</span>
        </div>
        <div className={styles.linkItem} onClick={handleIssuesClick}>
          <img src="https://img.alicdn.com/imgextra/i1/6000000000916/O1CN01NW9maG1IdZt8svn6Y_!!6000000000916-2-gg_dtc.png" alt="Issues图标" className={styles.linkIcon} />
          <span className={styles.linkText}>{t('issues')}</span>
        </div>
        <div className={styles.linkItem} onClick={handleReleasesClick}>
          <img src="https://img.alicdn.com/imgextra/i1/6000000004089/O1CN01D4uHHL1g4obavGUNv_!!6000000004089-2-gg_dtc.png" alt="Releases图标" className={styles.linkIcon} />
          <span className={styles.linkText}>{t('releases')}</span>
        </div>
      </div>
      <span className={styles.copyright}>{t('copyright')}</span>
    </div>
  );
};

export default Footer;