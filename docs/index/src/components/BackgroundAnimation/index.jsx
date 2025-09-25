import React, { useEffect, useRef } from 'react';
import { ColorLerp2Effect } from 'background-animations';
import './index.css';

const BackgroundAnimation = () => {
  const containerRef = useRef(null);
  const effectRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // 创建渐变背景动画效果
    const gradientEffect = new ColorLerp2Effect({
      pairs: [
        ['rgb(148, 59, 222)', 'rgb(26, 255, 251)'],    // 紫色到青色
        ['rgb(74, 124, 217)', 'rgb(167, 35, 232)'],    // 蓝色到紫色
        ['rgb(247, 140, 99)', 'rgb(209, 29, 40)']      // 橙色到红色
      ],
      percent: 0.03
    });

    // 将效果添加到容器中
    containerRef.current.appendChild(gradientEffect.element);
    effectRef.current = gradientEffect;

    // 设置尺寸并启动动画
    const updateSize = () => {
      if (effectRef.current && containerRef.current) {
        const { offsetWidth, offsetHeight } = containerRef.current;
        effectRef.current.resize(offsetWidth, offsetHeight);
      }
    };

    updateSize();
    gradientEffect.start();

    // 监听窗口大小变化
    window.addEventListener('resize', updateSize);

    // 清理函数
    return () => {
      window.removeEventListener('resize', updateSize);
      if (effectRef.current) {
        effectRef.current.stop && effectRef.current.stop();
      }
      if (containerRef.current && gradientEffect.element) {
        containerRef.current.removeChild(gradientEffect.element);
      }
    };
  }, []);

  return <div ref={containerRef} className="background-animation" />;
};

export default BackgroundAnimation;