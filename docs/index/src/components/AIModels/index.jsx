import React, { useEffect, useRef } from 'react';
import './index.css';

const AIModels = () => {
  const scrollRef = useRef(null);

  const modelsData = [
    {
      id: 'deepseek-r1',
      logo: 'https://img.alicdn.com/imgextra/i3/6000000000047/O1CN014VGRsY1CDZhv66Xn9_!!6000000000047-2-gg_dtc.png',
      name: 'DeepSeek-R1',
      description: '一系列先进的推理优化模型（包括671B MoE），通过强化学习训练；在复杂推理、数学和代码任务上表现优异。RTP-LLM为Deepseek v3/R1模型提供特定优化'
    },
    {
      id: 'deepseek-v2',
      logo: 'https://img.alicdn.com/imgextra/i3/6000000000047/O1CN014VGRsY1CDZhv66Xn9_!!6000000000047-2-gg_dtc.png',
      name: 'DeepSeek-V2',
      description: '一系列先进的推理优化模型（包括671B MoE），通过强化学习训练；在复杂推理、数学和代码任务上表现优异。'
    },
    {
      id: 'qwen3-30b-a3b',
      logo: 'https://img.alicdn.com/imgextra/i1/6000000006774/O1CN01FLglvR1zuY2eKsvVJ_!!6000000006774-2-gg_dtc.png',
      name: 'Qwen3-30B-A3B',
      description: '阿里巴巴最新的Qwen3Moe系列，用于复杂推理、语言理解和生成任务；支持MoE变体以及前代3等。'
    },
    {
      id: 'qwen3-coder',
      logo: 'https://img.alicdn.com/imgextra/i1/6000000006774/O1CN01FLglvR1zuY2eKsvVJ_!!6000000006774-2-gg_dtc.png',
      name: 'Qwen3-Coder-480B-A35B-Instruct',
      description: '阿里巴巴最新的Qwen3Moe系列，用于复杂推理、语言理解和生成任务；支持MoE变体以及前代3等。'
    },
    {
      id: 'qwen3-32b',
      logo: 'https://img.alicdn.com/imgextra/i1/6000000006774/O1CN01FLglvR1zuY2eKsvVJ_!!6000000006774-2-gg_dtc.png',
      name: 'Qwen3-32B',
      description: '阿里巴巴最新的Qwen3系列，用于复杂推理、语言理解和生成任务；支持密集变体以及前代3等。'
    },
    {
      id: 'qwen2-72b',
      logo: 'https://img.alicdn.com/imgextra/i1/6000000006774/O1CN01FLglvR1zuY2eKsvVJ_!!6000000006774-2-gg_dtc.png',
      name: 'Qwen2-72B',
      description: '阿里巴巴最新的Qwen2系列，用于复杂推理、语言理解和生成任务；支持密集变体以及前代2.5、2、1.5等。'
    },
    {
      id: 'qwen-72b',
      logo: 'https://img.alicdn.com/imgextra/i1/6000000006774/O1CN01FLglvR1zuY2eKsvVJ_!!6000000006774-2-gg_dtc.png',
      name: 'Qwen-72B',
      description: '阿里巴巴最新的Qwen3系列，用于复杂推理、语言理解和生成任务；支持MoE变体以及前代2.5、2等。'
    },
    {
      id: 'llama-4-scout',
      logo: 'https://img.alicdn.com/imgextra/i1/6000000007074/O1CN01ohXbQo227wrSuN0yP_!!6000000007074-2-gg_dtc.png',
      name: 'Llama-4-Scout-17B-16E-Instruct',
      description: 'Meta的开源LLM系列，参数规模从7B到400B（Llama 2、3和新Llama 4），性能卓越。'
    },
    {
      id: 'mistral-7b',
      logo: 'https://img.alicdn.com/imgextra/i1/6000000007074/O1CN01ohXbQo227wrSuN0yP_!!6000000007074-2-gg_dtc.png',
      name: 'Mistral-7B-Instruct-v0.2',
      description: 'Mistral AI开源的7B LLM，性能强劲；扩展为MoE（"Mixtral"）和NeMo Megatron变体以支持更大规模。'
    },
    {
      id: 'gemma-3-1b',
      logo: 'https://img.alicdn.com/imgextra/i3/6000000003764/O1CN01Zam8gP1dfxsqXfTkB_!!6000000003764-2-gg_dtc.png',
      name: 'gemma-3-1b-it',
      description: 'Google高效的多语言模型系列（1B-27B）；Gemma 3提供128K上下文窗口，其较大的（4B+）变体支持视觉输入。'
    }
  ];

  // 将模型数据分为两行
  const topRowModels = modelsData.filter((_, index) => index % 2 === 0);
  const bottomRowModels = modelsData.filter((_, index) => index % 2 === 1);

  // 复制数据以实现无缝循环
  const duplicatedTopRow = [...topRowModels, ...topRowModels, ...topRowModels];
  const duplicatedBottomRow = [...bottomRowModels, ...bottomRowModels, ...bottomRowModels];

  useEffect(() => {
    const topScrollContainer = document.querySelector('.ai-models-scroll-container.top-row');
    const bottomScrollContainer = document.querySelector('.ai-models-scroll-container.bottom-row');
    
    if (!topScrollContainer || !bottomScrollContainer) return;

    let topScrollPosition = 0;
    let bottomScrollPosition = 0;
    const cardWidth = 390; // 卡片宽度 + 间距
    const topRowWidth = cardWidth * topRowModels.length;
    const bottomRowWidth = cardWidth * bottomRowModels.length;

    const scroll = () => {
      // 上行向右滚动
      topScrollPosition += 1;
      if (topScrollPosition >= topRowWidth) {
        topScrollPosition = 0;
      }
      topScrollContainer.scrollLeft = topScrollPosition;

      // 下行向左滚动（反向）
      bottomScrollPosition -= 1;
      if (bottomScrollPosition <= -bottomRowWidth) {
        bottomScrollPosition = 0;
      }
      bottomScrollContainer.scrollLeft = Math.abs(bottomScrollPosition);
    };

    const intervalId = setInterval(scroll, 50);

    return () => clearInterval(intervalId);
  }, [topRowModels.length, bottomRowModels.length]);

  return (
    <div className="ai-models-section">
      <div className="ai-models-header">
        <img 
          src="https://img.alicdn.com/imgextra/i4/6000000003738/O1CN01O7UP7k1dU3aP2hgQ8_!!6000000003738-2-gg_dtc.png" 
          className="ai-models-icon"
          alt="Models Icon"
        />
        <div className="ai-models-header-text">
          <span className="ai-models-title">We also support the following models</span>
          <div className="ai-models-divider"></div>
        </div>
      </div>
      
      <div className="ai-models-rows-container">
        {/* 上行 */}
        <div className="ai-models-scroll-container top-row">
          <div className="ai-models-scroll-content">
            {duplicatedTopRow.map((model, index) => (
              <div key={`top-${model.id}-${index}`} className="ai-model-card">
                <div className="ai-model-header">
                  <img 
                    src={model.logo}
                    className="ai-model-logo"
                    alt={`${model.id} Logo`}
                  />
                  <div className="ai-model-name">
                    {model.name}
                  </div>
                </div>
                <div className="ai-model-description">
                  {model.description}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 下行 - 错位排列 */}
        <div className="ai-models-scroll-container bottom-row">
          <div className="ai-models-scroll-content offset">
            {duplicatedBottomRow.map((model, index) => (
              <div key={`bottom-${model.id}-${index}`} className="ai-model-card">
                <div className="ai-model-header">
                  <img 
                    src={model.logo}
                    className="ai-model-logo"
                    alt={`${model.id} Logo`}
                  />
                  <div className="ai-model-name">
                    {model.name}
                  </div>
                </div>
                <div className="ai-model-description">
                  {model.description}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AIModels;