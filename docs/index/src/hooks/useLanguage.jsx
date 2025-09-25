import React, { createContext, useContext, useState } from 'react'

const LanguageContext = createContext();

export const useLanguage = () => {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
};

export const LanguageProvider = ({ children }) => {
  const [isEnglish, setIsEnglish] = useState(true);

  const toggleLanguage = () => {
    setIsEnglish(!isEnglish);
  };

  const t = (key) => {
    // 定义翻译映射表，使用对象形式组织英文和中文内容
    const translations = {
      // Header
      github: {
        en: 'GitHub',
        zh: 'GitHub'
      },
      language: {
        en: '中文',
        zh: 'EN'
      },
      
      // Hero Section
      heroSubtitle: {
        en: 'Production-Ready Large Language Model Inference Engine',
        zh: '生产级大语言模型推理引擎'
      },
      frameworkText: {
        en: ' LLM Serving Framework',
        zh: ' 大语言模型服务框架'
      },
      viewGithub: {
        en: 'View on GitHub',
        zh: '查看 GitHub'
      },
      
      // Features Section
      whyChoose: {
        en: 'Why Choose RTP-LLM?',
        zh: '为什么选择 RTP-LLM？'
      },
      
      // Feature Cards
      productionProven: {
        en: 'Production Proven',
        zh: '生产验证'
      },
      productionDesc: {
        en: "Deployed across Alibaba's ecosystem serving millions of users daily. Powers Taobao Wenwen, Aidge AI platform, and OpenSearch LLM services.",
        zh: '已在阿里巴巴生态系统中部署，每日服务数百万用户。为淘宝问问、Aidge AI平台和OpenSearch LLM服务提供支持。'
      },
      
      highPerformance: {
        en: 'High Performance',
        zh: '高性能'
      },
      performanceDesc: {
        en: 'Advanced CUDA kernels, quantization (INT8/INT4), KV cache optimization, and speculative decoding for maximum throughput.',
        zh: '先进的CUDA内核、量化技术(INT8/INT4)、KV缓存优化和推测解码，实现最大吞吐量。'
      },
      
      developerFriendly: {
        en: 'Developer Friendly',
        zh: '开发者友好'
      },
      developerDesc: {
        en: 'Seamless HuggingFace integration, multiple weight formats, multi-LoRA serving, and intuitive APIs.',
        zh: '无缝HuggingFace集成、多种权重格式、多LoRA服务和直观的API。'
      },
      
      extensiveModel: {
        en: 'Extensive Model Support',
        zh: '广泛模型支持'
      },
      extensiveDesc: {
        en: 'Support for 20+ model families including Llama, Qwen, ChatGLM, multimodal models, and embedding models.',
        zh: '支持20+模型家族，包括Llama、Qwen、ChatGLM、多模态模型和嵌入模型。'
      },
      
      // AIStats Section
      trustedBy: {
        en: 'Trusted by Industry Leaders',
        zh: '深受行业领导者信赖'
      },
      supportedModels: {
        en: 'Supported Model Families',
        zh: '支持的模型家族'
      },
      dailyUsers: {
        en: 'Daily Active Users',
        zh: '日活跃用户'
      },
      battleTested: {
        en: 'Battle Tested',
        zh: '实战验证'
      },
      communityDriven: {
        en: 'Community Driven',
        zh: '社区驱动'
      },
      
      // AIFeatures Section
      advancedFeatures: {
        en: 'Advanced Features',
        zh: '高级特性'
      },
      performanceFeature: {
        en: 'Performance',
        zh: '性能'
      },
      performanceFeatures: {
        en: [
          'RadixAttention for prefix caching',
          'Zero-overhead CPU scheduler',
          'Prefill-decode disaggregation',
          'Speculative decoding',
          'Advanced quantization (FP8, INT4, AWQ, GPTQ)'
        ],
        zh: [
          'RadixAttention 前缀缓存',
          '零开销 CPU 调度器',
          '预填充-解码分离',
          '推测解码',
          '高级量化技术 (FP8, INT4, AWQ, GPTQ)'
        ]
      },
      scalability: {
        en: 'Scalability',
        zh: '可扩展性'
      },
      scalabilityFeatures: {
        en: [
          'Tensor parallelism',
          'Pipeline parallelism',
          'Expert parallelism for MoE',
          'Multi-LoRA batching',
          'Distributed inference'
        ],
        zh: [
          '张量并行',
          '流水线并行',
          'MoE 专家并行',
          '多 LoRA 批处理',
          '分布式推理'
        ]
      },
      flexibility: {
        en: 'Flexibility',
        zh: '灵活性'
      },
      flexibilityFeatures: {
        en: [
          'Multiple weight formats',
          'Multimodal inputs (text + images)',
          'Structured outputs',
          'P-tuning and LoRA support',
          'OpenAI-compatible APIs'
        ],
        zh: [
          '多种权重格式',
          '多模态输入（文本+图像）',
          '结构化输出',
          'P-tuning 和 LoRA 支持',
          'OpenAI 兼容 API'
        ]
      },
      
      // BrandShowcase Section
      alsoSupport: {
        en: 'We also support the following models',
        zh: '我们还支持以下模型'
      },
      trustedByAlibaba: {
        en: "Trusted by Alibaba Group's mission-critical applications including Taobao, Tmall, and more.",
        zh: '深受包括淘宝、天猫等阿里巴巴集团核心应用信赖。'
      },
      
      // Model Cards - DeepSeek
      deepseekR1: {
        name: {
          en: 'DeepSeek-R1',
          zh: 'DeepSeek-R1'
        },
        description: {
          en: 'A series of advanced reasoning-optimized models (including 671B MoE) trained through reinforcement learning; excels in complex reasoning, math, and code tasks. RTP-LLM provides specific optimizations for Deepseek v3/R1 models.',
          zh: '一系列先进的推理优化模型（包括671B MoE），通过强化学习训练；在复杂推理、数学和代码任务上表现优异。RTP-LLM为Deepseek v3/R1模型提供特定优化'
        }
      },
      deepseekV2: {
        name: {
          en: 'DeepSeek-V2',
          zh: 'DeepSeek-V2'
        },
        description: {
          en: 'A series of advanced reasoning-optimized models (including 671B MoE) trained through reinforcement learning; excels in complex reasoning, math, and code tasks.',
          zh: '一系列先进的推理优化模型（包括671B MoE），通过强化学习训练；在复杂推理、数学和代码任务上表现优异。'
        }
      },
      
      // Model Cards - Qwen
      qwen3_30b_a3b: {
        name: {
          en: 'Qwen3-30B-A3B',
          zh: 'Qwen3-30B-A3B'
        },
        description: {
          en: "Alibaba's latest Qwen3Moe series for complex reasoning, language understanding, and generation tasks; supports MoE variants and generation 3 predecessors.",
          zh: '阿里巴巴最新的Qwen3Moe系列，用于复杂推理、语言理解和生成任务；支持MoE变体以及前代3等。'
        }
      },
      qwen3_coder: {
        name: {
          en: 'Qwen3-Coder-480B-A35B-Instruct',
          zh: 'Qwen3-Coder-480B-A35B-Instruct'
        },
        description: {
          en: "Alibaba's latest Qwen3Moe series for complex reasoning, language understanding, and generation tasks; supports MoE variants and generation 3 predecessors.",
          zh: '阿里巴巴最新的Qwen3Moe系列，用于复杂推理、语言理解和生成任务；支持MoE变体以及前代3等。'
        }
      },
      qwen3_32b: {
        name: {
          en: 'Qwen3-32B',
          zh: 'Qwen3-32B'
        },
        description: {
          en: "Alibaba's latest Qwen3 series for complex reasoning, language understanding, and generation tasks; supports dense variants and generation 3 predecessors.",
          zh: '阿里巴巴最新的Qwen3系列，用于复杂推理、语言理解和生成任务；支持密集变体以及前代3等。'
        }
      },
      qwen2_72b: {
        name: {
          en: 'Qwen2-72B',
          zh: 'Qwen2-72B'
        },
        description: {
          en: "Alibaba's latest Qwen2 series for complex reasoning, language understanding, and generation tasks; supports dense variants and generation 2.5, 2, 1.5 predecessors.",
          zh: '阿里巴巴最新的Qwen2系列，用于复杂推理、语言理解和生成任务；支持密集变体以及前代2.5、2、1.5等。'
        }
      },
      qwen_72b: {
        name: {
          en: 'Qwen-72B',
          zh: 'Qwen-72B'
        },
        description: {
          en: "Alibaba's latest Qwen3 series for complex reasoning, language understanding, and generation tasks; supports MoE variants and generation 2.5, 2 predecessors.",
          zh: '阿里巴巴最新的Qwen3系列，用于复杂推理、语言理解和生成任务；支持MoE变体以及前代2.5、2等。'
        }
      },
      
      // Model Cards - Llama
      llama_4_scout: {
        name: {
          en: 'Llama-4-Scout-17B-16E-Instruct',
          zh: 'Llama-4-Scout-17B-16E-Instruct'
        },
        description: {
          en: "Meta's open-source LLM series with parameter scales from 7B to 400B (Llama 2, 3, and new Llama 4), with excellent performance.",
          zh: 'Meta的开源LLM系列，参数规模从7B到400B（Llama 2、3和新Llama 4），性能卓越。'
        }
      },
      
      // Model Cards - Mistral
      mistral_7b: {
        name: {
          en: 'Mistral-7B-Instruct-v0.2',
          zh: 'Mistral-7B-Instruct-v0.2'
        },
        description: {
          en: 'Mistral AI\'s open-source 7B LLM with strong performance; expanded to MoE ("Mixtral") and NeMo Megatron variants to support larger scales.',
          zh: 'Mistral AI开源的7B LLM，性能强劲；扩展为MoE（"Mixtral"）和NeMo Megatron变体以支持更大规模。'
        }
      },
      
      // Model Cards - Gemma
      gemma_3_1b: {
        name: {
          en: 'gemma-3-1b-it',
          zh: 'gemma-3-1b-it'
        },
        description: {
          en: "Google's efficient multilingual model series (1B-27B); Gemma 3 provides a 128K context window, and its larger (4B+) variants support visual inputs.",
          zh: 'Google高效的多语言模型系列（1B-27B）；Gemma 3提供128K上下文窗口，其较大的（4B+）变体支持视觉输入。'
        }
      },
      
      // Footer Section
      documentation: {
        en: 'Documentation',
        zh: '文档'
      },
      issues: {
        en: 'Issues',
        zh: '问题'
      },
      releases: {
        en: 'Releases',
        zh: '发布'
      },
      copyright: {
        en: '© 2023-2025 RTP-LLM Team. Open source project by Alibaba Foundation Model Inference Team.',
        zh: '© 2023–2025 RTP-LLM 团队 | 阿里巴巴基础模型推理团队开源项目',
      },
      fastTag: {
        en: 'Fast',
        zh: '快速'
      },
      flexibleTag: {
        en: 'Flexible',
        zh: '灵活'
      },
      scalableTag: {
        en: 'Scalable',
        zh: '可扩展'
      }
    };
    
    // 获取对应语言的翻译内容
    // 支持嵌套 key，如 "deepseekR1.description"
    const keys = key.split('.');
    let translation = translations;
    
    for (const k of keys) {
      if (translation && typeof translation === 'object') {
        translation = translation[k];
      } else {
        return key; // 如果路径不存在，返回原始 key
      }
    }
    
    if (!translation) return key;
    
    // 如果是数组，直接返回对应语言的数组
    if (Array.isArray(translation)) {
      return translation;
    }
    
    // 如果是对象，返回对应语言的值
    if (typeof translation === 'object' && translation.en && translation.zh) {
      return isEnglish ? translation.en : translation.zh;
    }
    
    return isEnglish ? translation.en : translation.zh;
  };

  return (
    <LanguageContext.Provider value={{ isEnglish, toggleLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
};