import React, { useEffect, useRef } from "react";
import { useLanguage } from '../../hooks/useLanguage.jsx';
import styles from "./index.module.css";

const BrandShowcase: React.FC = () => {
  const { t } = useLanguage();
  const scrollRef = useRef(null);

  const modelsData = [
    {
      id: "deepseek-r1",
      logo: "https://img.alicdn.com/imgextra/i3/6000000000047/O1CN014VGRsY1CDZhv66Xn9_!!6000000000047-2-gg_dtc.png",
      nameKey: "deepseekR1",
    },
    {
      id: "deepseek-v2",
      logo: "https://img.alicdn.com/imgextra/i3/6000000000047/O1CN014VGRsY1CDZhv66Xn9_!!6000000000047-2-gg_dtc.png",
      nameKey: "deepseekV2",
    },
    {
      id: "qwen3-30b-a3b",
      logo: "https://img.alicdn.com/imgextra/i1/6000000006774/O1CN01FLglvR1zuY2eKsvVJ_!!6000000006774-2-gg_dtc.png",
      nameKey: "qwen3_30b_a3b",
    },
    {
      id: "qwen3-coder",
      logo: "https://img.alicdn.com/imgextra/i1/6000000006774/O1CN01FLglvR1zuY2eKsvVJ_!!6000000006774-2-gg_dtc.png",
      nameKey: "qwen3_coder",
    },
    {
      id: "qwen3-32b",
      logo: "https://img.alicdn.com/imgextra/i1/6000000006774/O1CN01FLglvR1zuY2eKsvVJ_!!6000000006774-2-gg_dtc.png",
      nameKey: "qwen3_32b",
    },
    {
      id: "qwen2-72b",
      logo: "https://img.alicdn.com/imgextra/i1/6000000006774/O1CN01FLglvR1zuY2eKsvVJ_!!6000000006774-2-gg_dtc.png",
      nameKey: "qwen2_72b",
    },
    {
      id: "qwen-72b",
      logo: "https://img.alicdn.com/imgextra/i1/6000000006774/O1CN01FLglvR1zuY2eKsvVJ_!!6000000006774-2-gg_dtc.png",
      nameKey: "qwen_72b",
    },
    {
      id: "llama-4-scout",
      logo: "https://img.alicdn.com/imgextra/i1/6000000007074/O1CN01ohXbQo227wrSuN0yP_!!6000000007074-2-gg_dtc.png",
      nameKey: "llama_4_scout",
    },
    {
      id: "mistral-7b",
      logo: "https://img.alicdn.com/imgextra/i1/6000000007074/O1CN01ohXbQo227wrSuN0yP_!!6000000007074-2-gg_dtc.png",
      nameKey: "mistral_7b",
    },
    {
      id: "gemma-3-1b",
      logo: "https://img.alicdn.com/imgextra/i3/6000000003764/O1CN01Zam8gP1dfxsqXfTkB_!!6000000003764-2-gg_dtc.png",
      nameKey: "gemma_3_1b",
    },
  ];

  // 将模型数据分为两行
  const topRowModels = modelsData.filter((_, index) => index % 2 === 0);
  const bottomRowModels = modelsData.filter((_, index) => index % 2 === 1);

  // 复制数据以实现无缝循环
  const duplicatedTopRow = [...topRowModels, ...topRowModels, ...topRowModels];
  const duplicatedBottomRow = [
    ...bottomRowModels,
    ...bottomRowModels,
    ...bottomRowModels,
  ];

  useEffect(() => {
    const topScrollContainer = document.querySelector(
      ".ai-models-scroll-container.top-row"
    );
    const bottomScrollContainer = document.querySelector(
      ".ai-models-scroll-container.bottom-row"
    );

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
    <div className={styles.mainSection}>
      {/* 背景图片 */}
      <img
        src="https://img.alicdn.com/imgextra/i4/O1CN01YlHfvr1kOkItZMBel_!!6000000004674-2-tps-5760-4800.png"
        alt="背景"
        className={styles.backgroundImage}
      />

      {/* 内容覆盖层 */}
      <div className={styles.contentOverlay}>
        {/* AI Models Section */}
        <div className={styles.aiModelsSection}>
          <div className={styles.aiModelsHeader}>
            <img
              src="https://img.alicdn.com/imgextra/i4/6000000003738/O1CN01O7UP7k1dU3aP2hgQ8_!!6000000003738-2-gg_dtc.png"
              className={styles.aiModelsIcon}
              alt="Models Icon"
            />
            <div className={styles.aiModelsHeaderText}>
              <span className={styles.aiModelsTitle}>
                {t('alsoSupport')}
              </span>
              <div className={styles.aiModelsDivider}></div>
            </div>
          </div>

          <div className={styles.aiModelsRowsContainer}>
            {/* 上行 */}
            <div className="ai-models-scroll-container top-row">
              <div className="ai-models-scroll-content">
                {duplicatedTopRow.map((model, index) => (
                  <div
                    key={`top-${model.id}-${index}`}
                    className={styles.aiModelCard}
                  >
                    <div className={styles.aiModelHeader}>
                      <img
                        src={model.logo}
                        className={styles.aiModelLogo}
                        alt={`${model.id} Logo`}
                      />
                      <div className={styles.aiModelName}>{t(model.nameKey + '.name')}</div>
                    </div>
                    <div className={styles.aiModelDescription}>
                      {t(model.nameKey + '.description')}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 下行 - 错位排列 */}
            <div className="ai-models-scroll-container bottom-row">
              <div className="ai-models-scroll-content offset">
                {duplicatedBottomRow.map((model, index) => (
                  <div
                    key={`bottom-${model.id}-${index}`}
                    className={styles.aiModelCard}
                  >
                    <div className={styles.aiModelHeader}>
                      <img
                        src={model.logo}
                        className={styles.aiModelLogo}
                        alt={`${model.id} Logo`}
                      />
                      <div className={styles.aiModelName}>{t(model.nameKey + '.name')}</div>
                    </div>
                    <div className={styles.aiModelDescription}>
                      {t(model.nameKey + '.description')}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* 第一条分割线 */}
        <div className={styles.dividerLine} />

        {/* 第一行品牌Logo */}
        <div className={styles.brandRow1}>
          <img
            src="https://img.alicdn.com/imgextra/i3/6000000006617/O1CN01zvfqYM1ykdrMK2i4A_!!6000000006617-2-gg_dtc.png"
            alt="品牌1"
            className={styles.brandLogo}
          />
          <img
            src="https://img.alicdn.com/imgextra/i2/6000000006561/O1CN01tnYGOo1yKzgE1vyiM_!!6000000006561-2-gg_dtc.png"
            alt="品牌2"
            className={styles.brandLogo}
          />
          <img
            src="https://img.alicdn.com/imgextra/i4/6000000000618/O1CN01jX9mKs1GR5rRNWld4_!!6000000000618-2-gg_dtc.png"
            alt="品牌3"
            className={styles.brandLogo}
          />
          <img
            src="https://img.alicdn.com/imgextra/i2/6000000003461/O1CN01Jqv24N1bRBsOOaKFB_!!6000000003461-2-gg_dtc.png"
            alt="品牌4"
            className={styles.brandLogo}
          />
          <img
            src="https://img.alicdn.com/imgextra/i3/6000000006688/O1CN01P2bye11zH9yqeRRXH_!!6000000006688-2-gg_dtc.png"
            alt="品牌5"
            className={styles.brandLogo}
          />
          <img
            src="https://img.alicdn.com/imgextra/i2/6000000002048/O1CN01NpY2ck1R02Ee13Qve_!!6000000002048-2-gg_dtc.png"
            alt="品牌6"
            className={styles.brandLogo}
          />
          <img
            src="https://img.alicdn.com/imgextra/i2/6000000003110/O1CN01OhDkMt1YqQrCnqlzK_!!6000000003110-2-gg_dtc.png"
            alt="品牌7"
            className={styles.brandLogo}
          />
        </div>
        <div className={styles.dividerLine} />

        {/* 中央展示区域 */}
        <div className={styles.centerSection}>
          <div className={styles.centerText}>
            {t('trustedByAlibaba')}
          </div>
        </div>

        <div className={styles.dividerLine} />
        {/* 第二行品牌Logo */}
        <div className={styles.brandRow2}>
          <img
            src="https://img.alicdn.com/imgextra/i3/6000000004558/O1CN01fjN1gQ1jXcMOSmEgW_!!6000000004558-2-gg_dtc.png"
            alt="品牌8"
            className={styles.brandLogo}
          />
          <img
            src="https://img.alicdn.com/imgextra/i1/6000000000745/O1CN01DWlAeH1HNGA3LNBZe_!!6000000000745-2-gg_dtc.png"
            alt="品牌9"
            className={styles.brandLogo}
          />
          <img
            src="https://img.alicdn.com/imgextra/i1/6000000007077/O1CN011MiJFZ229K39fjl9z_!!6000000007077-2-gg_dtc.png"
            alt="品牌10"
            className={styles.brandLogo}
          />
          <img
            src="https://img.alicdn.com/imgextra/i3/6000000001014/O1CN01Jwp0ay1JMShfY5CF9_!!6000000001014-2-gg_dtc.png"
            alt="品牌11"
            className={styles.brandLogo}
          />
          <img
            src="https://img.alicdn.com/imgextra/i2/6000000001754/O1CN01ZzsVPC1OpNn9Nksd1_!!6000000001754-2-gg_dtc.png"
            alt="品牌12"
            className={styles.brandLogo}
          />
          <img
            src="https://img.alicdn.com/imgextra/i1/6000000002039/O1CN01SJWPAO1QvufeJlBVo_!!6000000002039-2-gg_dtc.png"
            alt="品牌13"
            className={styles.brandLogo}
          />
        </div>

        {/* 底部分割线 */}
        <div className={styles.dividerLine} />
      </div>
    </div>
  );
};

export default BrandShowcase;