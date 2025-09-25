import React from 'react'
import './index.css'

const FeatureCard = ({ title, description, type, imageUrl }) => {
  const renderProductionCard = () => (
    <div className="production-card">
      <div className="icon-grid">
        <div className="icon-row">
          <div className="gray-icon"></div>
          <div className="icon-with-overlay">
            <div className="gray-icon-border"></div>
            <img 
              src="https://img.alicdn.com/imgextra/i1/6000000002938/O1CN01pur4IK1XZejXJhCqH_!!6000000002938-2-gg_dtc.png" 
              className="overlay-icon-1"
              alt="Icon"
            />
            <img 
              src="https://img.alicdn.com/imgextra/i3/6000000002974/O1CN011HyboC1Xq8zZHcj23_!!6000000002974-2-gg_dtc.png" 
              className="overlay-icon-2"
              alt="Icon"
            />
            <div className="small-dot"></div>
            <div className="mini-dot"></div>
          </div>
          <div className="right-icons-column">
            <div className="right-icons-row">
              <div className="gray-icon"></div>
              <div className="white-icon-container">
                <div className="white-icon-gradient"></div>
                <div className="white-icon-content">
                  <img 
                    src="https://img.alicdn.com/imgextra/i4/6000000002425/O1CN01ZT1rrC1TmhYHUi2mW_!!6000000002425-2-gg_dtc.png" 
                    className="white-icon-image"
                    alt="Icon"
                  />
                </div>
              </div>
            </div>
            <div className="right-icons-row-bottom">
              <img 
                src="https://img.alicdn.com/imgextra/i4/6000000004886/O1CN01fN9HiM1lxqGlJQpPu_!!6000000004886-2-gg_dtc.png" 
                className="bottom-left-icon"
                alt="Icon"
              />
              <div className="bottom-right-icon-container">
                <img 
                  src="https://img.alicdn.com/imgextra/i2/6000000003030/O1CN01fr4fNy1YFnAgHeCEO_!!6000000003030-2-gg_dtc.png" 
                  className="bottom-right-icon"
                  alt="Icon"
                />
              </div>
            </div>
          </div>
        </div>
        <div className="icon-row-bottom">
          <img 
            src="https://img.alicdn.com/imgextra/i3/6000000004167/O1CN01kYXZsG1geXUxFUEx7_!!6000000004167-2-gg_dtc.png" 
            className="bottom-icon"
            alt="Icon"
          />
          <div className="gradient-icon"></div>
        </div>
        <img 
          src="https://img.alicdn.com/imgextra/i4/6000000001535/O1CN01jXCWSd1ND53Lsz451_!!6000000001535-2-gg_dtc.png" 
          className="center-icon"
          alt="Icon"
        />
        <div className="bottom-bar">
          <div className="bottom-bar-icon">
            <img 
              src="https://img.alicdn.com/imgextra/i2/6000000004249/O1CN01A5L9Yo1hG5yYrbIcF_!!6000000004249-2-gg_dtc.png" 
              className="bottom-bar-image"
              alt="Icon"
            />
          </div>
          <div className="bottom-bar-gray"></div>
          <div className="bottom-bar-gray"></div>
          <div className="bottom-bar-gray-border"></div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="feature-card">
      {type === 'production' ? renderProductionCard() : (
        <img 
          src={imageUrl}
          className="feature-image"
          alt={title}
        />
      )}
      <span className="feature-title">{title}</span>
      <span className="feature-description">
        {description}
      </span>
    </div>
  );
};

export default FeatureCard;