// Language Switcher JavaScript
(function () {
  "use strict";

  // Language configuration
  const LANGUAGES = {
    en: { name: "English", flag: "🇺🇸", display: "EN" },
    zh: { name: "中文", flag: "🇨🇳", display: "中文" },
  };

  // Get current language from URL
  function getCurrentLanguage() {
    const path = window.location.pathname;
    if (path.includes("/zh_CN/") || path.includes("/zh/")) {
      return "zh";
    }
    return "en"; // Default to English if no language prefix found
  }

  // Switch to target language
  function switchLanguage(targetLang) {
    const currentPath = window.location.pathname;
    let newPath;

    // Check if current path already has a language prefix
    const hasEnPrefix = currentPath.includes("/en/");
    const hasZhPrefix =
      currentPath.includes("/zh_CN/") || currentPath.includes("/zh/");

    if (targetLang === "en") {
      if (hasZhPrefix) {
        // Replace Chinese prefix with English
        newPath = currentPath.replace(/\/zh(_CN)?\//, "/en/");
      } else if (hasEnPrefix) {
        // Already English, no change needed
        newPath = currentPath;
      } else {
        // No language prefix, add English prefix
        if (currentPath === "/" || currentPath === "") {
          newPath = "/en/";
        } else {
          newPath = "/en" + currentPath;
        }
      }
    } else {
      // Target language is Chinese
      if (hasEnPrefix) {
        // Replace English prefix with Chinese
        newPath = currentPath.replace(/\/en\//, "/zh_CN/");
      } else if (hasZhPrefix) {
        // Already Chinese, no change needed
        newPath = currentPath;
      } else {
        // No language prefix, add Chinese prefix
        if (currentPath === "/" || currentPath === "") {
          newPath = "/zh_CN/";
        } else {
          newPath = "/zh_CN" + currentPath;
        }
      }
    }

    // Navigate to new URL
    window.location.href = window.location.origin + newPath;
  }

  // Create language switcher HTML
  function createLanguageSwitcher() {
    const currentLang = getCurrentLanguage();
    const switcher = document.createElement("div");
    switcher.className = "language-switcher";
    switcher.innerHTML = `
      <button id="language-toggle" class="btn btn-sm navbar-btn" title="Switch Language">
        <i class="fas fa-globe"></i>
        <span>${LANGUAGES[currentLang].display}</span>
      </button>
      <div class="language-dropdown" id="language-dropdown">
        <a href="#" class="language-option ${
          currentLang === "en" ? "active" : ""
        }" data-lang="en">
          <span class="flag">🇺🇸</span>
          <span>English</span>
        </a>
        <a href="#" class="language-option ${
          currentLang === "zh" ? "active" : ""
        }" data-lang="zh">
          <span class="flag">🇨🇳</span>
          <span>中文</span>
        </a>
      </div>
    `;
    return switcher;
  }

  // Insert the language switcher into the page
  function insertLanguageSwitcher() {
    const insertionPoint = document.querySelector(".article-header-buttons");
    if (!insertionPoint) return null;

    const switcher = createLanguageSwitcher();
    insertionPoint.appendChild(switcher);
    return switcher;
  }

  // Initialize event listeners
  function initializeEventListeners(switcher) {
    const toggleButton = switcher.querySelector("#language-toggle");
    const dropdown = switcher.querySelector("#language-dropdown");

    // Toggle dropdown
    toggleButton.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropdown.classList.toggle("show");
    });

    // Handle language selection
    switcher.querySelectorAll(".language-option").forEach((option) => {
      option.addEventListener("click", (e) => {
        e.preventDefault();
        switchLanguage(option.getAttribute("data-lang"));
      });
    });

    // Close dropdown when clicking outside or pressing escape
    document.addEventListener("click", (e) => {
      if (!switcher.contains(e.target)) {
        dropdown.classList.remove("show");
      }
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        dropdown.classList.remove("show");
      }
    });
  }

  // Main initialization function
  function initLanguageSwitcher() {
    // Remove existing switchers
    document.querySelectorAll(".language-switcher").forEach((s) => s.remove());

    // Create and insert new switcher
    const switcher = insertLanguageSwitcher();
    if (switcher) {
      initializeEventListeners(switcher);
    }
  }

  // Initialize
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initLanguageSwitcher);
  } else {
    initLanguageSwitcher();
  }
})();
