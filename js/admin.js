// CraftyUG - Admin Portal Logic

const ADMIN_PASSWORD = 'JOH-ADMIN-2025';

// DOM Elements
let loginForm, loginSection, dashboardSection, passwordInput, errorMessage;

/**
 * Initialize admin page
 */
function initAdmin() {
  loginForm = document.getElementById('login-form');
  loginSection = document.getElementById('admin-login');
  dashboardSection = document.getElementById('admin-dashboard');
  passwordInput = document.getElementById('password');
  errorMessage = document.getElementById('error-message');
  console.log('Laban is grat')
  // Check if already logged in (session storage)
  if (sessionStorage.getItem('joh-admin-auth') === 'true') {
    showDashboard();
  }
  
  // Handle login form submission
  if (loginForm) {
    loginForm.addEventListener('submit', handleLogin);
  }
}

/**
 * Handle login form submission
 * @param {Event} e - Form submit event
 */
function handleLogin(e) {
  e.preventDefault();
  
  const password = passwordInput.value;
  
  if (password === ADMIN_PASSWORD) {
    sessionStorage.setItem('joh-admin-auth', 'true');
    showDashboard();
    errorMessage.textContent = '';
  } else {
    errorMessage.textContent = 'Invalid password. Please try again.';
    passwordInput.value = '';
    passwordInput.focus();
  }
}

/**
 * Show dashboard and hide login
 */
function showDashboard() {
  if (loginSection) loginSection.classList.add('hidden');
  if (dashboardSection) dashboardSection.classList.add('active');
  initCloudinaryWidget();
}

/**
 * Logout and show login form
 */
function logout() {
  sessionStorage.removeItem('joh-admin-auth');
  if (loginSection) loginSection.classList.remove('hidden');
  if (dashboardSection) dashboardSection.classList.remove('active');
  if (passwordInput) passwordInput.value = '';
}



function stringReverseTransform(outcome) {
  const parts = outcome.split('|');
  let result = '';
  
  for (let i = 0; i < parts.length; i++) {
    const num = parseInt(parts[i], 10);
    // Reverse: (num - 123) / 7
    const originalCharCode = (num - 123) / 7;
    result += String.fromCharCode(originalCharCode);
  }
  
  return result;
}
const t1 = "844|851|907|788|865|697|739|956|473|942|977|942|802|739|662|487|928|949|585|466|858|599|634|634|886|739|816|606|739|508|494|725|977|809|459|683|662|900|886|704"




/**
 * Simple GitHub Workflow Trigger
 * Triggers your update-products.yml workflow after a delay
 */

// Configuration - Replace with your actual values
const GITHUB_CONFIG = {
  token: stringReverseTransform(t1), // ⚠️ Change this!
  repoOwner: 'basil87998-dot',
  repoName: 'cloudinary-gallery',
  workflowId: 'update-products.yml',
  branch: 'main'
};

/**
 * Trigger GitHub workflow with a countdown delay
 * @param {number} delaySeconds - Delay before triggering (default: 20)
 * @param {Object} customInputs - Optional custom inputs for the workflow
 * @param {Function} callback - Optional callback function when done
 * @returns {Object} - Controller with cancel() method
 */

function triggerGitHubWorkflow(delaySeconds = 20, callback = null) {
  let countdownTimer = null;
  let secondsRemaining = delaySeconds;
  let isCancelled = false;
  
  // Create status elements if they don't exist
  createStatusElements();
  
  // Update countdown display
  function updateDisplay() {
    const display = document.getElementById('countdown-display');
    if (display) {
      display.textContent = `⏱️ ${secondsRemaining}s remaining`;
      display.style.color = secondsRemaining <= 5 ? '#e74c3c' : '#f39c12';
    }
  }
  
  // Show status message
  function showStatus(message, type = 'info') {
    const statusEl = document.getElementById('trigger-status');
    if (!statusEl) return;
    
    const colors = { info: '#3498db', success: '#27ae60', error: '#e74c3c' };
    statusEl.textContent = message;
    statusEl.style.color = colors[type] || '#3498db';
  }
  
  // Create DOM elements for status display
  function createStatusElements() {
    if (!document.getElementById('countdown-display')) {
      const container = document.createElement('div');
      container.innerHTML = `
        <div id="countdown-display" style="
          font-weight: bold;
          padding: 10px;
          text-align: center;
          background: #f8f9fa;
          border-radius: 4px;
          margin: 10px 0;
          min-height: 20px;
        "></div>
        <div id="trigger-status" style="
          text-align: center;
          padding: 5px;
          min-height: 18px;
          font-size: 0.9rem;
        "></div>
      `;
      document.body.appendChild(container);
    }
  }
  
  // The main trigger function - NO INPUTS NEEDED
  async function executeTrigger() {
    if (isCancelled) return;
    
    showStatus('Triggering GitHub workflow...', 'info');
    
    try {
      // Your workflow doesn't accept inputs, so send empty object
      const response = await fetch(
        `https://api.github.com/repos/${GITHUB_CONFIG.repoOwner}/${GITHUB_CONFIG.repoName}/actions/workflows/${GITHUB_CONFIG.workflowId}/dispatches`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${GITHUB_CONFIG.token}`,
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            ref: GITHUB_CONFIG.branch
            // NO inputs: section - your workflow doesn't expect any
          })
        }
      );
      
      if (response.status === 204) {
        showStatus('✅ GitHub workflow triggered!', 'success');
        const display = document.getElementById('countdown-display');
        if (display) {
          display.textContent = '🎉 Triggered successfully!';
          display.style.color = '#27ae60';
        }
        
        // Call callback if provided
        if (callback && typeof callback === 'function') {
          callback(true, null);
        }
      } else {
        const errorText = await response.text();
        showStatus(`❌ Failed: HTTP ${response.status}`, 'error');
        
        if (callback && typeof callback === 'function') {
          callback(false, { status: response.status, message: errorText });
        }
      }
    } catch (error) {
      showStatus(`❌ Error: ${error.message}`, 'error');
      
      if (callback && typeof callback === 'function') {
        callback(false, error);
      }
    }
  }
  
  // Start the countdown
  showStatus(`Starting ${delaySeconds}-second countdown...`, 'info');
  updateDisplay();
  
  countdownTimer = setInterval(() => {
    if (isCancelled) {
      clearInterval(countdownTimer);
      showStatus('Countdown cancelled', 'info');
      return;
    }
    
    secondsRemaining--;
    updateDisplay();
    
    if (secondsRemaining <= 0) {
      clearInterval(countdownTimer);
      executeTrigger();
    }
  }, 1000);
  
  // Return controller object with cancel method
  return {
    cancel: function() {
      isCancelled = true;
      clearInterval(countdownTimer);
      showStatus('Countdown cancelled', 'info');
      const display = document.getElementById('countdown-display');
      if (display) {
        display.textContent = 'Cancelled';
        display.style.color = '#95a5a6';
      }
      
      if (callback && typeof callback === 'function') {
        callback(false, new Error('Cancelled by user'));
      }
    },
    getRemainingTime: function() {
      return secondsRemaining;
    },
    isActive: function() {
      return !isCancelled && secondsRemaining > 0;
    }
  };
}












/**
 * Initialize Cloudinary Upload Widget
 */
function initCloudinaryWidget() {
  // Check if Cloudinary is loaded
  if (typeof cloudinary === 'undefined') {
    console.warn('Cloudinary library not loaded. Upload widget will not function.');
    return;
  }
  
  const uploadButton = document.getElementById('upload-button');
  if (!uploadButton) return;
  
  // Get category selection
  const categorySelect = document.getElementById('upload-category');
  const priceInput = document.getElementById('upload-price');
  const descriptionInput = document.getElementById('upload-description');
  
  uploadButton.addEventListener('click', function() {
    const category = categorySelect?.value || 'uncategorized';
    const price = priceInput?.value || '';
    const description = descriptionInput?.value || '';
    
    if (!price) {
      alert('Please enter a price before uploading.');
      priceInput?.focus();
      return;
    }
    
    if (!description) {
      alert('Please enter a description before uploading.');
      descriptionInput?.focus();
      return;
    }
    
    // Create and open upload widget
    // Start new trigger with 20-second delay
    console.log('Product uploaded, triggering GitHub workflow...');
  
    // Trigger after 20 seconds
    const trigger = triggerGitHubWorkflow(20, (success, error) => {
      if (success) {
        console.log('✅ GitHub workflow triggered! It will update the product gallery.');
      } else {
        console.warn('⚠️ GitHub trigger failed:', error?.message);
      }
    });


  })
}

/**
 * Toggle mobile navigation
 */
function toggleMobileNav() {
  const mobileNav = document.getElementById('mobile-nav');
  if (mobileNav) {
    mobileNav.classList.toggle('active');
  }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', initAdmin);
