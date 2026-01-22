// CraftyUG - Gallery/Product Rendering Logic

/**
 * Get products by category
 * @param {string} category - 'laptops', 'desktops', or 'accessories'
 * @returns {Array} - Filtered products
 */
function getProductsByCategory(category) {
  return SAMPLE_PRODUCTS.filter(p => p.category === category);
}

/**
 * Get featured products (first 6)
 * @returns {Array} - Featured products
 */
function getFeaturedProducts() {
  return SAMPLE_PRODUCTS.slice(0, 6);
}

/**
 * Generate WhatsApp link for a product
 * @param {Object} product - Product object
 * @returns {string} - WhatsApp URL
 */
function generateWhatsAppLink(product) {
  const message = encodeURIComponent(
    WHATSAPP_CONFIG.messageTemplate(product.name, product.price)
  );
  return `https://wa.me/${WHATSAPP_CONFIG.phone}?text=${message}`;
}

/**
 * Create a product card HTML element
 * @param {Object} product - Product data
 * @param {number} index - Index for animation delay
 * @returns {string} - HTML string
 */
function createProductCard(product, index = 0) {
  const whatsappLink = generateWhatsAppLink(product);
  const delay = index * 100;
  
  return `
    <article class="product-card" style="animation-delay: ${delay}ms">
      <div class="product-card-image">
        <img 
          src="${product.imageUrl1}" 
          alt="${product.name}"
          loading="lazy"
        />
        <div class="product-card-image-overlay"></div>
        <span class="product-card-badge">${product.category}</span>
      </div>
      <div class="product-card-content">
        <h3>${product.name}</h3>
        <p class="product-card-description">${product.description}</p>
        <div class="product-card-footer">
          <span class="product-card-price">${product.price}</span>
          <a href="${whatsappLink}" target="_blank" rel="noopener noreferrer" class="btn-whatsapp">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path>
            </svg>
            Order
          </a>
        </div>
      </div>
    </article>
  `;
}

/**
 * Render products to a container
 * @param {string} containerId - ID of the container element
 * @param {Array} products - Array of products to render
 */
function renderProducts(containerId, products) {
  const container = document.getElementById(containerId);
  if (!container) return;
  
  if (products.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect>
            <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path>
          </svg>
        </div>
        <h3>No Products Found</h3>
        <p>Check back soon for new arrivals!</p>
      </div>
    `;
    return;
  }
  
  container.innerHTML = products.map((product, index) => createProductCard(product, index)).join('');
}

/**
 * Render featured products on homepage
 */
function renderFeaturedProducts() {
  const products = getFeaturedProducts();
  renderProducts('featured-products', products);
}

/**
 * Render products by category
 * @param {string} category - Category to filter by
 */
function renderCategoryProducts(category) {
  const products = getProductsByCategory(category);
  renderProducts('product-grid', products);
}

/**
 * Fetch products from Cloudinary (for future implementation)
 * This function can be expanded to fetch real data from Cloudinary
 * @param {string} folder - Cloudinary folder path
 * @returns {Promise<Array>} - Array of products
 */
async function fetchCloudinaryProducts(folder) {
  // TODO: Implement Cloudinary Client-side List API
  // This requires setting up the Cloudinary API properly
  // For now, we return sample data
  
  // Example implementation:
  // const url = `https://res.cloudinary.com/${CLOUDINARY_CONFIG.cloudName}/image/list/${folder}.json`;
  // const response = await fetch(url);
  // const data = await response.json();
  // return data.resources.map(parseCloudinaryResource);
  
  console.log(`Would fetch from Cloudinary folder: ${folder}`);
  return SAMPLE_PRODUCTS.filter(p => p.category === folder);
}

/**
 * Parse Cloudinary resource to product format
 * @param {Object} resource - Cloudinary resource object
 * @returns {Object} - Product object
 */
function parseCloudinaryResource(resource) {
  return {
    id: resource.public_id,
    name: resource.context?.custom?.caption || resource.public_id.split('/').pop(),
    price: resource.context?.custom?.alt || 'Price on request',
    description: resource.context?.custom?.description || 'Contact us for details.',
    imageUrl: `https://res.cloudinary.com/${CLOUDINARY_CONFIG.cloudName}/image/upload/f_auto,q_auto/${resource.public_id}`,
    category: resource.folder || 'uncategorized',
  };
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function() {
  // Check which page we're on and render appropriate products
  const featuredContainer = document.getElementById('featured-products');
  const productGrid = document.getElementById('product-grid');
  
  if (featuredContainer) {
    renderFeaturedProducts();
  }
  
  if (productGrid) {
    const category = productGrid.dataset.category;
    if (category) {
      renderCategoryProducts(category);
    }
  }
});
