// CraftyUG - Cloudinary Configuration
// Replace these values with your actual Cloudinary credentials

const CLOUDINARY_CONFIG = {
  cloudName: 'your-cloud-name', // Replace with your Cloudinary cloud name
  apiKey: 'your-api-key',       // Replace with your Cloudinary API key
  uploadPreset: 'joh-tech',     // Create this unsigned preset in Cloudinary
};

// WhatsApp Configuration
const WHATSAPP_CONFIG = {
  phone: '256776576547',
  messageTemplate: (productName, price) => 
    `Hello CraftyUG, I am interested in ${productName} priced at ${price}. Is it available?`
};

// Sample products for demonstration (will be replaced with Cloudinary data)
const SAMPLE_PRODUCTS = [
  {
    id: '1',
    name: 'MacBook Pro 16"',
    price: 'UGX 8,500,000',
    description: 'Apple M3 Pro chip, 18GB RAM, 512GB SSD. Perfect for professionals.',
    imageUrl: 'https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=600&q=80',
    category: 'laptops',
  },
  {
    id: '2',
    name: 'Dell XPS 15',
    price: 'UGX 6,200,000',
    description: 'Intel Core i7, 16GB RAM, 512GB SSD. Stunning InfinityEdge display.',
    imageUrl: 'https://images.unsplash.com/photo-1593642632559-0c6d3fc62b89?w=600&q=80',
    category: 'laptops',
  },
  {
    id: '3',
    name: 'HP Spectre x360',
    price: 'UGX 5,800,000',
    description: 'Convertible laptop with Intel Core i7, 16GB RAM, and OLED display.',
    imageUrl: 'https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=600&q=80',
    category: 'laptops',
  },
  {
    id: '4',
    name: 'Gaming Desktop Pro',
    price: 'UGX 12,000,000',
    description: 'RTX 4080, Intel i9, 32GB RAM, 2TB NVMe. Ultimate gaming experience.',
    imageUrl: 'https://images.unsplash.com/photo-1587202372775-e229f172b9d7?w=600&q=80',
    category: 'desktops',
  },
  {
    id: '5',
    name: 'iMac 24"',
    price: 'UGX 7,500,000',
    description: 'Apple M3 chip, 8GB RAM, 256GB SSD. Beautiful 4.5K Retina display.',
    imageUrl: 'https://images.unsplash.com/photo-1527443224154-c4a3942d3acf?w=600&q=80',
    category: 'desktops',
  },
  {
    id: '6',
    name: 'Workstation Tower',
    price: 'UGX 9,800,000',
    description: 'AMD Threadripper, 64GB RAM, Quadro RTX. For creative professionals.',
    imageUrl: 'https://images.unsplash.com/photo-1593640408182-31c70c8268f5?w=600&q=80',
    category: 'desktops',
  },
  {
    id: '7',
    name: 'Mechanical Keyboard RGB',
    price: 'UGX 450,000',
    description: 'Cherry MX switches, per-key RGB, aluminum frame. Premium typing.',
    imageUrl: 'https://images.unsplash.com/photo-1511467687858-23d96c32e4ae?w=600&q=80',
    category: 'accessories',
  },
  {
    id: '8',
    name: 'Wireless Gaming Mouse',
    price: 'UGX 280,000',
    description: '25K DPI sensor, 70-hour battery, lightweight design.',
    imageUrl: 'https://images.unsplash.com/photo-1527864550417-7fd91fc51a46?w=600&q=80',
    category: 'accessories',
  },
  {
    id: '9',
    name: '27" 4K Monitor',
    price: 'UGX 1,800,000',
    description: 'IPS panel, 144Hz, HDR600, USB-C with 90W charging.',
    imageUrl: 'https://images.unsplash.com/photo-1527443195645-1133f7f28990?w=600&q=80',
    category: 'accessories',
  },
];
