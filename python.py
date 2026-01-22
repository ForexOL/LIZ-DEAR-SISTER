from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import textwrap

class LogoMaker:
    def __init__(self, image_path=None):
        """
        Initialize LogoMaker with optional base image
        """
        self.image = None
        if image_path:
            self.load_image(image_path)
    
    def load_image(self, image_path):
        """Load image from path and ensure RGBA mode for transparency"""
        try:
            self.image = Image.open(image_path)
            # Convert to RGBA to support transparency
            if self.image.mode != 'RGBA':
                self.image = self.image.convert('RGBA')
            print(f"Loaded image: {image_path}")
            print(f"Size: {self.image.size}, Mode: {self.image.mode}")
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def create_simple_logo(self, text, output_size=(500, 500), 
                          bg_color=None,  # Default to transparent
                          text_color=(0, 0, 0, 255),
                          font_size=40,
                          shadow=True,
                          shadow_color=(0, 0, 0, 100),
                          shadow_offset=(2, 2),
                          output_path="logo.png"):
        """
        Create a simple text-based logo with optional transparent background
        """
        # Create new image with transparent background by default
        if bg_color is None:
            logo = Image.new('RGBA', output_size, (255, 255, 255, 0))
        else:
            # Ensure bg_color has alpha channel
            if len(bg_color) == 3:
                bg_color = bg_color + (255,)
            logo = Image.new('RGBA', output_size, bg_color)
        
        draw = ImageDraw.Draw(logo)
        
        # Load font
        font = self._load_font(font_size)
        
        # Calculate text position
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (output_size[0] - text_width) / 2
        y = (output_size[1] - text_height) / 2
        
        # Add shadow if requested
        if shadow:
            # Ensure shadow_color has alpha channel
            if len(shadow_color) == 3:
                shadow_color = shadow_color + (100,)
            draw.text((x + shadow_offset[0], y + shadow_offset[1]), 
                     text, fill=shadow_color, font=font)
        
        # Draw main text
        draw.text((x, y), text, fill=text_color, font=font)
        
        # Save with transparency
        logo.save(output_path, 'PNG', optimize=True)
        print(f"Simple logo saved to: {output_path}")
        return logo
    
    def image_to_logo(self, output_size=(500, 500),
                     remove_background=True,
                     background_color=None,  # None for transparent
                     border=False,
                     border_color=(255, 255, 255, 255),
                     border_width=10,
                     add_text=None,
                     text_color=(255, 255, 255, 255),
                     text_position="bottom",
                     text_padding=20,
                     font_size=30,
                     output_path="image_logo.png"):
        """
        Convert an image to a logo with maintained transparency
        """
        if self.image is None:
            print("No image loaded!")
            return None
        
        # Resize image while maintaining aspect ratio
        img_resized = self._resize_with_aspect(output_size)
        
        # Remove or change background
        if remove_background:
            img_resized = self._remove_background_advanced(img_resized)
        elif background_color is not None:
            # Replace background with specified color
            img_resized = self._replace_background(img_resized, background_color)
        
        # Create canvas with transparent background
        if background_color is None:
            canvas = Image.new('RGBA', output_size, (255, 255, 255, 0))
        else:
            # Ensure background_color has alpha channel
            if len(background_color) == 3:
                background_color = background_color + (255,)
            canvas = Image.new('RGBA', output_size, background_color)
        
        # Center the resized image on canvas
        paste_x = (output_size[0] - img_resized.width) // 2
        paste_y = (output_size[1] - img_resized.height) // 2
        canvas.paste(img_resized, (paste_x, paste_y), img_resized if img_resized.mode == 'RGBA' else None)
        
        # Add border if requested
        if border:
            canvas = self._add_border(canvas, border_width, border_color)
        
        # Add text if provided
        if add_text:
            canvas = self._add_text_to_image(canvas, add_text, text_color, 
                                           text_position, font_size, text_padding)
        
        # Save with transparency
        canvas.save(output_path, 'PNG', optimize=True)
        print(f"Image logo saved to: {output_path}")
        return canvas
    
    def _resize_with_aspect(self, target_size):
        """Resize image while maintaining aspect ratio"""
        original_width, original_height = self.image.size
        target_width, target_height = target_size
        
        # Calculate aspect ratios
        original_ratio = original_width / original_height
        target_ratio = target_width / target_height
        
        if original_ratio > target_ratio:
            # Image is wider than target
            new_width = target_width
            new_height = int(target_width / original_ratio)
        else:
            # Image is taller than target
            new_height = target_height
            new_width = int(target_height * original_ratio)
        
        return self.image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def _remove_background_advanced(self, image, threshold=240, feather_edges=True):
        """
        Advanced background removal with edge feathering
        """
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Convert to numpy array for processing
        data = np.array(image)
        
        # Create alpha channel based on brightness threshold
        rgb = data[:, :, :3]
        brightness = np.mean(rgb, axis=2)
        
        # Create alpha mask
        alpha = np.where(brightness > threshold, 0, 255)
        
        # Feather edges for smoother transparency
        if feather_edges:
            from scipy import ndimage
            alpha = ndimage.gaussian_filter(alpha.astype(float), sigma=1)
            alpha = np.clip(alpha, 0, 255).astype(np.uint8)
        
        # Apply alpha channel
        data[:, :, 3] = alpha
        
        return Image.fromarray(data, 'RGBA')
    
    def _replace_background(self, image, new_bg_color):
        """Replace background with specified color"""
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Create new image with background color
        bg = Image.new('RGBA', image.size, new_bg_color)
        
        # Extract alpha channel as mask
        alpha = image.getchannel('A')
        
        # Paste foreground onto background using alpha as mask
        bg.paste(image, (0, 0), alpha)
        
        return bg
    
    def _add_border(self, image, border_width, border_color):
        """Add border to image with transparency support"""
        # Ensure border_color has alpha channel
        if len(border_color) == 3:
            border_color = border_color + (255,)
        
        # Create border
        bordered = ImageOps.expand(image, border=border_width, fill=border_color)
        return bordered
    
    def _load_font(self, font_size, bold=False):
        """Load font with fallbacks"""
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else 
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            "./arialbd.ttf" if bold else "./arial.ttf"
        ]
        
        for font_path in font_paths:
            try:
                return ImageFont.truetype(font_path, font_size)
            except:
                continue
        
        print("Using default font")
        return ImageFont.load_default()
    
    def _add_text_to_image(self, image, text, text_color, 
                          position="bottom", font_size=30, padding=20):
        """
        Add text to image with proper transparency handling
        """
        draw = ImageDraw.Draw(image)
        
        # Load font
        font = self._load_font(font_size, bold=True)
        
        # Wrap text
        avg_char_width = font_size * 0.6
        max_chars = int(image.width / avg_char_width)
        wrapped_text = textwrap.fill(text, width=max_chars)
        
        # Calculate text position
        lines = wrapped_text.split('\n')
        line_height = font_size * 1.2
        total_height = len(lines) * line_height
        
        bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
        text_width = bbox[2] - bbox[0]
        
        # Position calculation
        if position == "top":
            y = padding
        elif position == "center":
            y = (image.height - total_height) // 2
        elif position == "bottom":
            y = image.height - total_height - padding
        else:
            y = padding
        
        x = (image.width - text_width) // 2
        
        # Add text background for better readability (optional)
        bg_padding = 5
        draw.rounded_rectangle(
            [x - bg_padding, y - bg_padding, 
             x + text_width + bg_padding, y + total_height + bg_padding],
            radius=5,
            fill=(0, 0, 0, 150)  # Semi-transparent black
        )
        
        # Draw text
        draw.multiline_text((x, y), wrapped_text, fill=text_color, 
                           font=font, align='center')
        
        return image
    
    def create_icon_logo(self, output_size=(256, 256),
                        shape="circle",
                        padding=20,
                        shadow=True,
                        shadow_blur=5,
                        shadow_offset=(0, 2),
                        output_path="icon_logo.png"):
        """
        Create an icon-style logo with transparent background
        """
        if self.image is None:
            print("No image loaded!")
            return None
        
        # Create canvas with transparent background
        canvas = Image.new('RGBA', output_size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(canvas)
        
        # Create shape mask
        mask = Image.new('L', output_size, 0)
        mask_draw = ImageDraw.Draw(mask)
        
        shape_area = [
            (padding, padding), 
            (output_size[0] - padding, output_size[1] - padding)
        ]
        
        if shape == "circle":
            mask_draw.ellipse(shape_area, fill=255)
        elif shape == "square":
            mask_draw.rectangle(shape_area, fill=255)
        elif shape == "rounded":
            radius = output_size[0] // 4
            mask_draw.rounded_rectangle(shape_area, radius=radius, fill=255)
        
        # Resize and crop image
        img_resized = self._resize_with_aspect(output_size)
        
        # Convert to RGBA if needed
        if img_resized.mode != 'RGBA':
            img_resized = img_resized.convert('RGBA')
        
        # Apply mask
        img_masked = Image.new('RGBA', output_size, (255, 255, 255, 0))
        img_masked.paste(img_resized, (0, 0), mask)
        
        # Add shadow if requested
        if shadow:
            shadow_layer = Image.new('RGBA', output_size, (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow_layer)
            
            shadow_area = [
                (padding + shadow_offset[0], padding + shadow_offset[1]),
                (output_size[0] - padding + shadow_offset[0], 
                 output_size[1] - padding + shadow_offset[1])
            ]
            
            if shape == "circle":
                shadow_draw.ellipse(shadow_area, fill=(0, 0, 0, 100))
            elif shape == "square":
                shadow_draw.rectangle(shadow_area, fill=(0, 0, 0, 100))
            elif shape == "rounded":
                radius = output_size[0] // 4
                shadow_draw.rounded_rectangle(shadow_area, radius=radius, 
                                            fill=(0, 0, 0, 100))
            
            # Apply blur to shadow
            shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(shadow_blur))
            canvas.alpha_composite(shadow_layer)
        
        # Paste masked image
        canvas.alpha_composite(img_masked)
        
        # Save with transparency
        canvas.save(output_path, 'PNG', optimize=True)
        print(f"Icon logo saved to: {output_path}")
        return canvas
    
    def create_modern_logo(self, text, tagline=None,
                          primary_color=None,  # None for gradient with transparency
                          secondary_color=None,
                          gradient_direction="vertical",
                          text_color=(255, 255, 255, 255),
                          tagline_color=(200, 200, 200, 200),
                          output_size=(800, 400),
                          add_shadow=True,
                          output_path="modern_logo.png"):
        """
        Create a modern gradient logo with optional transparent gradient
        """
        # Create canvas with transparent background
        canvas = Image.new('RGBA', output_size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(canvas)
        
        # Create gradient background
        if primary_color and secondary_color:
            # Ensure colors have alpha channel
            if len(primary_color) == 3:
                primary_color = primary_color + (255,)
            if len(secondary_color) == 3:
                secondary_color = secondary_color + (255,)
            
            # Create gradient
            for i in range(output_size[1] if gradient_direction == "vertical" else output_size[0]):
                ratio = i / (output_size[1] if gradient_direction == "vertical" else output_size[0])
                r = int(primary_color[0] * (1 - ratio) + secondary_color[0] * ratio)
                g = int(primary_color[1] * (1 - ratio) + secondary_color[1] * ratio)
                b = int(primary_color[2] * (1 - ratio) + secondary_color[2] * ratio)
                a = int(primary_color[3] * (1 - ratio) + secondary_color[3] * ratio)
                
                if gradient_direction == "vertical":
                    draw.line([(0, i), (output_size[0], i)], fill=(r, g, b, a))
                else:
                    draw.line([(i, 0), (i, output_size[1])], fill=(r, g, b, a))
        else:
            # Create subtle transparent gradient
            for i in range(output_size[1]):
                alpha = int(150 * (i / output_size[1]))
                draw.line([(0, i), (output_size[0], i)], fill=(255, 255, 255, alpha))
        
        # Load fonts
        title_font = self._load_font(72, bold=True)
        tagline_font = self._load_font(24, bold=False)
        
        # Calculate text positions
        title_bbox = draw.textbbox((0, 0), text, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]
        
        title_x = (output_size[0] - title_width) // 2
        title_y = output_size[1] // 3 - title_height // 2
        
        # Add shadow if requested
        if add_shadow:
            shadow_alpha = 100
            draw.text((title_x + 3, title_y + 3), text, 
                     fill=(0, 0, 0, shadow_alpha), font=title_font)
        
        # Add main text
        draw.text((title_x, title_y), text, fill=text_color, font=title_font)
        
        # Add tagline if provided
        if tagline:
            tagline_bbox = draw.textbbox((0, 0), tagline, font=tagline_font)
            tagline_width = tagline_bbox[2] - tagline_bbox[0]
            
            tagline_x = (output_size[0] - tagline_width) // 2
            tagline_y = title_y + title_height + 20
            
            # Add tagline background
            tagline_bg_padding = 5
            draw.rounded_rectangle(
                [tagline_x - tagline_bg_padding, tagline_y - tagline_bg_padding,
                 tagline_x + tagline_width + tagline_bg_padding, 
                 tagline_y + tagline_bbox[3] - tagline_bbox[1] + tagline_bg_padding],
                radius=3,
                fill=(0, 0, 0, 50)
            )
            
            draw.text((tagline_x, tagline_y), tagline, 
                     fill=tagline_color, font=tagline_font)
        
        # Add decorative element
        decor_height = 3
        decor_y = tagline_y + 40 if tagline else title_y + title_height + 40
        draw.rectangle([(output_size[0]//4, decor_y), 
                       (3*output_size[0]//4, decor_y + decor_height)], 
                      fill=(255, 255, 255, 180))
        
        # Save with transparency
        canvas.save(output_path, 'PNG', optimize=True)
        print(f"Modern logo saved to: {output_path}")
        return canvas
    
    def batch_resize_logos(self, input_folder, output_folder, sizes='all'):
        """
        Batch resize logos while maintaining transparency
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        # Common logo sizes
        standard_sizes = {
            'favicon': (16, 16),
            'icon_small': (32, 32),
            'icon_medium': (64, 64),
            'icon_large': (128, 128),
            'social_media': (400, 400),
            'website_header': (800, 200),
            'print_small': (1000, 1000),
            'print_large': (2000, 2000)
        }
        
        sizes_to_use = standard_sizes if sizes == 'all' else sizes
        
        # Process images
        for img_file in input_path.glob('*'):
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                try:
                    img = Image.open(img_file)
                    # Convert to RGBA for transparency
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    
                    for size_name, dimensions in sizes_to_use.items():
                        # Resize while maintaining aspect ratio
                        img_copy = img.copy()
                        img_copy.thumbnail(dimensions, Image.Resampling.LANCZOS)
                        
                        # Create new image with transparent background
                        new_img = Image.new('RGBA', dimensions, (255, 255, 255, 0))
                        
                        # Center the resized image
                        paste_x = (dimensions[0] - img_copy.width) // 2
                        paste_y = (dimensions[1] - img_copy.height) // 2
                        new_img.paste(img_copy, (paste_x, paste_y), img_copy)
                        
                        # Save
                        output_file = output_path / f"{img_file.stem}_{size_name}.png"
                        new_img.save(output_file, 'PNG', optimize=True)
                        print(f"Saved: {output_file}")
                        
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
    
    def preview_logo(self, logo_image, title="Logo Preview", background=None):
        """
        Preview the logo on different backgrounds
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Test on different backgrounds
        backgrounds = [
            (255, 255, 255, 255),  # White
            (0, 0, 0, 255),        # Black
            (200, 200, 200, 255)   # Gray
        ]
        
        titles = ["White Background", "Black Background", "Gray Background"]
        
        for idx, (bg_color, ax, bg_title) in enumerate(zip(backgrounds, axes, titles)):
            # Create background
            if len(bg_color) == 3:
                bg_color = bg_color + (255,)
            
            preview = Image.new('RGBA', logo_image.size, bg_color)
            preview.paste(logo_image, (0, 0), logo_image)
            
            ax.imshow(preview)
            ax.set_title(bg_title)
            ax.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def get_logo_specifications(self, logo_image):
        """
        Get detailed specifications of the created logo
        """
        # Count transparent pixels
        if logo_image.mode == 'RGBA':
            data = np.array(logo_image)
            alpha_channel = data[:, :, 3]
            transparent_pixels = np.sum(alpha_channel < 10)
            total_pixels = logo_image.width * logo_image.height
            transparency_percentage = (transparent_pixels / total_pixels) * 100
        else:
            transparency_percentage = 0
        
        specs = {
            'size': logo_image.size,
            'mode': logo_image.mode,
            'transparency': f"{transparency_percentage:.1f}%",
            'has_transparency': logo_image.mode in ['RGBA', 'LA', 'PA'],
            'recommended_usage': self._get_recommended_usage(logo_image.size),
            'file_size_estimate': self._estimate_file_size(logo_image)
        }
        
        print("\n" + "="*60)
        print("LOGO SPECIFICATIONS")
        print("="*60)
        for key, value in specs.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        return specs
    
    def _get_recommended_usage(self, size):
        """Get recommended usage based on logo size"""
        width, height = size
        
        if width <= 64:
            return "Favicon, small icons"
        elif width <= 256:
            return "Mobile app icons, social media"
        elif width <= 800:
            return "Website headers, presentations"
        elif width <= 2000:
            return "Print materials, large displays"
        else:
            return "Billboards, very large prints"
    
    def _estimate_file_size(self, image):
        """Estimate file size in KB"""
        if image.mode == 'RGB':
            bytes_per_pixel = 3
        elif image.mode == 'RGBA':
            bytes_per_pixel = 4
        elif image.mode == 'L':
            bytes_per_pixel = 1
        else:
            bytes_per_pixel = 3
        
        estimated_bytes = image.width * image.height * bytes_per_pixel
        return f"{estimated_bytes / 1024:.1f} KB"
    
    def save_as_svg(self, logo_image, output_path="logo.svg"):
        """
        Save logo as SVG (vector format) - basic implementation
        Note: This is a simplified SVG export
        """
        width, height = logo_image.size
        
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <!-- Add definitions here if needed -->
  </defs>
  
  <!-- Background rectangle -->
  <rect width="100%" height="100%" fill="none"/>
  
  <!-- Note: For complex logos, consider converting to vector in dedicated software -->
  <!-- This SVG contains only a placeholder. Export as PNG for detailed logos -->
  
  <text x="50%" y="50%" font-family="Arial" font-size="40" 
        text-anchor="middle" dominant-baseline="middle" fill="black">
    [Vector conversion required]
  </text>
</svg>'''
        
        with open(output_path, 'w') as f:
            f.write(svg_content)
        
        print(f"SVG template saved to: {output_path}")
        print("Note: For best results, recreate logo in vector format using design software")

# =============================================================================
# EXAMPLE USAGE WITH TRANSPARENT BACKGROUNDS
# =============================================================================

def main():
    print("LOGO MAKER WITH TRANSPARENT BACKGROUNDS")
    print("=" * 60)
    
    # Create instance
    logo_maker = LogoMaker()
    
    # Example 1: Transparent text logo
    print("\n1. Creating transparent text logo...")
    transparent_logo = logo_maker.create_simple_logo(
        text="MyBrand",
        output_size=(500, 300),
        bg_color=None,  # Transparent background
        text_color=(41, 128, 185, 255),  # Blue with full opacity
        font_size=60,
        shadow=True,
        shadow_color=(0, 0, 0, 50),
        output_path="transparent_text_logo.png"
    )
    
    if transparent_logo:
        logo_maker.preview_logo(transparent_logo, "Transparent Text Logo")
        logo_maker.get_logo_specifications(transparent_logo)
    
    # Example 2: Convert image with transparent background
    print("\n2. Creating image logo with transparent background...")
    # Load an image first (uncomment and set your image path)
    # if logo_maker.load_image("your_logo.png"):
    #     image_logo = logo_maker.image_to_logo(
    #         output_size=(400, 400),
    #         remove_background=True,
    #         background_color=None,  # Transparent
    #         border=True,
    #         border_color=(255, 255, 255, 180),  # Semi-transparent white
    #         border_width=15,
    #         add_text="Premium",
    #         text_color=(255, 255, 255, 255),
    #         text_position="bottom",
    #         output_path="transparent_image_logo.png"
    #     )
    #     
    #     if image_logo:
    #         logo_maker.preview_logo(image_logo, "Transparent Image Logo")
    
    # Example 3: Transparent icon logo
    print("\n3. Creating transparent icon logo...")
    # Load an image first
    # if logo_maker.load_image("your_icon.png"):
    #     icon_logo = logo_maker.create_icon_logo(
    #         output_size=(512, 512),
    #         shape="circle",
    #         padding=40,
    #         shadow=True,
    #         shadow_blur=10,
    #         shadow_offset=(0, 5),
    #         output_path="transparent_icon_logo.png"
    #     )
    #     
    #     if icon_logo:
    #         logo_maker.preview_logo(icon_logo, "Transparent Icon Logo")
    
    # Example 4: Modern logo with transparent gradient
    print("\n4. Creating modern logo with transparent gradient...")
    modern_logo = logo_maker.create_modern_logo(
        text="INNOVATE",
        tagline="Design with Purpose",
        primary_color=(46, 204, 113, 200),  # Green with transparency
        secondary_color=(52, 152, 219, 100),  # Blue with transparency
        gradient_direction="vertical",
        text_color=(255, 255, 255, 255),
        tagline_color=(255, 255, 255, 180),
        output_size=(800, 400),
        add_shadow=True,
        output_path="transparent_modern_logo.png"
    )
    
    if modern_logo:
        logo_maker.preview_logo(modern_logo, "Modern Logo with Transparency")
    
    # Example 5: Batch process with transparency
    print("\n5. Batch processing logos (example)...")
    # logo_maker.batch_resize_logos(
    #     input_folder="./input_logos",
    #     output_folder="./resized_transparent_logos",
    #     sizes=['favicon', 'icon_small', 'social_media']
    # )
    
    print("\n" + "="*60)
    print("TRANSPARENT LOGO CREATION COMPLETE!")
    print("All logos maintain transparent backgrounds.")
    print("="*60)

# =============================================================================
# TRANSPARENCY TIPS AND BEST PRACTICES
# =============================================================================

def print_transparency_tips():
    """
    Print tips for working with transparent logos
    """
    print("\n" + "="*60)
    print("TRANSPARENT LOGO BEST PRACTICES")
    print("="*60)
    
    tips = [
        "1. ALWAYS SAVE AS PNG: PNG supports transparency, JPEG does not",
        "2. TEST ON DIFFERENT BACKGROUNDS: Ensure readability on light/dark backgrounds",
        "3. ADD SHADOWS/DROP SHADOWS: Helps logos stand out on various backgrounds",
        "4. KEEP ORIGINAL WITH TRANSPARENCY: Maintain a master PNG with transparency",
        "5. USE RGBA COLOR MODE: Ensure alpha channel is preserved",
        "6. AVOID THIN LINES: They may disappear on similar backgrounds",
        "7. ADD A BORDER: Semi-transparent borders help visibility",
        "8. CHECK AT DIFFERENT SIZES: Ensure transparency works at all sizes",
        "9. USE VECTOR FORMATS WHEN POSSIBLE: SVG for web, EPS for print",
        "10. DOCUMENT COLOR VALUES: Note RGB/RGBA values for consistency"
    ]
    
    for tip in tips:
        print(tip)
    
    print("\nCOMMON PITFALLS TO AVOID:")
    print("- Saving as JPEG (loses transparency)")
    print("- Using pure white/black on transparent backgrounds")
    print("- Forgetting to test on real-world backgrounds")
    print("- Not having a solid color fallback version")

if __name__ == "__main__":
    # Run examples
    main()
    
    # Print transparency tips
    print_transparency_tips()