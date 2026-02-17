# Math Drawings

A project that creates artistic images using pure mathematical formulas - no traditional graphics or drawing tools, just mathematical equations evaluated at each pixel.

## Overview

This project generates a beautiful snow-covered village scene entirely from mathematical formulas. Each pixel's RGB color value is computed using complex mathematical expressions involving trigonometric functions, exponentials, and various parameters.

## What's Included

- `gen_math_village_img.py` - Python script that generates the snow village image from mathematical formulas
- `snow_covered_village_original.jpg` - Original reference image that inspired the mathematical recreation
- `math-village-generated.png` - The generated output image (2000x1200 pixels)

## How It Works

The script generates a 2000x1200 pixel image where each pixel's RGB color is defined by three functions H₀(x,y), H₁(x,y), and H₂(x,y). The implementation uses several mathematical components:

### Key Mathematical Components

1. **E(x,y)** - Fractal terrain texture using 40 iterations of scaled cosine functions
2. **B(x,y)** - Background/ground mask that separates sky from terrain
3. **M(x,y)** - Snow silhouette mask that creates the snow-covered effect
4. **Houses (s=1..67)** - 67 individual house structures, each computed with:
   - Coordinate transformations and rotations
   - Window components (J_s)
   - Wall components (K_s)
   - Occlusion handling (Z variable)

### Technical Approach

- Uses NumPy vectorization for efficiency (loops over structures, not pixels)
- Normalized coordinates: x ∈ [-5/3, 5/3], y ∈ [-5/6, 5/6]
- Safe exponential functions with clamping to avoid numerical overflow
- Each house has unique positioning, rotation, and detail based on its index

## Requirements

```bash
pip install numpy pillow
```

## Usage

Simply run the Python script:

```bash
python gen_math_village_img.py
```

The script will:
1. Generate the image (takes several seconds due to the complex calculations)
2. Save the output as `math-village-generated.png`
3. Display progress and timing information

## Output

The generated image features:
- A snow-covered village with 67 houses
- Fractal terrain texture
- Atmospheric perspective
- Windows with lighting effects
- Snow accumulation on roofs
- All created purely from mathematical formulas!

## Performance

Generation takes approximately 10-30 seconds depending on your CPU, with progress updates shown for the house generation phase (the most computationally intensive part).

## Mathematical Art

This project demonstrates how complex, artistic images can be created using nothing but mathematical expressions. The entire scene - from the terrain texture to individual house windows - emerges from carefully crafted formulas evaluated at each pixel coordinate.
