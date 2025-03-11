const starColors = {
    // Color palettes for stars
    palettes: {
      white: [
        { r: 255, g: 255, b: 255 },   // Pure white
        { r: 240, g: 240, b: 255 },   // Slightly blue white
        { r: 255, g: 240, b: 230 },   // Slightly yellow white
        { r: 220, g: 220, b: 255 },   // Pale blue
        { r: 255, g: 220, b: 220 },   // Pale red
      ],
      rainbow: [
        { r: 255, g: 100, b: 100 },   // Red
        { r: 255, g: 200, b: 100 },   // Orange
        { r: 255, g: 255, b: 100 },   // Yellow
        { r: 100, g: 255, b: 100 },   // Green
        { r: 100, g: 200, b: 255 },   // Cyan
        { r: 130, g: 100, b: 255 },   // Blue
        { r: 240, g: 100, b: 255 },   // Purple
      ],
      blue: [
        { r: 100, g: 150, b: 255 },   // Light blue
        { r: 80, g: 120, b: 255 },    // Medium blue
        { r: 60, g: 100, b: 255 },    // Deep blue
        { r: 150, g: 200, b: 255 },   // Sky blue
        { r: 200, g: 220, b: 255 },   // Pale blue
      ],
      red: [
        { r: 255, g: 100, b: 100 },   // Light red
        { r: 255, g: 80, b: 80 },     // Medium red
        { r: 255, g: 60, b: 60 },     // Deep red
        { r: 255, g: 150, b: 100 },   // Orange-red
        { r: 255, g: 200, b: 180 },   // Pale red
      ]
    },
    
    getColor(mode) {
      const palette = this.palettes[mode] || this.palettes.white;
      return palette[Math.floor(Math.random() * palette.length)];
    }
  };