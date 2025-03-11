class Nebula {
    constructor(canvas) {
      this.canvas = canvas;
      this.particles = [];
      this.numParticles = 5;
      this.createParticles();
    }
    
    createParticles() {
      for (let i = 0; i < this.numParticles; i++) {
        this.particles.push({
          x: Math.random() * this.canvas.width,
          y: Math.random() * this.canvas.height,
          size: Math.random() * 300 + 200,
          opacity: Math.random() * 0.1 + 0.05,
          hue: Math.floor(Math.random() * 360),
          phase: Math.random() * Math.PI * 2
        });
      }
    }
    
    update() {
      this.particles.forEach(p => {
        p.phase += 0.003;
        p.opacity = (Math.sin(p.phase) * 0.05) + 0.1;
        
        // Slowly move the nebula
        p.x += Math.sin(p.phase * 0.5) * 0.2;
        p.y += Math.cos(p.phase * 0.3) * 0.2;
        
        // Wrap around edges
        if (p.x < -p.size) p.x = this.canvas.width + p.size;
        if (p.x > this.canvas.width + p.size) p.x = -p.size;
        if (p.y < -p.size) p.y = this.canvas.height + p.size;
        if (p.y > this.canvas.height + p.size) p.y = -p.size;
      });
    }
    
    draw(ctx) {
      this.particles.forEach(p => {
        const gradient = ctx.createRadialGradient(
          p.x, p.y, 0,
          p.x, p.y, p.size
        );
        
        const hue = (p.hue + (Date.now() / 100) % 360) % 360;
        
        gradient.addColorStop(0, `hsla(${hue}, 100%, 70%, ${p.opacity})`);
        gradient.addColorStop(0.5, `hsla(${(hue + 30) % 360}, 100%, 50%, ${p.opacity * 0.5})`);
        gradient.addColorStop(1, `hsla(${(hue + 60) % 360}, 100%, 40%, 0)`);
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }