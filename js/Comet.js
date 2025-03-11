class Comet {
    constructor(canvas) {
      this.canvas = canvas;
      this.reset();
      this.active = false;
      this.trail = [];
      this.maxTrailLength = 50;
    }
    
    reset() {
      this.x = (Math.random() - 0.5) * this.canvas.width * 3;
      this.y = (Math.random() - 0.5) * this.canvas.height * 3;
      this.z = Math.random() * 1000 + 1000;
      
      this.speed = Math.random() * 25 + 30;
      this.size = Math.random() * 2 + 1;
      
      // Direction vector
      const angle = Math.random() * Math.PI * 2;
      this.dx = Math.sin(angle) * 5;
      this.dy = Math.cos(angle) * 5;
      
      this.trail = [];
      this.active = false;
      this.lifetime = Math.floor(Math.random() * 150) + 100;
      this.age = 0;
      
      // Color - blue-white to yellow-white
      const colorBase = Math.random();
      this.r = 220 + colorBase * 35;
      this.g = 220 + colorBase * 35;
      this.b = 255 - colorBase * 55;
    }
    
    activate() {
      this.active = true;
      this.age = 0;
      this.trail = [];
    }
    
    update(speed) {
      if (!this.active) return;
      
      this.age++;
      if (this.age > this.lifetime) {
        this.reset();
        return;
      }
      
      this.z -= this.speed + speed;
      this.x += this.dx;
      this.y += this.dy;
      
      if (this.z <= 0) {
        this.reset();
        return;
      }
      
      // Calculate screen position with perspective
      const screenX = (this.x / this.z) * 1000 + this.canvas.width / 2;
      const screenY = (this.y / this.z) * 1000 + this.canvas.height / 2;
      
      // Store position for trail
      this.trail.unshift({x: screenX, y: screenY, z: this.z});
      
      // Limit trail length
      if (this.trail.length > this.maxTrailLength) {
        this.trail.pop();
      }
      
      // Check if comet is outside the screen
      if (
        screenX < -this.canvas.width ||
        screenX > this.canvas.width * 2 ||
        screenY < -this.canvas.height ||
        screenY > this.canvas.height * 2
      ) {
        this.reset();
      }
    }
    
    draw(ctx) {
      if (!this.active || this.trail.length < 2) return;
      
      // Draw trail
      ctx.beginPath();
      ctx.moveTo(this.trail[0].x, this.trail[0].y);
      
      for (let i = 1; i < this.trail.length; i++) {
        ctx.lineTo(this.trail[i].x, this.trail[i].y);
      }
      
      // Create gradient for trail
      const gradient = ctx.createLinearGradient(
        this.trail[0].x, this.trail[0].y,
        this.trail[this.trail.length - 1].x, this.trail[this.trail.length - 1].y
      );
      
      gradient.addColorStop(0, `rgba(${this.r}, ${this.g}, ${this.b}, 1)`);
      gradient.addColorStop(0.3, `rgba(${this.r * 0.8}, ${this.g * 0.8}, ${this.b}, 0.6)`);
      gradient.addColorStop(1, 'rgba(80, 80, 180, 0)');
      
      ctx.strokeStyle = gradient;
      ctx.lineWidth = this.size * 2;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.stroke();
      
      // Draw comet head
      const headSize = this.size * 3;
      const glow = ctx.createRadialGradient(
        this.trail[0].x, this.trail[0].y, 0,
        this.trail[0].x, this.trail[0].y, headSize * 2
      );
      
      glow.addColorStop(0, `rgba(255, 255, 255, 1)`);
      glow.addColorStop(0.3, `rgba(${this.r}, ${this.g}, ${this.b}, 0.8)`);
      glow.addColorStop(1, 'rgba(0, 0, 0, 0)');
      
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(this.trail[0].x, this.trail[0].y, headSize * 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }