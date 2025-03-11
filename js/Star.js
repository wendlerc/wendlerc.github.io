class Star {
    constructor(canvas, options = {}) {
      this.canvas = canvas;
      this.colorMode = options.colorMode || 'white';
      this.reset();
      this.pulse = Math.random() * 0.2 + 0.8;
      this.pulseSpeed = Math.random() * 0.01 + 0.005;
      this.pulseFactor = 0;
      this.twinklePhase = Math.random() * Math.PI * 2;
      // Add velocity properties for deflection
      this.vx = 0;
      this.vy = 0;
      // Add chase mode property
      this.chaseMode = false;
      // Maximum velocity
      this.maxVelocity = 5;
      // Track last hit time to prevent multiple hits when stationary
      this.lastHitTime = 0;
    }
  
    reset() {
      this.x = (Math.random() - 0.5) * this.canvas.width * 2;
      this.y = (Math.random() - 0.5) * this.canvas.height * 2;
      this.z = Math.random() * 1500 + 500;
      this.origZ = this.z;
      this.size = Math.random() * 2 + 0.5;
      
      // Increase chance of larger stars to make the game more playable
      // 20% chance of being a brighter star (up from 10%)
      if (Math.random() < 0.2) {
        this.size *= 2;
      }
      
      // 5% chance of being a much brighter star (up from 1%)
      if (Math.random() < 0.05) {
        this.size *= 3;
      }
      
      this.baseColor = starColors.getColor(this.colorMode);
      
      // Reset velocity when star is reset
      this.vx = 0;
      this.vy = 0;
      
      // Reset hit time when star is reset
      this.lastHitTime = 0;
    }
  
    update(speed) {
      this.z -= speed;
      if (this.z <= 0) {
        this.reset();
      }
  
      // Apply deflection velocity to position
      this.x += this.vx;
      this.y += this.vy;
      
      // Limit maximum velocity to prevent stars from moving too fast
      const currentVelocity = Math.sqrt(this.vx * this.vx + this.vy * this.vy);
      if (currentVelocity > this.maxVelocity) {
        const scale = this.maxVelocity / currentVelocity;
        this.vx *= scale;
        this.vy *= scale;
      }
      
      // Gradually reduce velocity (friction)
      // Apply even less friction to chasing stars to make them more persistent
      if (this.chaseMode) {
        this.vx *= 0.995;
        this.vy *= 0.995;
      } else {
        this.vx *= 0.95;
        this.vy *= 0.95;
      }
  
      // Calculate screen position with perspective
      this.screenX = (this.x / this.z) * 1000 + this.canvas.width / 2;
      this.screenY = (this.y / this.z) * 1000 + this.canvas.height / 2;
  
      // Calculate size based on z position with more dramatic perspective
      const perspective = Math.pow(1 - this.z / (this.origZ * 2), 1.5);
      this.displaySize = this.size * (perspective * 3 + 0.5);
  
      // Update pulse effect
      this.twinklePhase += this.pulseSpeed;
      this.pulseFactor = (Math.sin(this.twinklePhase) + 1) * 0.5 * this.pulse;
  
      // Check if star is outside the screen with a larger margin
      // This prevents stars from disappearing too quickly when chasing
      const margin = 100;
      if (
        this.screenX < -margin ||
        this.screenX > this.canvas.width + margin ||
        this.screenY < -margin ||
        this.screenY > this.canvas.height + margin
      ) {
        this.reset();
      }
    }
  
    draw(ctx, isGameOver = false) {
      const distanceFactor = 1 - this.z / 2000;
      const opacity = Math.min(distanceFactor * 1.5, 1);
      
      // Apply pulse effect to color and size
      const finalSize = this.displaySize * (0.8 + this.pulseFactor * 0.4);
      
      // Calculate glow size based on star size
      const glowSize = finalSize * (3 + this.pulseFactor * 2);
      
      // Create a glow effect
      const gradient = ctx.createRadialGradient(
        this.screenX, this.screenY, 0,
        this.screenX, this.screenY, glowSize
      );
      
      const { r, g, b } = this.baseColor;
      
      // Make chasing stars more visible with a brighter glow
      const intensityFactor = this.chaseMode ? 1.3 : 1.0;
      
      gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${opacity * intensityFactor})`);
      gradient.addColorStop(0.1, `rgba(${r}, ${g}, ${b}, ${opacity * 0.8 * intensityFactor})`);
      gradient.addColorStop(0.4, `rgba(${r}, ${g}, ${b}, ${opacity * 0.2 * intensityFactor})`);
      gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
      
      // Draw glow
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(this.screenX, this.screenY, glowSize, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw main star
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${opacity * intensityFactor})`;
      ctx.beginPath();
      ctx.arc(this.screenX, this.screenY, finalSize, 0, Math.PI * 2);
      ctx.fill();
      
      // Add highlight in the center for brighter stars
      if (finalSize > 1.5) {
        ctx.fillStyle = `rgba(255, 255, 255, ${opacity * 0.8 * intensityFactor})`;
        ctx.beginPath();
        ctx.arc(this.screenX, this.screenY, finalSize * 0.5, 0, Math.PI * 2);
        ctx.fill();
      }
      
      // Draw red circle around stars that are large enough to register hits
      // Only draw if not in game over state
      if (this.displaySize > 10.0 && !isGameOver) {
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(this.screenX, this.screenY, finalSize + 2, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }