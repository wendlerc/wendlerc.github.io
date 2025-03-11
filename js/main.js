document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('starfield');
    const ctx = canvas.getContext('2d');
    
    // Create spaceship cursor element
    const spaceshipCursor = document.createElement('div');
    spaceshipCursor.className = 'spaceship-cursor';
    document.body.appendChild(spaceshipCursor);
    
    // Set canvas size to window size
    function resizeCanvas() {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }
    
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // State with default values
    let stars = [];
    let comets = [];
    let nebula = new Nebula(canvas);
    let speed = 10; // Default speed
    let starCount = 500; // Default star count
    let colorMode = 'blue'; // Changed from 'rainbow' to 'blue'
    let lastFrameTime = 0;
    let mouseX = 0;
    let mouseY = 0;
    let isMouseMoving = false;
    let mouseMovementTimer = null;
    
    // Collision detection variables
    const collisionRadius = 60; // Increased from 40 to 60 to match larger cursor
    const deflectionForce = 2.5; // Slightly increased from 2 to 2.5 for more dramatic effect
    
    // Create stars
    function createStars() {
      stars = Array(starCount).fill().map(() => new Star(canvas, { colorMode }));
    }
    
    // Create comets
    function createComets() {
      comets = Array(5).fill().map(() => new Comet(canvas));
      
      // Activate one comet to start
      if (comets.length > 0) {
        comets[0].activate();
      }
    }
    
    // Initialize
    createStars();
    createComets();
    
    // Update spaceship cursor position
    document.addEventListener('mousemove', (e) => {
      // Update spaceship cursor position
      spaceshipCursor.style.left = `${e.clientX}px`;
      spaceshipCursor.style.top = `${e.clientY}px`;
      
      // Rotate spaceship to follow mouse movement direction
      if (mouseX !== 0 && mouseY !== 0) {
        const dx = e.clientX - mouseX;
        const dy = e.clientY - mouseY;
        if (Math.abs(dx) > 1 || Math.abs(dy) > 1) {
          const angle = Math.atan2(dy, dx) * (180 / Math.PI);
          spaceshipCursor.style.transform = `translate(-50%, -50%) rotate(${angle + 90}deg)`;
        }
      }
      
      // Update mouse position for parallax and collision
      mouseX = e.clientX;
      mouseY = e.clientY;
      isMouseMoving = true;
      
      clearTimeout(mouseMovementTimer);
      mouseMovementTimer = setTimeout(() => {
        isMouseMoving = false;
      }, 100);
      
      // Randomly activate a comet on mouse movement (2% chance)
      if (Math.random() < 0.02) {
        const inactiveComets = comets.filter(c => !c.active);
        if (inactiveComets.length > 0) {
          const randomComet = inactiveComets[Math.floor(Math.random() * inactiveComets.length)];
          randomComet.activate();
        }
      }
    });
    
    // Check for collisions between cursor and stars
    function checkCollisions() {
      stars.forEach(star => {
        // Calculate distance between star and cursor
        const dx = star.screenX - mouseX;
        const dy = star.screenY - mouseY;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // If collision detected
        if (distance < collisionRadius + star.size) {
          // Calculate deflection angle
          const angle = Math.atan2(dy, dx);
          
          // Apply deflection force
          const deflectionX = Math.cos(angle) * deflectionForce;
          const deflectionY = Math.sin(angle) * deflectionForce;
          
          // Add deflection to star velocity
          star.vx += deflectionX;
          star.vy += deflectionY;
          
          // Add a small visual effect (pulse)
          star.size *= 1.2;
          setTimeout(() => {
            if (star.size > 1) {
              star.size /= 1.2;
            }
          }, 100);
        }
      });
    }
    
    // Animation loop with delta time for consistent speed
    function animate(timestamp) {
      const deltaTime = timestamp - lastFrameTime;
      lastFrameTime = timestamp;
      
      // Calculate fps-independent motion factor
      const timeFactor = deltaTime / 16.67; // 16.67ms is roughly 60fps
      
      // Apply parallax effect based on mouse position
      let parallaxOffsetX = 0;
      let parallaxOffsetY = 0;
      
      if (isMouseMoving) {
        // Calculate mouse position relative to center
        parallaxOffsetX = ((mouseX - canvas.width / 2) / canvas.width) * 20;
        parallaxOffsetY = ((mouseY - canvas.height / 2) / canvas.height) * 20;
        
        // Check for collisions when mouse is moving
        checkCollisions();
      }
      
      // Clear canvas with a fade effect for motion blur
      ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Draw nebula first (background layer)
      nebula.update();
      nebula.draw(ctx);
      
      // Update and draw stars
      stars.forEach((star) => {
        star.update(speed * timeFactor);
        
        // Apply parallax effect based on star distance
        if (isMouseMoving) {
          const parallaxFactor = 1 - star.z / 2000;
          star.screenX += parallaxOffsetX * parallaxFactor;
          star.screenY += parallaxOffsetY * parallaxFactor;
        }
        
        star.draw(ctx);
      });
      
      // Update and draw comets
      comets.forEach((comet) => {
        comet.update(speed * timeFactor);
        comet.draw(ctx);
      });
      
      // Randomly activate comets
      if (Math.random() < 0.002) {
        const inactiveComets = comets.filter(c => !c.active);
        if (inactiveComets.length > 0) {
          const randomComet = inactiveComets[Math.floor(Math.random() * inactiveComets.length)];
          randomComet.activate();
        }
      }
      
      requestAnimationFrame(animate);
    }
    
    // Start animation
    animate(0);
  });