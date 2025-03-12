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
    
    // Game state variables
    let hitCounter = 0;
    // Expose hit counter to window object for the UI
    window.hitCounter = hitCounter;
    let gamePhase = 0; // 0: normal, 1: chase blue, 2: chase red, 3: chase rainbow, 4: explosion
    let explosionTime = 0;
    let explosionParticles = [];
    
    // Timer variables
    let gameStartTime = Date.now();
    let gameTime = 0;
    let finalTime = 0;
    let timerInterval = null;
    
    // Start the timer
    function startTimer() {
      gameStartTime = Date.now();
      timerInterval = setInterval(updateTimer, 1000);
      updateTimer(); // Update immediately
    }
    
    // Update the timer display
    function updateTimer() {
      if (gamePhase < 4) { // Only update if game is not over
        gameTime = Math.floor((Date.now() - gameStartTime) / 1000);
        const minutes = Math.floor(gameTime / 60).toString().padStart(2, '0');
        const seconds = (gameTime % 60).toString().padStart(2, '0');
        document.getElementById('time').textContent = `${minutes}:${seconds}`;
        // Expose game time to window object for the UI
        window.gameTime = gameTime;
      }
    }
    
    // Stop the timer and show final time
    function stopTimer() {
      clearInterval(timerInterval);
      finalTime = gameTime;
      // Expose final time to window object for the UI
      window.finalTime = finalTime;
      
      // Format final time
      const minutes = Math.floor(finalTime / 60).toString().padStart(2, '0');
      const seconds = (finalTime % 60).toString().padStart(2, '0');
      
      // Update final time display
      document.getElementById('final-time-value').textContent = `${minutes}:${seconds}`;
      document.getElementById('final-time').style.display = 'block';
    }
    
    // State with default values - start with lower values
    let stars = [];
    let comets = [];
    let nebula = new Nebula(canvas);
    let speed = 5; // Start with lower speed (was 10)
    let starCount = 300; // Start with lower density (was 500)
    let colorMode = 'blue'; // Default color mode
    let lastFrameTime = 0;
    let mouseX = 0;
    let mouseY = 0;
    let lastMouseX = 0;
    let lastMouseY = 0;
    let isMouseMoving = false;
    let mouseMovementTimer = null;
    
    // Maximum values for speed and star count
    const MAX_SPEED = 15;
    const MAX_STAR_COUNT = 700;
    
    // Collision detection variables
    const collisionRadius = 60; // Increased from 40 to 60 to match larger cursor
    const deflectionForce = 2.5; // Slightly increased from 2 to 2.5 for more dramatic effect
    
    // Trail particles for the spaceship
    let trailParticles = [];
    const MAX_TRAIL_PARTICLES = 30;
    
    // Special stars (power-ups)
    let specialStars = [];
    let lastSpecialStarTime = 0;
    
    // Sound effects
    let soundEnabled = false;
    let sounds = {};
    
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
    
    // Load sound effects
    function loadSounds() {
      // Create audio context
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      const audioContext = new AudioContext();
      
      // Define sound effects
      const soundEffects = {
        collision: {
          url: 'sounds/collision.mp3',
          buffer: null
        },
        shield: {
          url: 'sounds/shield.mp3',
          buffer: null
        },
        reduce: {
          url: 'sounds/reduce.mp3',
          buffer: null
        },
        phaseChange: {
          url: 'sounds/phase-change.mp3',
          buffer: null
        },
        explosion: {
          url: 'sounds/explosion.mp3',
          buffer: null
        }
      };
      
      // Create fallback sounds using oscillators if files aren't available
      sounds = {
        collision: () => {
          if (!soundEnabled) return;
          
          const oscillator = audioContext.createOscillator();
          const gainNode = audioContext.createGain();
          
          oscillator.type = 'sine';
          oscillator.frequency.setValueAtTime(220, audioContext.currentTime);
          oscillator.frequency.exponentialRampToValueAtTime(110, audioContext.currentTime + 0.2);
          
          gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
          gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);
          
          oscillator.connect(gainNode);
          gainNode.connect(audioContext.destination);
          
          oscillator.start();
          oscillator.stop(audioContext.currentTime + 0.2);
        },
        shield: () => {
          if (!soundEnabled) return;
          
          const oscillator = audioContext.createOscillator();
          const gainNode = audioContext.createGain();
          
          oscillator.type = 'sine';
          oscillator.frequency.setValueAtTime(440, audioContext.currentTime);
          oscillator.frequency.exponentialRampToValueAtTime(880, audioContext.currentTime + 0.3);
          
          gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
          gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
          
          oscillator.connect(gainNode);
          gainNode.connect(audioContext.destination);
          
          oscillator.start();
          oscillator.stop(audioContext.currentTime + 0.3);
        },
        reduce: () => {
          if (!soundEnabled) return;
          
          const oscillator = audioContext.createOscillator();
          const gainNode = audioContext.createGain();
          
          oscillator.type = 'triangle';
          oscillator.frequency.setValueAtTime(660, audioContext.currentTime);
          oscillator.frequency.exponentialRampToValueAtTime(330, audioContext.currentTime + 0.4);
          
          gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
          gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.4);
          
          oscillator.connect(gainNode);
          gainNode.connect(audioContext.destination);
          
          oscillator.start();
          oscillator.stop(audioContext.currentTime + 0.4);
        },
        phaseChange: () => {
          if (!soundEnabled) return;
          
          const oscillator1 = audioContext.createOscillator();
          const oscillator2 = audioContext.createOscillator();
          const gainNode = audioContext.createGain();
          
          oscillator1.type = 'sine';
          oscillator1.frequency.setValueAtTime(330, audioContext.currentTime);
          oscillator1.frequency.exponentialRampToValueAtTime(660, audioContext.currentTime + 0.5);
          
          oscillator2.type = 'sine';
          oscillator2.frequency.setValueAtTime(440, audioContext.currentTime);
          oscillator2.frequency.exponentialRampToValueAtTime(880, audioContext.currentTime + 0.5);
          
          gainNode.gain.setValueAtTime(0.2, audioContext.currentTime);
          gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
          
          oscillator1.connect(gainNode);
          oscillator2.connect(gainNode);
          gainNode.connect(audioContext.destination);
          
          oscillator1.start();
          oscillator2.start();
          oscillator1.stop(audioContext.currentTime + 0.5);
          oscillator2.stop(audioContext.currentTime + 0.5);
        },
        explosion: () => {
          if (!soundEnabled) return;
          
          const oscillator = audioContext.createOscillator();
          const gainNode = audioContext.createGain();
          const noiseNode = audioContext.createBufferSource();
          
          // Create noise
          const bufferSize = audioContext.sampleRate;
          const buffer = audioContext.createBuffer(1, bufferSize, audioContext.sampleRate);
          const data = buffer.getChannelData(0);
          
          for (let i = 0; i < bufferSize; i++) {
            data[i] = Math.random() * 2 - 1;
          }
          
          noiseNode.buffer = buffer;
          
          oscillator.type = 'sawtooth';
          oscillator.frequency.setValueAtTime(100, audioContext.currentTime);
          oscillator.frequency.exponentialRampToValueAtTime(50, audioContext.currentTime + 1);
          
          gainNode.gain.setValueAtTime(0.5, audioContext.currentTime);
          gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 1);
          
          oscillator.connect(gainNode);
          noiseNode.connect(gainNode);
          gainNode.connect(audioContext.destination);
          
          oscillator.start();
          noiseNode.start();
          oscillator.stop(audioContext.currentTime + 1);
          noiseNode.stop(audioContext.currentTime + 1);
        }
      };
      
      // Add sound toggle button
      const soundButton = document.createElement('div');
      soundButton.className = 'sound-button';
      soundButton.innerHTML = soundEnabled ? '<i class="fas fa-volume-up"></i>' : '<i class="fas fa-volume-mute"></i>';
      document.body.appendChild(soundButton);
      
      soundButton.addEventListener('click', () => {
        soundEnabled = !soundEnabled;
        soundButton.innerHTML = soundEnabled ? '<i class="fas fa-volume-up"></i>' : '<i class="fas fa-volume-mute"></i>';
        
        // Resume audio context if it was suspended
        if (soundEnabled && audioContext.state === 'suspended') {
          audioContext.resume();
        }
      });
    }
    
    // Create explosion particles
    function createExplosion() {
      const particleCount = 150;
      explosionParticles = [];
      
      for (let i = 0; i < particleCount; i++) {
        const angle = Math.random() * Math.PI * 2;
        const speed = Math.random() * 10 + 5;
        
        explosionParticles.push({
          x: mouseX,
          y: mouseY,
          vx: Math.cos(angle) * speed,
          vy: Math.sin(angle) * speed,
          size: Math.random() * 4 + 2,
          color: `hsl(${Math.random() * 60 + 10}, 100%, 60%)`,
          life: Math.random() * 0.5 + 0.5
        });
      }
      
      explosionTime = Date.now();
      
      // Play explosion sound
      sounds.explosion();
      
      // Hide spaceship cursor
      spaceshipCursor.style.display = 'none';
      
      // Show normal cursor
      document.body.style.cursor = 'default';
      canvas.style.cursor = 'default';
      
      // Set game phase to explosion but keep stars chasing
      gamePhase = 4;
      colorMode = 'rainbow'; // Keep rainbow scheme in explosion phase
      stars.forEach(star => {
        star.colorMode = colorMode;
        star.baseColor = starColors.getColor(colorMode);
        // Keep chase mode active - don't reset it
      });
      
      // Stop the timer and show final time
      stopTimer();
      
      // Show game over notification with final time
      const minutes = Math.floor(finalTime / 60).toString().padStart(2, '0');
      const seconds = (finalTime % 60).toString().padStart(2, '0');
      showGameOverNotification(`GAME OVER - TIME: ${minutes}:${seconds}`);
    }
    
    // Create a trail particle
    function createTrailParticle(x, y) {
      trailParticles.push({
        x: x,
        y: y,
        size: Math.random() * 3 + 1,
        life: 1.0,
        // Changed all trail particles to yellow with varying opacity
        color: `rgba(255, 215, 0, ${0.6 + Math.random() * 0.4})`
      });
      
      // Limit the number of particles
      if (trailParticles.length > MAX_TRAIL_PARTICLES) {
        trailParticles.shift();
      }
    }
    
    // Update and draw trail particles
    function updateTrailParticles() {
      for (let i = trailParticles.length - 1; i >= 0; i--) {
        const particle = trailParticles[i];
        
        // Reduce life
        particle.life -= 0.05;
        
        // Remove dead particles
        if (particle.life <= 0) {
          trailParticles.splice(i, 1);
          continue;
        }
        
        // Draw particle with yellow glow
        ctx.fillStyle = particle.color.replace(/[\d\.]+\)$/, `${particle.life.toFixed(2)})`);
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size * particle.life, 0, Math.PI * 2);
        ctx.fill();
        
        // Add a subtle glow effect for yellow particles
        if (particle.life > 0.5) {
          ctx.fillStyle = `rgba(255, 255, 150, ${particle.life * 0.3})`;
          ctx.beginPath();
          ctx.arc(particle.x, particle.y, particle.size * particle.life * 2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
    
    // Create a special star (power-up)
    function createSpecialStar() {
      // Only create a special star if enough time has passed since the last one
      if (Date.now() - lastSpecialStarTime < 3000) return; // 3 seconds cooldown
      
      // Create a special star at a random position
      const specialStar = {
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size: 15,
        pulse: 0,
        // Add back the blue power-up with 1/3 probability for each type
        type: Math.random() < 0.33 ? 'shield' : (Math.random() < 0.5 ? 'reduce' : 'clear'),
        active: true
      };
      
      specialStars.push(specialStar);
      lastSpecialStarTime = Date.now();
    }
    
    // Update and draw special stars
    function updateSpecialStars() {
      // Randomly create a special star - adjusted spawn rate to 0.003
      if (gamePhase < 4 && Math.random() < 0.003) {
        createSpecialStar();
      }
      
      // Update and draw special stars
      for (let i = specialStars.length - 1; i >= 0; i--) {
        const star = specialStars[i];
        
        // Skip inactive stars
        if (!star.active) {
          specialStars.splice(i, 1);
          continue;
        }
        
        // Update pulse
        star.pulse = (star.pulse + 0.05) % (Math.PI * 2);
        
        // Draw special star
        const pulseSize = star.size + Math.sin(star.pulse) * 3;
        
        // Draw glow
        const gradient = ctx.createRadialGradient(
          star.x, star.y, 0,
          star.x, star.y, pulseSize * 2
        );
        
        if (star.type === 'shield') {
          // Gold shield star
          gradient.addColorStop(0, 'rgba(255, 215, 0, 0.8)');
          gradient.addColorStop(0.5, 'rgba(255, 215, 0, 0.4)');
          gradient.addColorStop(1, 'rgba(255, 215, 0, 0)');
        } else if (star.type === 'reduce') {
          // Green reduce hits star
          gradient.addColorStop(0, 'rgba(50, 205, 50, 0.8)');
          gradient.addColorStop(0.5, 'rgba(50, 205, 50, 0.4)');
          gradient.addColorStop(1, 'rgba(50, 205, 50, 0)');
        } else if (star.type === 'clear') {
          // Blue clear stars star
          gradient.addColorStop(0, 'rgba(0, 0, 255, 0.8)');
          gradient.addColorStop(0.5, 'rgba(0, 0, 255, 0.4)');
          gradient.addColorStop(1, 'rgba(0, 0, 255, 0)');
        }
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(star.x, star.y, pulseSize * 2, 0, Math.PI * 2);
        ctx.fill();
        
        // Draw star shape
        ctx.fillStyle = star.type === 'shield' ? '#FFD700' : (star.type === 'reduce' ? '#32CD32' : '#0000FF');
        ctx.beginPath();
        
        // Draw a star shape
        for (let j = 0; j < 5; j++) {
          const angle = (j * 2 * Math.PI / 5) - Math.PI / 2;
          const x = star.x + Math.cos(angle) * pulseSize;
          const y = star.y + Math.sin(angle) * pulseSize;
          
          if (j === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
          
          // Inner points
          const innerAngle = angle + Math.PI / 5;
          const innerX = star.x + Math.cos(innerAngle) * (pulseSize * 0.4);
          const innerY = star.y + Math.sin(innerAngle) * (pulseSize * 0.4);
          ctx.lineTo(innerX, innerY);
        }
        
        ctx.closePath();
        ctx.fill();
        
        // Check for collision with spaceship
        const dx = star.x - mouseX;
        const dy = star.y - mouseY;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < collisionRadius + pulseSize && gamePhase < 4) {
          // Collect the power-up
          star.active = false;
          
          // Apply power-up effect
          if (star.type === 'shield') {
            // Shield effect - temporary invincibility
            activateShield();
          } else if (star.type === 'reduce') {
            // Reduce hits effect - decrease hit counter by 5
            reduceHits();
          } else if (star.type === 'clear') {
            // Clear stars effect - remove all stars
            clearStars();
          }
          
          // Create a shockwave effect
          createShockwave(star.x, star.y, 
            star.type === 'shield' ? 'rgba(255, 215, 0, 0.8)' : (star.type === 'reduce' ? 'rgba(50, 205, 50, 0.8)' : 'rgba(0, 0, 255, 0.8)'));
        }
      }
    }
    
    // Shield power-up effect
    let shieldActive = false;
    let shieldEndTime = 0;
    
    function activateShield() {
      shieldActive = true;
      shieldEndTime = Date.now() + 5000; // 5 seconds of invincibility
      
      // Play shield sound
      sounds.shield();
      
      // Show shield notification
      const notification = document.createElement('div');
      notification.className = 'game-notification';
      notification.textContent = 'SHIELD ACTIVATED!';
      notification.style.color = '#FFD700';
      document.body.appendChild(notification);
      
      // Fade out and remove
      setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
          notification.remove();
        }, 1000);
      }, 2000);
    }
    
    // Reduce hits power-up effect
    function reduceHits() {
      // Play reduce sound
      sounds.reduce();
      
      // Reduce hit counter by 5, but not below 0 (changed from 10 to 5)
      hitCounter = Math.max(0, hitCounter - 5);
      
      // Update window.hitCounter for the UI
      window.hitCounter = hitCounter;
      
      // Show reduce notification
      const notification = document.createElement('div');
      notification.className = 'game-notification';
      notification.textContent = '-5 HITS!';  // Changed from -10 to -5
      notification.style.color = '#32CD32';
      document.body.appendChild(notification);
      
      // Fade out and remove
      setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
          notification.remove();
        }, 1000);
      }, 2000);
    }
    
    // Clear stars power-up effect
    function clearStars() {
      // Play a sound for clearing stars
      sounds.phaseChange();
      
      // Store the current star count
      const currentStarCount = stars.length;
      
      // Clear all stars
      stars = [];
      
      // Show clear notification
      const notification = document.createElement('div');
      notification.className = 'game-notification';
      notification.textContent = 'STARS CLEARED!';
      notification.style.color = '#0088FF';
      document.body.appendChild(notification);
      
      // Fade out and remove
      setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
          notification.remove();
        }, 1000);
      }, 2000);
      
      // Gradually bring stars back after 3 seconds
      setTimeout(() => {
        // Recreate stars with the same count
        stars = Array(currentStarCount).fill().map(() => new Star(canvas, { colorMode }));
        
        // Restore chase mode for appropriate game phase
        if (gamePhase >= 1) {
          const chasePercentage = gamePhase === 1 ? 0.4 : (gamePhase === 2 ? 0.6 : 0.8);
          stars.forEach(s => {
            if (gamePhase >= 2) {
              s.colorMode = colorMode;
              s.baseColor = starColors.getColor(colorMode);
            }
            if (Math.random() < chasePercentage) {
              s.chaseMode = true;
            }
          });
        }
      }, 3000);
    }
    
    // Draw shield effect
    function drawShield() {
      if (shieldActive) {
        // Check if shield has expired
        if (Date.now() > shieldEndTime) {
          shieldActive = false;
          return;
        }
        
        // Calculate shield opacity based on remaining time
        const remainingTime = (shieldEndTime - Date.now()) / 5000;
        const opacity = Math.min(0.7, remainingTime * 0.7);
        
        // Draw shield
        ctx.strokeStyle = `rgba(255, 215, 0, ${opacity})`;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(mouseX, mouseY, collisionRadius + 10, 0, Math.PI * 2);
        ctx.stroke();
        
        // Draw inner shield
        ctx.strokeStyle = `rgba(255, 255, 255, ${opacity * 0.7})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(mouseX, mouseY, collisionRadius + 5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    
    // Initialize
    createStars();
    createComets();
    loadSounds();
    startTimer(); // Start the timer when the game initializes
    
    // Update spaceship cursor position
    document.addEventListener('mousemove', (e) => {
      // Don't update cursor if in explosion mode
      if (gamePhase === 4) return;
      
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
      // Skip collision detection if shield is active
      if (shieldActive) {
        return;
      }
      
      stars.forEach(star => {
        // Calculate distance between star and cursor
        const dx = star.screenX - mouseX;
        const dy = star.screenY - mouseY;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // If collision detected
        if (distance < collisionRadius + star.size) {
          // Only count hits from larger stars (displaySize > 10.0)
          if (star.displaySize > 10.0) {
            // Add a cooldown to prevent multiple hits from the same star
            // when the ship is stationary
            if (!star.lastHitTime || Date.now() - star.lastHitTime > 500) {
              // Increment hit counter
              hitCounter++;
              // Update window.hitCounter for the UI
              window.hitCounter = hitCounter;
              
              // Play collision sound
              sounds.collision();
              
              // Record the hit time
              star.lastHitTime = Date.now();
              
              // Create a shockwave effect at the collision point
              createShockwave(star.screenX, star.screenY, `rgba(255, 100, 100, 0.8)`);
              
              // Check for phase transitions
              if (hitCounter === 42 && gamePhase === 0) {
                // Transition to chase mode - blue
                gamePhase = 1;
                
                // Play phase change sound
                sounds.phaseChange();
                
                // Make only a portion of stars chase for better balance
                stars.forEach(s => {
                  // 40% of larger stars will chase
                  if (s.size > 1.0 && Math.random() < 0.4) {
                    s.chaseMode = true;
                  }
                });
                
                // Increase speed and density slightly
                speed = 7;
                increaseStarCount(400);
              } else if (hitCounter === 84 && gamePhase === 1) {
                // Transition to chase mode - red
                gamePhase = 2;
                colorMode = 'red';
                
                // Play phase change sound
                sounds.phaseChange();
                
                // Make more stars chase in red mode
                stars.forEach(s => {
                  s.colorMode = colorMode;
                  s.baseColor = starColors.getColor(colorMode);
                  
                  // 60% of stars will chase in red mode
                  // Preserve existing chasing stars and add more
                  if (!s.chaseMode && s.size > 1.0 && Math.random() < 0.6) {
                    s.chaseMode = true;
                  }
                });
                
                // Increase speed and density more
                speed = 10;
                increaseStarCount(500);
              } else if (hitCounter === 126 && gamePhase === 2) {
                // Transition to rainbow mode
                gamePhase = 3;
                colorMode = 'rainbow';
                
                // Play phase change sound
                sounds.phaseChange();
                
                // Make even more stars chase in rainbow mode
                stars.forEach(s => {
                  s.colorMode = colorMode;
                  s.baseColor = starColors.getColor(colorMode);
                  
                  // 80% of stars will chase in rainbow mode
                  // Preserve existing chasing stars and add more
                  if (!s.chaseMode && Math.random() < 0.8) {
                    s.chaseMode = true;
                  }
                });
                
                // Increase speed and density to maximum
                speed = MAX_SPEED;
                increaseStarCount(MAX_STAR_COUNT);
              } else if (hitCounter === 168 && gamePhase === 3) {
                // Transition to explosion
                createExplosion();
              }
            }
          }
          
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
    
    // Increase star count gradually
    function increaseStarCount(targetCount) {
      const currentCount = stars.length;
      const difference = targetCount - currentCount;
      
      if (difference <= 0) return;
      
      // Add new stars gradually
      const newStars = Array(difference).fill().map(() => new Star(canvas, { colorMode }));
      stars = stars.concat(newStars);
      starCount = stars.length;
    }
    
    // Show game over notification (only one we're keeping)
    function showGameOverNotification(message) {
      const notification = document.createElement('div');
      notification.className = 'game-notification';
      notification.textContent = message;
      document.body.appendChild(notification);
      
      // Fade out and remove after 5 seconds
      setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
          document.body.removeChild(notification);
        }, 1000);
      }, 5000);
    }
    
    // Draw explosion
    function drawExplosion() {
      const currentTime = Date.now();
      const elapsed = (currentTime - explosionTime) / 1000; // seconds
      
      // Update and draw explosion particles
      for (let i = explosionParticles.length - 1; i >= 0; i--) {
        const p = explosionParticles[i];
        
        // Update position
        p.x += p.vx;
        p.y += p.vy;
        
        // Apply gravity and friction
        p.vy += 0.2;
        p.vx *= 0.99;
        p.vy *= 0.99;
        
        // Reduce life
        p.life -= 0.01;
        
        // Remove dead particles
        if (p.life <= 0) {
          explosionParticles.splice(i, 1);
          continue;
        }
        
        // Draw particle
        ctx.fillStyle = p.color;
        ctx.globalAlpha = p.life;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
      }
    }
    
    // Create a visual shockwave effect
    function createShockwave(x, y, color) {
      const shockwave = {
        x: x,
        y: y,
        radius: 5,
        maxRadius: 50,
        color: color || 'rgba(255, 255, 255, 0.8)',
        life: 1.0
      };
      
      // Animate the shockwave
      const animate = () => {
        shockwave.radius += 3;
        shockwave.life -= 0.05;
        
        if (shockwave.life > 0 && shockwave.radius < shockwave.maxRadius) {
          // Draw the shockwave
          ctx.strokeStyle = shockwave.color.replace('0.8', shockwave.life.toFixed(2));
          ctx.lineWidth = 2 * shockwave.life;
          ctx.beginPath();
          ctx.arc(shockwave.x, shockwave.y, shockwave.radius, 0, Math.PI * 2);
          ctx.stroke();
          
          // Continue animation
          requestAnimationFrame(animate);
        }
      };
      
      // Start animation
      animate();
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
      
      // Check for collisions regardless of mouse movement
      // Only skip collision detection in explosion phase
      if (gamePhase !== 4) {
        checkCollisions();
      }
      
      // Check if mouse is moving
      if (Math.abs(mouseX - lastMouseX) > 0.5 || Math.abs(mouseY - lastMouseY) > 0.5) {
        isMouseMoving = true;
        
        // Create trail particles when moving - increased frequency from 0.3 to 0.5
        if (gamePhase !== 4 && Math.random() < 0.5) {
          createTrailParticle(mouseX, mouseY);
        }
      } else {
        isMouseMoving = false;
      }
      
      // Update last mouse position
      lastMouseX = mouseX;
      lastMouseY = mouseY;
      
      if (isMouseMoving && gamePhase !== 4) {
        // Calculate mouse position relative to center
        parallaxOffsetX = ((mouseX - canvas.width / 2) / canvas.width) * 20;
        parallaxOffsetY = ((mouseY - canvas.height / 2) / canvas.height) * 20;
        
        // Parallax effect only applies when mouse is moving
      }
      
      // Clear canvas with a fade effect for motion blur
      ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Draw nebula first (background layer)
      nebula.update();
      nebula.draw(ctx);
      
      // Update and draw trail particles
      updateTrailParticles();
      
      // Update and draw special stars
      updateSpecialStars();
      
      // Update and draw stars
      stars.forEach((star) => {
        // In chase mode, make stars chase the cursor
        if ((gamePhase === 1 || gamePhase === 2 || gamePhase === 3 || gamePhase === 4) && star.chaseMode) {
          // Only larger stars chase effectively
          if (star.size > 1.0) {
            // Calculate direction to cursor
            const dx = mouseX - star.screenX;
            const dy = mouseY - star.screenY;
            const dist = Math.sqrt(dx * dx + dy * dy);
            
            // Apply a force towards cursor that increases with distance
            // This creates a more natural chasing behavior
            if (dist > 0) {
              // Enhanced chase strength for more aggressive behavior
              const chaseStrength = Math.min(dist / 200, 1.5); // Increased from 300 to 200 and max from 1.0 to 1.5
              star.vx += (dx / dist) * chaseStrength;
              star.vy += (dy / dist) * chaseStrength;
              
              // Add slight randomness to make movement more interesting
              star.vx += (Math.random() - 0.5) * 0.2;
              star.vy += (Math.random() - 0.5) * 0.2;
              
              // Increase speed for stars that are far away - more aggressive acceleration
              if (dist > 150) { // Reduced from 200 to 150
                star.vx *= 1.05; // Increased from 1.03 to 1.05
                star.vy *= 1.05;
              }
            }
          }
        }
        
        star.update(speed * timeFactor);
        
        // Apply parallax effect based on star distance
        if (isMouseMoving && gamePhase !== 4) {
          const parallaxFactor = 1 - star.z / 2000;
          star.screenX += parallaxOffsetX * parallaxFactor;
          star.screenY += parallaxOffsetY * parallaxFactor;
        }
        
        star.draw(ctx, gamePhase === 4); // Pass a flag to indicate game over state
      });
      
      // Update and draw comets
      comets.forEach((comet) => {
        comet.update(speed * timeFactor);
        comet.draw(ctx);
      });
      
      // Draw shield effect if active
      drawShield();
      
      // Draw explosion if in explosion phase
      if (gamePhase === 4) {
        drawExplosion();
      }
      
      // Request next frame
      requestAnimationFrame(animate);
    }
    
    // Start animation
    animate(0);
  });