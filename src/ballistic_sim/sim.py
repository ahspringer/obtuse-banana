from environment import Environment, Target
from bullets import Bullet, twist_rate_convert
import numpy as np
import matplotlib.pyplot as plt

def run_simulation(environment, bullet, target, dt=0.001, max_time=15.0):
    """
    Docstring for run_simulation
    
    :param environment: Description
    :param bullet: Description
    :param target: Description
    :param time: Description
    """
    t = 0.0  # Start the clock
    hit = False

    # Default result
    result = {
        'status': 'IN_FLIGHT',  # IN_FLIGHT, HIT, MISS, GROUND, TIMEOUT
        'dy': 0.0,              # Displacement Y off target center
        'dz': 0.0,              # Displacement Z off target center
        'range': 0.0,
        'flight_time': 0.0
    }

    print(' --- STARTING SIMULATION ---')
    while t < max_time and bullet.status == 0:
        bullet.step(dt=dt, env=environment, hit=hit)  # Step the bullet physics

        # Check target logic for hit/miss
        status, dy, dz = target.check_interaction(bullet)
        if status == 1:  # HIT TARGET
            hit = True
            result.update({
                'status': 'HIT',
                'dy': dy,  # Displacement Y off target center
                'dz': dz,  # Displacement Z off target center
                'range': bullet.state[0],
                'flight_time': t
            })
            bullet.status = 1  # Bullet no longer in-flight
            print(' --- SIMULATION ENDED: HIT ---')
            return bullet, result
        
        elif status == -1:  # MISS TARGET
            hit = False
            result.update({
                'status': 'MISS',
                'dy': dy,
                'dy': dz
            })
            # Continue loop to see where bullet lands...

        # Check to see if bullet hit ground
        if bullet.status == 2:
            if result['status'] == 'IN_FLIGHT':
                result['status'] = 'GROUND'  # Let the user know the bullet hit the ground before crossing the target plane

            result['range'] = bullet.state[0]
            result['flight_time'] = t

            print(' --- SIMULATION ENDED: GROUND ---')
            return bullet, result
        
        t += dt  # Step to the next time
        
    # If simulation runs out of time
    result['status'] = 'TIMEOUT'
    result['range'] = bullet.state[0]
    result['flight_time'] = t
    print(' --- SIMULATION ENDED: TIMEOUT ---')
    return bullet, result

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(bullet, target, dt):
    """
    Plots the bullet flight data in a 3x3 grid with wind vector fields and coordinate inversions.
    
    :param bullet: The Bullet object containing .history and .env
    :param target: Target object with .position [x, y, z] and .radius
    :param dt: Time step used in simulation for time-axis reconstruction
    """
    history = np.array(bullet.history)
    num_steps = history.shape[0]
    time = np.linspace(0, (num_steps - 1) * dt, num_steps)
    wind = bullet.env.wind_m_s

    # Extract components
    x, y, z = history[:, 0], history[:, 1], history[:, 2]
    vx, vy, vz = history[:, 3], history[:, 4], history[:, 5]
    spin = history[:, 6]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Bullet Ballistics Analysis (X-Fwd, Y-Right, Z-Down)", fontsize=16)

    # Row 1: Positions vs Time
    axes[0, 0].plot(time, x, color='blue')
    axes[0, 0].set_title("Range (X) vs Time")
    axes[0, 0].set_ylabel("Meters")

    axes[0, 1].plot(time, y, color='green')
    axes[0, 1].set_title("Drift (Y) vs Time")
    axes[0, 1].set_ylabel("Meters (Pos=Right)")
    axes[0, 1].invert_yaxis()

    axes[0, 2].plot(time, z, color='red')
    axes[0, 2].set_title("Drop (Z) vs Time")
    axes[0, 2].set_ylabel("Meters (Pos=Down)")
    axes[0, 2].invert_yaxis()

    # Row 2: Velocities vs Time
    axes[1, 0].plot(time, vx, color='blue')
    axes[1, 0].set_title("Vx vs Time")
    axes[1, 0].set_ylabel("m/s")

    axes[1, 1].plot(time, vy, color='green')
    axes[1, 1].set_title("Vy vs Time")
    axes[1, 1].set_ylabel("m/s")
    axes[1, 1].invert_yaxis()

    # Row 3, Col 1: Spin vs Time
    axes[2, 0].plot(time, spin, color='purple')
    axes[2, 0].set_title("Spin Rate vs Time")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_ylabel("rad/s")

    # Helper function to draw wind field
    def draw_wind_field(ax, x_data, y_data, w_u, w_v, inverted=False):
        # Create a grid of points covering the plot area
        x_min, x_max = x_data.min(), x_data.max()
        y_min, y_max = y_data.min(), y_data.max()
        
        # Add padding to grid
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # If the range is zero (start of sim), use defaults
        if x_range == 0: x_range = 1000
        if y_range == 0: y_range = 10
        
        grid_x, grid_y = np.meshgrid(
            np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 15),
            np.linspace(y_min - 5*y_range, y_max + 5*y_range, 10)
        )
        
        # FIX: If the Y-axis is inverted (Right=Down), we must negate the V component
        # so the arrow points in the correct visual direction.
        v_comp = -w_v if inverted else w_v

        # Plot vector field in light blue
        ax.quiver(grid_x, grid_y, w_u, v_comp, 
                  color='lightblue', alpha=0.5, width=0.003, 
                  label='Wind Field' if ax == axes[2, 1] else "")

    # Row 3, Col 2: Side View (X vs Z) with Wind Field
    # Z is inverted (Positive Down = Down on graph), so negate Z wind component
    draw_wind_field(axes[2, 1], x, z, wind[0], wind[2], inverted=True)
    axes[2, 1].plot(x, z, color='black', linewidth=2, label='Path')
    axes[2, 1].scatter(target.position[0], target.position[2], color='red', marker='x', label='Target', zorder=5)
    
    t_circle_side = plt.Circle((target.position[0], target.position[2]), target.radius, 
                                color='red', fill=False, linestyle='--', zorder=5)
    axes[2, 1].add_patch(t_circle_side)
    
    axes[2, 1].set_title("Side View (X-Z)")
    axes[2, 1].set_xlabel("Range (m)")
    axes[2, 1].set_ylabel("Drop (m)")
    axes[2, 1].invert_yaxis()
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend(loc='upper right', fontsize='small')

    # Row 3, Col 3: Top View (X vs Y) with Wind Field
    # Y is inverted (Positive Right = Down on graph), so negate Y wind component
    draw_wind_field(axes[2, 2], x, y, wind[0], wind[1], inverted=True)
    axes[2, 2].plot(x, y, color='black', linewidth=2, label='Path')
    axes[2, 2].scatter(target.position[0], target.position[1], color='red', marker='x', label='Target', zorder=5)
    
    t_circle_top = plt.Circle((target.position[0], target.position[1]), target.radius, 
                               color='red', fill=False, linestyle='--', zorder=5)
    axes[2, 2].add_patch(t_circle_top)
    
    axes[2, 2].set_title("Top View (X-Y)")
    axes[2, 2].set_xlabel("Range (m)")
    axes[2, 2].set_ylabel("Drift (m)")
    axes[2, 2].invert_yaxis()
    axes[2, 2].grid(True, alpha=0.3)
    axes[2, 2].legend(loc='upper right', fontsize='small')

    for ax in axes.flat:
        ax.grid(alpha=0.2)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_target_impact(target):
    """
    Shows the 'bullet hole' on the target face (Y-Z plane).
    """
    if not target.has_been_hit:
        print("No hit recorded on target; cannot plot impact.")
        return
    else:
        dy = target.dy_at_hit
        dz = target.dz_at_hit
    fig, ax = plt.subplots(figsize=(7, 7))
    
    target_circle = plt.Circle((0, 0), target.radius, color='black', fill=False, linewidth=3)
    bullseye = plt.Circle((0, 0), target.radius * 0.1, color='red', alpha=0.3)
    ax.add_patch(target_circle)
    ax.add_patch(bullseye)
    
    ax.scatter(dy, dz, color='red', s=100, marker='o', edgecolors='black', label='Impact Point', zorder=10)
    
    ax.annotate(f"dy: {dy:.4f}m\ndz: {dz:.4f}m", (dy, dz), xytext=(10, 10), 
                textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax.set_aspect('equal')
    limit = target.radius * 1.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    ax.invert_yaxis()
    ax.set_xlabel('Horizontal Deviation (m) [+Y Right]')
    ax.set_ylabel('Vertical Deviation (m) [+Z Down]')
    ax.set_title(f'Target Impact Analysis (Face View)\nRange: {target.position[0]}m')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.show()

if __name__ == "__main__":
    # 1 m/s = 2.23694 mph = 1.94384 knots
    # 1. Define the environment (Wind 0 m/s)
    environment = Environment(altitude_m=0.0,
                              ground=0.0,
                              temperature='standard',
                              wind_m_s=[3, 3, 0])
    
    # 2. Define the target (1000m downrange, 1 meter diameter)
    target = Target(x=1000, y=0, z=-5, radius=0.5)

    # 3. Define the bullet (308 Win 168 gr ELD® Match)
    bullet308 = Bullet(mass=168,
                       diameter=7.82,  # 0.308"
                       bc_g7=0.5,  # guessed
                       muzzle_vel=2700,
                       twist_rate=twist_rate_convert(10),
                       environment=environment,
                       initial_pos=[0, 0, -5],
                       initial_orientation=[0, 0.03, 0])
    
    bullet, result = run_simulation(environment=environment,
                                    bullet=bullet308,
                                    target=target,
                                    dt=0.001)
    print(result)
    bullet.print_state()

    plot_trajectory(bullet, target, dt=0.001)
    plot_target_impact(target)