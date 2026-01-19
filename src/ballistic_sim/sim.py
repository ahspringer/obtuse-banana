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

            return bullet, result
        
        t += dt  # Step to the next time
        print(t)
        # bullet.print_state()
        # return bullet, result
        
    # If simulation runs out of time
    result['status'] = 'TIMEOUT'
    result['range'] = bullet.state[0]
    result['flight_time'] = t
    return bullet, result

def plot_trajectory(bullet, target):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    xs = bullet.history[:, 0]
    ys = bullet.history[:, 1]
    zs = bullet.history[:, 2]

    # Side View
    ax1.plot(xs, zs)
    ax1.scatter(target.position[0], target.position[2], c='red', marker='x', s=100)
    ax1.invert_yaxis() # Down is Positive Z
    ax1.set_title("Side View (Drop)")
    ax1.set_xlabel("Range (m)")
    ax1.set_ylabel("Z (m)")
    ax1.grid(True)
    
    # Top View
    ax2.plot(xs, ys)
    ax2.scatter(target.position[0], target.position[1], c='red', marker='x', s=100)
    ax2.set_title("Top View (Drift)")
    ax2.set_xlabel("Range (m)")
    ax2.set_ylabel("Y (m)")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. Define the environment (Wind 0 m/s)
    environment = Environment(altitude_m=0.0,
                              ground=0.0,
                              temperature='standard',
                              wind_m_s=[0, 0, 0])
    
    # 2. Define the target (1000m downrange, 1 meter diameter)
    target = Target(x=1000, y=0, z=0, radius=0.5)

    # 3. Define the bullet (308 Win 168 gr ELD® Match)
    bullet308 = Bullet(mass=168,
                       diameter=7.82,  # 0.308"
                       bc_g7=0.5,  # guessed
                       muzzle_vel=2700,
                       twist_rate=twist_rate_convert(10),
                       environment=environment,
                       initial_pos=[0, 0, -5],
                       initial_orientation=[0, 0, 0])
    
    bullet, result = run_simulation(environment=environment,
                                    bullet=bullet308,
                                    target=target)
    print(result)
    bullet.print_state()