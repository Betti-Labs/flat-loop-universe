
#!/usr/bin/env python3
"""
FIRMAMENT v0.7 - A Symbolic Recursive Substrate

This simulation demonstrates key concepts of the Flat Loop Universe theory:
- Toroidal space (finite but unbounded)
- Symbolic computation with emergent patterns
- Agent-based perception and memory
- Recursive nested universes

Part of the Flat Loop Universe project by Gregory Betti
https://github.com/bettilabs/flat-loop-universe
"""

import numpy as np
import random
import os
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import animation
from datetime import datetime
import argparse
import os

# Simulation parameters
GRID_SIZE = 32      # Size of the main universe grid
STEPS = 50          # Number of simulation steps
INNER_GRID_SIZE = 8 # Size of each agent's inner universe

SYMBOLS = ['_', 'A', 'B', 'C', 'P', 'T', 'P2', 'T2', 'P3', 'T3', 'M']
symbol_to_int = {s: i for i, s in enumerate(SYMBOLS)}
int_to_symbol = {i: s for s, i in symbol_to_int.items()}
colors = ['black', 'red', 'blue', 'green', 'yellow', 'white', 'orange', 'purple', 'gray', 'cyan', 'lime']
cmap = mcolors.ListedColormap(colors[:len(SYMBOLS)])

agent_symbols = [symbol_to_int['P'], symbol_to_int['P2'], symbol_to_int['P3']]
trail_symbols = [symbol_to_int['T'], symbol_to_int['T2'], symbol_to_int['T3']]
message_symbol = symbol_to_int['M']

base_rules = {
    (symbol_to_int['A'], symbol_to_int['A']): symbol_to_int['B'],
    (symbol_to_int['B'], symbol_to_int['B']): symbol_to_int['A'],
    (symbol_to_int['A'], symbol_to_int['B']): symbol_to_int['C'],
    (symbol_to_int['C'], symbol_to_int['_']): symbol_to_int['_'],
}

def get_neighbors(grid, x, y):
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx = (x + dx) % GRID_SIZE
            ny = (y + dy) % GRID_SIZE
            neighbors.append(grid[nx, ny])
    return neighbors

def update_agent_memory(grid, x, y, memory):
    neighbors = get_neighbors(grid, x, y)
    for n in neighbors:
        memory[n] += 1
    return memory

def move_agent_adaptive(grid, x, y, memory):
    best_score = -1
    best_pos = (x, y)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = (x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE
            if grid[nx, ny] not in agent_symbols + trail_symbols:
                neighbors = get_neighbors(grid, nx, ny)
                score = sum(memory[n] for n in neighbors)
                if score > best_score:
                    best_score = score
                    best_pos = (nx, ny)
    return best_pos

def apply_rules(center, neighbors):
    counter = Counter(neighbors)
    if center == symbol_to_int['A'] and counter[symbol_to_int['A']] >= 2:
        return symbol_to_int['B']
    elif center == symbol_to_int['B'] and counter[symbol_to_int['B']] >= 2:
        return symbol_to_int['A']
    elif center == symbol_to_int['A'] and counter[symbol_to_int['B']] >= 1:
        return symbol_to_int['C']
    elif center == symbol_to_int['C'] and counter[symbol_to_int['C']] < 2:
        return symbol_to_int['_']
    else:
        return center

def apply_agent_rules(center, neighbors, rules):
    counter = Counter(neighbors)
    for (src1, src2), result in rules.items():
        if counter[src1] >= 1 and counter[src2] >= 1:
            return result
    return center

def mutate_rules(rules):
    if random.random() < 0.05:
        keys = list(rules.keys())
        if keys:
            i = random.choice(keys)
            rules[i] = random.choice(list(symbol_to_int.values()))
    return rules

def init_inner_grid():
    return np.random.choice(
        [symbol_to_int['A'], symbol_to_int['_']],
        size=(INNER_GRID_SIZE, INNER_GRID_SIZE),
        p=[0.2, 0.8]
    )

def step_inner_grid(inner_grid, rules):
    new_inner = np.copy(inner_grid)
    for x in range(INNER_GRID_SIZE):
        for y in range(INNER_GRID_SIZE):
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx = (x + dx) % INNER_GRID_SIZE
                    ny = (y + dy) % INNER_GRID_SIZE
                    neighbors.append(inner_grid[nx, ny])
            counter = Counter(neighbors)
            for (src1, src2), result in rules.items():
                if counter[src1] >= 1 and counter[src2] >= 1:
                    new_inner[x, y] = result
                    break
    return new_inner

agents = []
grid = np.random.choice([symbol_to_int['A'], symbol_to_int['_']], size=(GRID_SIZE, GRID_SIZE), p=[0.2, 0.8])
for i in range(3):
    ax, ay = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    agents.append({
        'x': ax,
        'y': ay,
        'memory': np.zeros((len(SYMBOLS),), dtype=int),
        'msg': False,
        'rules': dict(base_rules),
        'inner_grid': init_inner_grid(),
        'id': i
    })
    grid[ax, ay] = agent_symbols[i]

def run_simulation(steps=STEPS, grid_size=GRID_SIZE, inner_size=INNER_GRID_SIZE, num_agents=3, save_animation=False):
    """
    Run the FIRMAMENT simulation
    
    Args:
        steps: Number of simulation steps
        grid_size: Size of the universe grid
        inner_size: Size of each agent's inner universe
        num_agents: Number of agents in the simulation
        save_animation: Whether to save the animation to a file
        
    Returns:
        frames: List of grid states for each step
        agents: Final state of all agents
    """
    global GRID_SIZE, INNER_GRID_SIZE
    GRID_SIZE = grid_size
    INNER_GRID_SIZE = inner_size
    
    # Initialize grid and agents
    agents = []
    grid = np.random.choice([symbol_to_int['A'], symbol_to_int['_']], size=(GRID_SIZE, GRID_SIZE), p=[0.2, 0.8])
    for i in range(num_agents):
        ax, ay = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
        agents.append({
            'x': ax,
            'y': ay,
            'memory': np.zeros((len(SYMBOLS),), dtype=int),
            'msg': False,
            'rules': dict(base_rules),
            'inner_grid': init_inner_grid(),
            'id': i % 3  # Ensure we don't exceed available agent symbols
        })
        grid[ax, ay] = agent_symbols[agents[-1]['id']]
    
    frames = []
    for step in range(steps):
        new_grid = np.copy(grid)
        rule_grid = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        for agent in agents:
            rule_grid[agent['x']][agent['y']] = agent['rules']

        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if grid[x, y] not in agent_symbols + trail_symbols + [message_symbol]:
                    neighbors = get_neighbors(grid, x, y)
                    rules = rule_grid[x][y]
                    if rules:
                        new_grid[x, y] = apply_agent_rules(grid[x, y], neighbors, rules)
                    else:
                        new_grid[x, y] = apply_rules(grid[x, y], neighbors)

        for agent in agents:
            x, y, mem, aid, msg, rules = agent['x'], agent['y'], agent['memory'], agent['id'], agent['msg'], agent['rules']
            mem = update_agent_memory(new_grid, x, y, mem)
            new_grid[x, y] = trail_symbols[aid]

            if message_symbol in get_neighbors(new_grid, x, y):
                mem[symbol_to_int['C']] += 2
            if random.random() < 0.1:
                new_grid[(x + 1) % GRID_SIZE, y] = message_symbol
            rules = mutate_rules(rules)
            agent['rules'] = rules

            agent['inner_grid'] = step_inner_grid(agent['inner_grid'], rules)
            if np.sum(agent['inner_grid'] == symbol_to_int['B']) > (INNER_GRID_SIZE ** 2) // 4:
                mem[symbol_to_int['B']] += 1

            new_x, new_y = move_agent_adaptive(new_grid, x, y, mem)
            agent.update({'x': new_x, 'y': new_y, 'memory': mem})
            new_grid[new_x, new_y] = agent_symbols[aid]

        frames.append(np.copy(new_grid))
        grid = new_grid
        
        if step % 10 == 0:
            print(f"Simulation step {step}/{steps}")
    
    return frames, agents

def visualize_simulation(frames, save_path=None):
    """
    Visualize the simulation frames as an animation
    
    Args:
        frames: List of grid states for each step
        save_path: Path to save the animation (if None, display only)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(frames[0], cmap=cmap, vmin=0, vmax=len(SYMBOLS)-1)
    ax.set_title('FIRMAMENT: Flat Loop Universe Simulation')
    
    def update(frame_num):
        im.set_array(frames[frame_num])
        return [im]
    
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=200, blit=True)
    
    if save_path:
        ani.save(save_path, writer='pillow', fps=5)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()
    
    return ani

def visualize_inner_grid(agent, save_path=None):
    """
    Visualize an agent's inner grid
    
    Args:
        agent: Agent dictionary containing inner_grid
        save_path: Path to save the visualization (if None, display only)
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(agent['inner_grid'], cmap=cmap, vmin=0, vmax=len(SYMBOLS)-1)
    plt.title(f"Agent {agent['id']} Inner Universe")
    plt.colorbar(ticks=range(len(SYMBOLS)), 
                 label="Symbols", 
                 orientation="vertical")
    
    # Add symbol labels to colorbar
    plt.clim(-0.5, len(SYMBOLS)-0.5)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Inner grid visualization saved to {save_path}")
    else:
        plt.show()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='FIRMAMENT: Flat Loop Universe Simulation')
    parser.add_argument('--steps', type=int, default=STEPS, help='Number of simulation steps')
    parser.add_argument('--grid-size', type=int, default=GRID_SIZE, help='Size of universe grid')
    parser.add_argument('--inner-size', type=int, default=INNER_GRID_SIZE, help='Size of agent inner grid')
    parser.add_argument('--agents', type=int, default=3, help='Number of agents')
    parser.add_argument('--save', action='store_true', help='Save visualizations')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save output')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Create output directory if saving results
    if args.save and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print(f"Running FIRMAMENT simulation with {args.steps} steps, {args.grid_size}x{args.grid_size} grid, {args.agents} agents")
    frames, agents = run_simulation(
        steps=args.steps,
        grid_size=args.grid_size,
        inner_size=args.inner_size,
        num_agents=args.agents,
        save_animation=args.save
    )
    
    # Visualize results
    if args.save:
        animation_path = os.path.join(args.output_dir, 'firmament_animation.gif')
        visualize_simulation(frames, save_path=animation_path)
        
        # Save final state
        plt.figure(figsize=(10, 10))
        plt.imshow(frames[-1], cmap=cmap, vmin=0, vmax=len(SYMBOLS)-1)
        plt.title('FIRMAMENT: Final State')
        plt.savefig(os.path.join(args.output_dir, 'final_state.png'))
        
        # Save inner grids for each agent
        for i, agent in enumerate(agents):
            inner_path = os.path.join(args.output_dir, f'agent_{i}_inner_grid.png')
            visualize_inner_grid(agent, save_path=inner_path)
    else:
        visualize_simulation(frames)
        
        # Show inner grids for each agent
        for agent in agents:
            visualize_inner_grid(agent)
    
    print("Simulation complete!")
