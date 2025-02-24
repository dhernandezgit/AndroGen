import os
import sys
sys.path.append(os.path.abspath('.'))
from abc import abstractmethod
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from src.backend.common.Pose import Pose
from src.backend.common.Color import Color
from src.backend.common.Distributions import NormalDistribution, UniformDistribution
from src.backend.Motion import Motion, StraightMotion


class SpermatozoonComponent:
    """
    Base class for spermatozoon components.
    """
    def __init__(self, width: int, height: int, color: Color, n_points=100):
        self.width = width
        self.height = height
        self.color = color
        self.n_points = n_points
        
    @abstractmethod
    def _calculate_segment(self, n_points=100):
        pass

# Head, Neck, and Tail
class Head(SpermatozoonComponent):
    def __init__(self, width: int, height: int, color: Color, scale_highlight: float, offset_highlight: int, color_highlight: Color, n_points=100):
        super().__init__(width, height, color, n_points)
        self.scale_highlight = scale_highlight
        self.offset_highlight = offset_highlight
        self.color_highlight = np.array(color_highlight.get())

    def _calculate_segment(self, alpha=1.0):
        colors = np.zeros((self.n_points, 3))
        colors[:,:] = self.color.get()
        alphas = np.linspace(alpha, alpha, num=self.n_points)
        
        half_n_points = int(self.n_points / 2)
        first_circle = self.width * NormalDistribution(0.8, 0.1).random_samples()
        medium_circle = self.width
        last_circle = self.width - first_circle/2
        circle_sizes = [first_circle, medium_circle, last_circle]
        section_size = [circle_sizes.pop(random.randrange(len(circle_sizes))),
                        max(circle_sizes),
                        min(circle_sizes)]
        sizes = np.concatenate((
            np.linspace(section_size[0], section_size[1], num=half_n_points),
            np.linspace(section_size[1], section_size[2], num=self.n_points - half_n_points)
            )).ravel()
        real_length = self.height - sizes[0]/2 - sizes[-1]/2  
        lengths = np.linspace(real_length/(self.n_points-1), real_length/(self.n_points-1), num=self.n_points).ravel()
        return lengths, sizes, colors, alphas

class Neck(SpermatozoonComponent):
    def __init__(self, width: int, height: int, color: Color, n_points=100):
        super().__init__(width, height, color, n_points)

    def _calculate_segment(self, alpha=1.0):
        colors = np.zeros((self.n_points, 3))
        colors[:,:] = self.color.get()
        alphas = np.linspace(alpha, alpha, num=self.n_points)
        sizes = np.linspace(self.width, 0.9*self.width, num=self.n_points)
        real_length = self.height + sizes[0]/2 + sizes[-1]/2 
        lengths = np.linspace(real_length/(self.n_points-1), real_length/(self.n_points-1), num=self.n_points).ravel()
        return lengths, sizes, colors, alphas

class Tail(SpermatozoonComponent):
    def __init__(self, width: int, height: int, color: Color, cycle_amplitude: float, cycle_speed: float, total_angle=4, angle_phase=0, n_points=100):
        super().__init__(width, height, color, n_points)
        self.cycle_amplitude = cycle_amplitude
        self.cycle_speed = cycle_speed
        self.total_angle = total_angle
        self.angle_phase = angle_phase

    def _calculate_segment(self, alpha=1.0):
        colors = np.zeros((self.n_points, 3))
        colors[:,:] = self.color.get()
        alphas = np.linspace(alpha, 0.8*alpha, num=self.n_points)
        sizes = np.linspace(0.9*self.width, 0.4*self.width, num=self.n_points)
        real_length = self.height + sizes[-1]/2
        lengths = np.linspace(real_length/(self.n_points-1), real_length/(self.n_points-1), num=self.n_points).ravel()
        return lengths, sizes, colors, alphas

# Droplet
class Droplet(SpermatozoonComponent):
    def __init__(self, diameter: float, position: float, color: Color, n_points=1):
        super().__init__(diameter, position, color, n_points)
        
    def _calculate_position(self, head: Head, neck: Neck):
        position = min(neck.height, self.height)
        points_position = int(position * neck.height / neck.n_points)
        return head.n_points + points_position 
    
    def _calculate_segment(self, alpha=1.0):
        colors = np.zeros((self.n_points, 3))
        colors[:,:] = self.color.get()
        alphas = alpha
        lengths = 0
        sizes = self.width
        return lengths, sizes, colors, alphas
    
    
class Shadow():
    def __init__(self, starting_color: Color, ending_color: Color, starting_alpha=1.0, ending_alpha=0.5, offset=0, starting_scale=2, ending_scale=4, n_iterations = 4):
        self.starting_color = np.array(starting_color.get())
        self.ending_color = np.array(ending_color.get())
        self.starting_alpha = starting_alpha
        self.ending_alpha = ending_alpha
        self.offset = offset
        self.starting_scale = starting_scale
        self.ending_scale = ending_scale
        self.n_iterations = n_iterations
        self.scales = np.linspace(self.starting_scale, self.ending_scale, self.n_iterations)

        # Interpolate colors and alphas
        colors = np.linspace(self.starting_color, self.ending_color, self.n_iterations)
        alphas = np.power(np.linspace(self.starting_alpha, self.ending_alpha, self.n_iterations), 2)
        self.rgba_colors = np.concatenate((colors, alphas[:, None]), axis=1)

    def calculate(self, sizes):
        new_sizes = []
        new_colors = []
        for scale in self.scales:
            new_sizes += list(np.clip(np.array(sizes) * scale + self.offset, 0, None))
        for rgba_color in self.rgba_colors:
            new_colors += list(np.repeat(rgba_color.reshape(1,-1), len(sizes), axis=0))
        return np.array(new_sizes), np.array(new_colors)
    
# Spermatozoon and its Morphology
class Spermatozoon:
    def __init__(self, sperm_id: int, pose: Pose, motion: Motion, components: List[SpermatozoonComponent], shadow: Shadow, n_points=100, dt=0.1):
        self.id = sperm_id
        self.x0, self.y0, self.theta0 = pose.get()
        self.theta0 = np.deg2rad(self.theta0 + 180)
        self.v, self.omega = motion.generate_trajectory()
        self.components = components
        self.head = [c for c in self.components if isinstance(c, Head)][0]
        self.neck = [c for c in self.components if isinstance(c, Neck)]
        self.tail = [c for c in self.components if isinstance(c, Tail)]
        self.droplet = [c for c in self.components if isinstance(c, Droplet)]
        self.shadow = shadow
        self.n_points = n_points
        self.dt = dt
        self._calculate_component_points(self.n_points)
        self.xt, self.yt, self.thetat = self._compute_trajectory()
    
    def _calculate_component_points(self, n_points):
        head_length = self.head.height
        neck_lengths = [neck.height for neck in self.neck]
        tail_lengths = [tail.height for tail in self.tail]
        neck_max_length = max(neck_lengths) if len(neck_lengths) > 0 else 0
        tail_max_length = max(tail_lengths) if len(tail_lengths) > 0 else 0
        length = head_length + neck_max_length + tail_max_length
        neck_points = 0
        
        self.head.n_points = round(head_length/length*n_points)
        for i, n in enumerate(self.neck):
            self.neck[i].n_points = round(n.height/length*n_points)
            neck_points = self.neck[i].n_points
        for i, t in enumerate(self.tail):
            self.tail[i].n_points = n_points - self.head.n_points - neck_points
        for i, d in enumerate(self.droplet):
            self.droplet[i].n_points = 1
            
    def _compute_trajectory(self):
        """
        Precomputes the trajectory and returns functions to get positions.
        """
        # Precompute cumulative headings
        thetat = self.theta0 + np.cumsum(self.omega * self.dt)
        # Precompute displacements
        dx = - self.v * np.cos(thetat) * self.dt
        dy = - self.v * np.sin(thetat) * self.dt
        dx[0] = 0
        dy[0] = 0
        
        # Precompute positions
        xt = self.x0 + np.cumsum(dx)
        yt = self.y0 + np.cumsum(dy)
        return xt, yt, thetat

    def calculate_movement(self, t=0):
        xs = np.zeros(self.n_points)
        ys = np.zeros(self.n_points)
        sizes = np.zeros(self.n_points)
        colors = np.zeros((self.n_points, 3))
        alphas = np.zeros(self.n_points)
        rgba_colors = np.concatenate((colors, alphas[:, None]), axis=1)

        # Precompute cumulative points for each component
        component_points = [self.head.n_points] + \
                        [max([neck.n_points for neck in self.neck]) if len(self.neck) else 0] + \
                        [max([tail.n_points for tail in self.tail]) if len(self.tail) else 0]
        component_points_cum = np.cumsum([0] + component_points)
    
        cycle_amplitude = self.tail[0].cycle_amplitude if self.tail else 0
        cycle_speed = self.tail[0].cycle_speed if self.tail else 0

        total_angle = self.tail[0].total_angle if self.tail else 0
        angle_phase = self.tail[0].angle_phase if self.tail else 0
        v_scale = 2*self.v[t]/50 if self.v[t]>0 else 2*UniformDistribution(50,60).random_samples()/50
        omega_scale = -0.2*self.omega[t]
        
        # Head calculations
        lengths_head, sizes_head, colors_head, alphas_head = self.head._calculate_segment()
        head_indices = np.arange(component_points_cum[0], component_points_cum[1])
        head_t = t*self.dt+self.head.n_points/self.n_points * self.dt
        
        angles_head = (np.linspace(+0.0*cycle_amplitude, +1.0*cycle_amplitude, num=self.head.n_points) *
                       np.sin(cycle_speed * np.linspace(t, head_t, num=self.head.n_points) + angle_phase)).ravel()
        theta_head = self.thetat[t] + np.cumsum(angles_head)
        xs[head_indices] = self.xt[t] + np.cumsum(lengths_head * np.cos(theta_head)) + sizes_head[0]/2 * np.cos(theta_head[0])
        ys[head_indices] = self.yt[t] + np.cumsum(lengths_head * np.sin(theta_head)) + sizes_head[0]/2 * np.sin(theta_head[0])
        sizes[head_indices] = sizes_head.ravel()
        colors[head_indices] = colors_head
        alphas[head_indices] = alphas_head
        
        # Neck calculations
        if self.neck:
            lengths_neck, sizes_neck, colors_neck, alphas_neck = self.neck[0]._calculate_segment()
            neck_indices = np.arange(component_points_cum[1], component_points_cum[2])
            neck_t = head_t+self.neck[0].n_points/self.n_points * self.dt
            angles_neck = (np.linspace(+1.0*cycle_amplitude, +1.0*cycle_amplitude, num=self.neck[0].n_points) *
                           np.sin(cycle_speed * np.linspace(head_t, neck_t, num=self.neck[0].n_points) + angle_phase) *
                           np.linspace(1*v_scale, 0.33*v_scale, num=self.neck[0].n_points)
                           ).ravel()
            theta_neck = theta_head[-1] + np.cumsum(angles_neck) + omega_scale
            xs[neck_indices] = xs[head_indices[-1]] + np.cumsum(lengths_neck * np.cos(theta_neck))
            ys[neck_indices] = ys[head_indices[-1]] + np.cumsum(lengths_neck * np.sin(theta_neck))
            sizes[neck_indices] = sizes_neck
            colors[neck_indices] = colors_neck
            alphas[neck_indices] = alphas_neck

        # Tail calculations
        if self.tail:
            lengths_tail, sizes_tail, colors_tail, alphas_tail = self.tail[0]._calculate_segment()
            tail_indices = np.arange(component_points_cum[2], component_points_cum[3])
            tail_t = neck_t+self.tail[0].n_points/self.n_points * self.dt
            angles_tail = (np.linspace(+1.0*cycle_amplitude, +1.0*cycle_amplitude, num=self.tail[0].n_points) *
                           np.sin(cycle_speed * np.linspace(neck_t, tail_t, num=self.tail[0].n_points) + angle_phase) *
                           np.linspace(1.0*v_scale, 0.33*v_scale, num=self.tail[0].n_points)
                           ).ravel()
            theta_tail = theta_neck[-1] + np.cumsum(angles_tail)
            xs[tail_indices] = xs[neck_indices[-1]] + np.cumsum(lengths_tail * np.cos(theta_tail))
            ys[tail_indices] = ys[neck_indices[-1]] + np.cumsum(lengths_tail * np.sin(theta_tail))
            sizes[tail_indices] = sizes_tail
            colors[tail_indices] = colors_tail
            alphas[tail_indices] = alphas_tail
            
        if self.droplet:
            index = self.droplet[0]._calculate_position(self.head, self.neck[0])
            lengths_droplet, sizes_droplet, colors_droplet, alphas_droplet = self.droplet[0]._calculate_segment()
            sizes[index] = sizes_droplet
            colors[index] = colors_droplet
            alphas[index] = alphas_droplet
        # Combine colors and alphas into RGBA
        rgba_colors = np.concatenate((colors, alphas[:, None]), axis=1)
        return np.array(xs[::-1]), np.array(ys[::-1]), np.array(sizes[::-1]), np.array(rgba_colors[::-1])
    
    
    def add_shadows(self, xs, ys, sizes, rgba_colors):
        sizes_shadow, rgba_colors_shadow = self.shadow.calculate(sizes)
                
        new_xs = list(xs) * (self.shadow.n_iterations + 1)
        new_ys = list(ys) * (self.shadow.n_iterations + 1)
        new_sizes = np.concatenate((sizes, sizes_shadow), axis=0)
        new_rgba_colors = np.concatenate((rgba_colors, rgba_colors_shadow), axis=0)
        
        return np.array(new_xs[::-1]), np.array(new_ys[::-1]), np.array(new_sizes[::-1]), np.array(new_rgba_colors[::-1])
    
    def add_highlight(self, xs, ys, sizes, rgba_colors):
        highlight_n_points = int(self.head.scale_highlight * self.head.n_points)
        highlight_offset = min(int(self.head.offset_highlight * self.head.n_points), self.head.n_points - highlight_n_points)
        xs_highlight = xs[-highlight_n_points-highlight_offset:-highlight_offset if highlight_offset > 0 else None]
        ys_highlight = ys[-highlight_n_points-highlight_offset:-highlight_offset if highlight_offset > 0 else None]
        colors_highlight = np.repeat(self.head.color_highlight.reshape(1,-1), highlight_n_points, axis=0)
        rgba_colors_highlight = np.concatenate((colors_highlight, np.linspace(1.0, 1.0, num=highlight_n_points)[:, None]), axis=1)
        sizes_highlight = sizes[np.linspace(-self.head.n_points, -1, num=highlight_n_points, dtype=int)] * self.head.scale_highlight

        new_xs = list(xs_highlight)
        new_ys = list(ys_highlight)
        new_sizes = sizes_highlight
        new_rgba_colors = rgba_colors_highlight

        return np.array(new_xs[::-1]), np.array(new_ys[::-1]), np.array(new_sizes[::-1]), np.array(new_rgba_colors[::-1])
    
def pixel_diameter_to_scatter_size(diameter_in_pixels, dpi=300):
    """
    Converts a diameter in pixels to the size required by ax.scatter in Matplotlib.

    Parameters:
        diameter_in_pixels (float): The diameter of the marker in pixels.
        dpi (float): The dots per inch (DPI) of the Matplotlib figure.

    Returns:
        float: The marker size for ax.scatter.
    """
    # Convert diameter in pixels to points (1 inch = 72 points)
    diameter_in_points = diameter_in_pixels * 144.0 / dpi
    # Scatter size is proportional to the area (diameter^2)
    sizes = (diameter_in_points / 2) ** 2
    return sizes

def render(ax, xs, ys, sizes, rgba_colors):
    fixed_sizes = pixel_diameter_to_scatter_size(sizes)
    # Create a scatter plot for the circles with RGBA colors
    scatter = ax.scatter(xs, ys, s=fixed_sizes, marker='o', c=rgba_colors, edgecolors='none', linewidth=0)

    
if __name__ == "__main__":
    # Example Pose, Motion, and Components for Visualization
    pose = Pose(x=640, y=512, angle=0)
    motion = Motion([StraightMotion()]*1000)

    # Create Components
    head_color = Color(r=0, g=0, b=0)  # Red head
    neck_color = Color(r=0, g=0, b=0)  # Green neck
    tail_color = Color(r=0, g=0, b=0)  # Blue tail
    droplet_color = Color(r=0, g=0, b=0)  # Yellow droplet


    head = Head(width=8, height=20, color=head_color)
    neck = Neck(width=2, height=20, color=neck_color)
    tail = Tail(width=2, height=60, angular_speed=10, amplitude=1.5, color=tail_color, angle_phase=0)
    droplet = Droplet(diameter=6, position=20, color=droplet_color)

    components = [head, neck, tail, droplet]

    shadow_color_start = Color(r=1, g=1, b=1)  # Black shadow
    shadow_color_end = Color(r=0.9, g=0.9, b=0.9)  # Black shadow
    shadow = Shadow(starting_color=shadow_color_start, ending_color=shadow_color_end)

    # Create a Spermatozoon instance
    s = Spermatozoon(
        sperm_id=1,
        pose=pose,
        motion=motion,
        components=components,
        shadow=shadow,
        n_points=100
    )

    # Define desired resolution in pixels
    width, height = 1280, 1024

    # Calculate figure size in inches based on desired resolution and dpi
    dpi = 300  # You can adjust this DPI value as needed
    fig_width = width / dpi
    fig_height = height / dpi

    # Create a figure with the calculated size
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # Remove padding by placing axis to fill the figure

    # Generate and render the spermatozoon
    xs, ys, sizes, rgba_colors = s.calculate_movement(t=0)
    xs, ys, sizes, rgba_colors = s.add_shadows(xs, ys, sizes, rgba_colors)
    render(ax, xs, ys, sizes, rgba_colors)

    # Remove borders, axes, and ticks
    ax.axis('off')

    # Set axis limits to match the image size in pixels
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    # Invert the y-axis to match the pixel coordinate system (origin at top-left)
    #ax.invert_yaxis()
    ax.grid(visible=True, linestyle='-', color='gray', alpha=1)

    # Save the figure without borders
    fig.savefig("output_image.png", dpi=dpi, pad_inches=0)
    plt.close(fig)  # Close the figure to free resources