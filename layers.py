import matplotlib
import matplotlib.patches as pat
import matplotlib.pyplot as plt
import numpy as np

from core import Color, Layer, Tuple
from geometry import Cube, Poly
from matplotlib.pyplot import figure   
from tqdm import tqdm


class Convolution(Layer):
    '''Convolution layer object.'''
    
    def __init__(self, **kwargs):
        '''Initializer.'''
        super().__init__(**kwargs)
        
    def draw(self):
        '''Draw 3D cube for convolution layer.'''
        return Cube(name=self.name, output_dim=self.output_dim, position=self.position,
                    depth=self.depth, facecolor=(255.0/255, 242.0/255, 204.0/255)).draw()
    
    
class Dense(Layer):
    '''Dense layer object.'''
    
    def __init__(self, **kwargs):
        '''Initializer.'''
        super().__init__(**kwargs)
        
    def draw(self):
        '''Draw the shapes.'''
        return Cube(name=self.name, output_dim=self.output_dim, position=self.position,
                    depth=self.depth, facecolor=(242.0/255, 207.0/255, 207.0/255)).draw()
    
    
    
class Input(Layer):
    '''Input layer object.'''
    
    def __init__(self, **kwargs):
        '''Initializer.'''
        super().__init__(**kwargs)
        
    def draw(self):
        '''Draw the shapes.'''
        return Cube(name=self.name, output_dim=self.output_dim, position=self.position,
                    depth=self.depth, facecolor=(207.0/255, 242.0/255, 212.0/255)).draw()
    
    
    
class Output(Layer):
    '''Output layer object.'''
    
    def __init__(self, **kwargs):
        '''Initializer.'''
        super().__init__(**kwargs)
        
    def draw(self):
        '''Draw the shapes.'''
        return Cube(name=self.name, output_dim=self.output_dim, position=self.position,
                    depth=self.depth, facecolor=(207.0/255, 242.0/255, 212.0/255)).draw()
    
    
class Funnel():
    '''Funnel drawing between layers.'''
    
    def __init__(self, prev_position, prev_depth, prev_output_dim,
                 curr_position, curr_depth, curr_output_dim, color):
        '''Initializer.'''
        self.prev_position = prev_position
        self.prev_depth = prev_depth
        self.prev_output_dim = prev_output_dim
        self.curr_position = curr_position
        self.curr_depth = curr_depth
        self.curr_output_dim = curr_output_dim
        self.color = color
        
    def draw(self):
        '''Draw the funnel.'''
        return Poly(prev_position=self.prev_position, prev_depth=self.prev_depth,
                    prev_output_dim=self.prev_output_dim, curr_position=self.curr_position,
                    curr_depth=self.curr_depth, curr_output_dim=self.curr_output_dim,
                    color=self.color).draw()
        
    

if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(8, 7), dpi=100)

    shapes = np.array([])
    shapes = np.append(shapes, Dense(name='dense1', position=Tuple(0,0), shape=Tuple(50, 500), weights=[1, 2, 3, 4, 5]).draw())
    shapes = np.append(shapes, Dense(name='dense2', position=Tuple(100,0), shape=Tuple(50, 500), weights=None).draw())
    shapes = np.array(shapes).flatten()
    [ax.add_patch(shape) for shape in shapes]
    #ax.autoscale_view()
    ax.set_xlim((-50, 800))
    ax.set_ylim((-50, 700))
    ax.figure.canvas.draw()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #ax.grid('on')
    plt.show()
