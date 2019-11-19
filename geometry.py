import matplotlib.patches as pat
import numpy as np

class Cube():
    '''Draw cube-like shape.'''
    
    def __init__(self, name, output_dim, position, depth,
                 edgecolor='black', linewidth=1, facecolor='blue'):
        '''Initializer.'''
        self.name = name
        self.output_dim = output_dim
        self.position = position
        self.depth = depth
        self.edgecolor = edgecolor
        self.linewidth = linewidth
        self.facecolor = facecolor
    
    
    def draw(self):
        '''Draw 3D rectangle.'''
        shape = np.array([])
        
        # Draw cube depth
        for i in range(self.depth):
            pos_x = self.position.x + self.depth - (i+1)
            pos_y = self.position.y + self.depth - (i+1)
            
            first_rectangle = pat.Rectangle((pos_x, pos_y),
                                             self.output_dim.x,
                                             self.output_dim.y,
                                             edgecolor=self.edgecolor,
                                             linewidth=self.linewidth,
                                             facecolor=self.facecolor)
            mid_rectangle = pat.Rectangle((pos_x, pos_y),
                                       self.output_dim.x,
                                       self.output_dim.y,
                                       linewidth=self.linewidth,
                                       facecolor=self.facecolor)
            if not i:
                shape = np.append(shape, first_rectangle)
                continue 
            shape = np.append(shape, mid_rectangle)
            
        shape = np.append(shape,
                          pat.Rectangle(self.position.get_tuple(),
                                        self.output_dim.x,
                                        self.output_dim.y,
                                        edgecolor=self.edgecolor,
                                        linewidth=self.linewidth,
                                        facecolor=self.facecolor))
        
        # Upper left connecting line
        shape = np.append(shape,
                          pat.ConnectionPatch(xyA=(self.position.x+0.5,
                                                   self.position.y+self.output_dim.y),
                                              xyB=(self.position.x+self.depth-0.7,
                                                   self.position.y+self.output_dim.y+self.depth-1.5),
                                              coordsA = "data",
                                              linewidth=1,
                                              edgecolor=self.edgecolor,
                                              capstyle='round',
                                              arrowstyle='-'))
        
        # Upper right connecting line
        shape = np.append(shape,
                          pat.ConnectionPatch(xyA=(self.position.x+self.output_dim.x+0.5,
                                                   self.position.y+self.output_dim.y),
                                              xyB=(self.position.x+self.output_dim.x+self.depth-1,
                                                   self.position.y+self.output_dim.y+self.depth-2),
                                              coordsA = "data",
                                              linewidth=1,
                                              edgecolor=self.edgecolor,
                                              capstyle='round',
                                              arrowstyle='-'))
        
        # Lower right connecting line
        shape = np.append(shape,
                          pat.ConnectionPatch(xyA=(self.position.x+self.output_dim.x+0.5,
                                                   self.position.y),
                                              xyB=(self.position.x+self.output_dim.x+self.depth-1,
                                                   self.position.y+self.depth-2),
                                              coordsA = "data",
                                              capstyle='round',
                                              edgecolor=self.edgecolor,
                                              linewidth=1,
                                              arrowstyle='-'))
        return shape
    
    
class Poly():
    '''Draw Funnel on top of layers.'''
    
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
        shapes = []
        
        coordinates = np.array([[self.prev_position.x+self.prev_output_dim.x,
                                self.prev_position.y+self.prev_output_dim.y],
                               [self.curr_position.x,
                                self.curr_position.y+self.curr_output_dim.y],
                               [self.curr_position.x+self.curr_depth,
                                self.curr_position.y+self.curr_depth+self.curr_output_dim.y],
                               [self.prev_position.x+self.prev_output_dim.x+self.prev_depth,
                                self.prev_position.y+self.prev_output_dim.y+self.prev_depth]])
        
        # Draw Polyogong
        shapes = np.append(shapes, pat.Polygon(coordinates, alpha=0.2))
        
        # Draw dotted lines
        shapes = np.append(shapes, pat.ConnectionPatch(xyA=(self.prev_position.x+self.prev_output_dim.x,
                                                            self.prev_position.y+self.prev_output_dim.y),
                                                       xyB=(self.curr_position.x,
                                                            self.curr_position.y+self.curr_output_dim.y),
                                                       coordsA = "data",
                                                       color=self.color,
                                                       linewidth=1,
                                                       linestyle=':',
                                                       arrowstyle='-'))
        shapes = np.append(shapes, pat.ConnectionPatch(xyA=(self.curr_position.x+self.curr_depth,
                                                            self.curr_position.y+self.curr_depth+self.curr_output_dim.y),
                                                       xyB=(self.prev_position.x+self.prev_output_dim.x+self.prev_depth,
                                                            self.prev_position.y+self.prev_output_dim.y+self.prev_depth),
                                                       coordsA = "data",
                                                       color=self.color,
                                                       linewidth=1,
                                                       linestyle=':',
                                                       arrowstyle='-'))
        return shapes
    