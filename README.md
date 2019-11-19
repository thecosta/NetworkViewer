# NetworkViewer
Visualization tool for feed-forward networks.

# Directions for use
Example scripts can be found in the `examples` folder. 

Each layer of your network should be created with the `Layer` object.

### Layer Attributes
* `name`: layer name. must include 'dense', 'conv', 'input', or 'output' if layer represents one of such.
* `output_dim`: dimensionality of the output layer. must be Tuple class.
* `depth`: depth or number of channels of the output layer.
* `output_dim_label`: output dimensionality label to use. useful for when reshaping output dimensions. 
`tuple` in this repository are represented by class `Tuple`, with attributes `x` and `y`. 


Your network needs to be translated into a list of `Layer`. feed them into the `Plotter` class.

### Plotter Attributes
* `layers`: list of Layer.
* `window_width`: pixel width of plotting window.
* `window_height`: pixel height of plotting window.
* `dpi`: dots per inch.
* `padding`: pixel padding between layers.
* `show_dim`: show layer dimensions.
* `show_lower_text`: show lower layer text, ie activation functions maxpooling, flatten.
* `title`: plot title.
* `x_scale`: scale all x dimensions.
* `y_scale`: scale all y dimensions.
* `depth_scale`: scale all depth dimensions.
* `log`: scale by log factor all dimensions.

# Examples

### Baseline Model
![Baseline Model](https://github.com/thecosta/NetworkViewer/blob/master/examples/BaselineModel.png)

### VGGNet
![VGGNet](https://github.com/thecosta/NetworkViewer/blob/master/examples/VGGNet.png)

### AlexNet
![AlexNet](https://github.com/thecosta/NetworkViewer/blob/master/examples/AlexNet.png)
