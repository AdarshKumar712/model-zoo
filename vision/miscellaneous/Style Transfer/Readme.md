# Neural Style Transfer
This is an implementation of <b>Neural Style Transfer</b> to blend two images using <b>VGG19</b> pre-trained model.

## Usage
In order to use the model, first change to `cd` to directory where `Content` and `Style` images are saved. Also change
the name for `Content.jpg` and `Style.jpg` in case different names given to `Content` and `Style` images.<br>
<br>
Then to train the model, and save the stylized file:
```julia
julia> include("NeuralStyleTransfer.jl")
```
## References
<li> https://github.com/avik-pal/FastStyleTransfer.jl.git
