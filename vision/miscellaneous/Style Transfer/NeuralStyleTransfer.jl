using Flux
using Metalhead
using Parameters: @with_kw
using Images

im_mean = reshape([0.485, 0.458, 0.408], (1,1,3)) * 255
im_mean2 = reshape([0.485, 0.458, 0.408], (1,1,3,1)) * 255 |> gpu

function load_image(filename; size_img::Int=-1, scale::Int=-1, display_img::Bool=true)
    img = load(filename)
    global original_size = size(img)
    if size_img != -1
        img = imresize(img, (size_img,size_img))
    elseif scale != -1
        dims = size(img, 1)
        img = imresize(img, (dims, dims))
    end
    display_img && display(img)
    img = Float32.(channelview(img)) * 255
    ndims(img) == 2 && return img
    permutedims(img, [3,2,1]) .- im_mean
end

function save_image(filename, img, expected_size;display_img::Bool = false)
    img = reshape(img, (size(img, 2), size(img, 1), 3))
    img = (img .+ im_mean) / 255
    img = permutedims(img, [3,2,1])
    img .-= minimum(img)
    img ./= maximum(img)
    img = colorview(RGB{eltype(img)}, img)
    img = imresize(img,expected_size)
    display_img && display(img)
    save(filename, img)
end

function save_image_1(filename, img, expected_size;display_img::Bool = false)
    img = reshape(img, (size(img, 2), size(img, 1), 3))
    img = (img .+ im_mean) / 255
    img = permutedims(img, [3,2,1])
    img = colorview(RGB{eltype(img)}, img)
    img = imresize(img,expected_size)
    display_img && display(img)
    save(filename, img)
end

function gram_matrix(x)
    w, h, ch, b = size(x)
    local features = reshape(x, w*h, ch, b)
    features = [features[:,:,i] for i in 1:b]
    cat([features[i]' * features[i] for i in 1:b]..., dims=3) / Float32(2 * w * h * ch)
end

function content_loss(x, original_image)
    w, h, ch, b = size(x)
    content_loss_ = sum((x .- original_image).^2) / (4*w*h*ch)
    return content_loss_
end

function style_loss(x, gram_style)
    style_loss_ = Float32(0.0)
    for i in 1:size(gram_style,1)
        style_loss_ += sum((gram_matrix(x[i]) .- gram_style[i]).^2)
    end
    return style_loss_ / size(gram_style, 1)
end

function tv_loss(img)
    w, h, ch, b = size(img)
    ver_comp = sum((img[2:end, :, :, :] - img[1:end-1, :, :, :]).^2)
    hor_comp = sum((img[:, 2:end, :, :] - img[:, 1:end-1, :, :]).^2)
    losses = (ver_comp + hor_comp) / (4* w * h * ch)
    return losses
end

mutable struct vgg19
    slice1
    slice2
    slice3
    slice4
    slice5
end

Flux.@functor vgg19

function vgg19()
    vgg = VGG19().layers
    slice1 = Chain(vgg[1:4]...)
    slice2 = Chain(vgg[5:8]...)
    slice3 = Chain(vgg[8:12]...)
    slice4 = Chain(vgg[13:16]...)
    slice5 = Chain(vgg[17:20]...)
    vgg19(slice1, slice2, slice3, slice4, slice5)
end

function (layer::vgg19)(x)
    res1 = layer.slice1(x)
    res2 = layer.slice2(res1)
    res3 = layer.slice3(res2)
    res4 = layer.slice4(res3)
    res5 = layer.slice5(res4)
    (res1, res2, res3, res4, res5)
end

@with_kw mutable struct Args
    size_img::Int = 224
    style_weight::Float32 = 0.01
    content_weight::Float32 = 10000.0
    tv_weight::Float32 = 30.0
    epochs::Int = 20
    steps_per_epoch::Int = 100
    content_file::String
    style_file::String
    lr::Float32 = 5.0
end

function train(content_file, style_file)
    args = Args()
    args.content_file = content_file
    args.style_file = style_file
    Content = load_image(args.content_file,size_img=args.size_img)
    Style = load_image(args.style_file, size_img=args.size_img);
    
    content = Float32.(reshape(Content, (size(Content)...,1)))
    style = Float32.(reshape(Style, (size(Style)...,1)))
    vgg = vgg19()
    @info("Style_features...")
    style_features = vgg(style)
    content_features = vgg(content);
    gram_style = gram_matrix.(style_features)

    img_generate = Float32.(copy(content))
    i = 1
    opt = ADAM(args.lr)
    @info("Training...")
    for i in 1:args.epochs
        for j in 1:args.steps_per_epoch
            ps = Flux.params(img_generate)
            gs = Flux.gradient(ps) do
                 out_ = vgg(img_generate)
                 (content_loss(out_[5], content_features[5]))*args.content_weight 
                         + style_loss(out_[1:4], gram_style[1:4]) * args.style_weight + tv_loss(img_generate) * args.tv_weight
            end
            Flux.Optimise.update!(opt, ps, gs)
            print(".")
        end
        print("\nIteration: ", i," Content_loss = ", content_loss(img_generate, content),"\n")
        save_image(string("./sample_",i,"_.jpg"), img_generate,(400, 300), display_img=false)
        i+=1
    end
    return img_generate
end

cd(@__DIR__)
img_generate = train("Content.jpg","Style.jpg")
