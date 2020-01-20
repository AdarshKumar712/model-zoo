#Declaration of Packages used
using Flux
using Flux: onehot, chunk, batchseq, throttle, crossentropy,@epochs
using StatsBase: wsample
using Base.Iterators: partition

cd(@__DIR__)

#Check for the input.txt file and if doesn't exist then downloads the text file 
isfile("input.txt") ||
  download("https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
           "input.txt")

#Text Preprocessing
text = collect(String(read("input.txt")))
alphabet = [unique(text)..., '_']
text = map(ch -> onehot(ch, alphabet), text)
stop = onehot('_', alphabet)

N = length(alphabet) #
seqlen = 20
nbatch = 50
epo = 1

#Partioning the data into batches of chunks for Training
Xs = collect(partition(batchseq(chunk(text, nbatch), stop), seqlen))
Ys = collect(partition(batchseq(chunk(text[2:end], nbatch), stop), seqlen))

#Defining the model
m = Chain(
    #Input of Shape N*M where M is the length of Input batch
  LSTM(N, 128,relu),
  LSTM(128, 256),
  LSTM(256,128,relu),
  Dense(128, N),
  softmax)

m = gpu(m)

function loss(xs, ys)
  l = sum(crossentropy.(m.(gpu.(xs)), gpu.(ys)))
  Flux.reset!(l)
  return l
end

#Compilation parameters
opt = ADAM(0.01)
tx, ty = (Xs[5], Ys[5])
evalcb = () -> @show loss(tx, ty)

#Training the model
@epochs epo Flux.train!(loss, params(m), zip(Xs, Ys), opt,
            cb = throttle(evalcb, 30))

# Sampling

function sample(m, alphabet, len)
  m = cpu(m)
  Flux.reset!(m)
  buf = IOBuffer()
  c = rand(alphabet)
  for i = 1:len
    write(buf, c)
    c = wsample(alphabet, m(onehot(c, alphabet)))
  end
  return String(take!(buf))
end

sample(m, alphabet, 1000) |> println

#Saving the text generated by our model to Sample.txt file
open("Sample.txt", "w") do f
    write(f, sample(m, alphabet, 1000))
end




