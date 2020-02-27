using Flux
using Flux:onehot,crossentropy,onecold
using CSV,MLBase
using Base.Iterators: repeated
using DataFrames,StatsBase
using PyCall,BSON
using Embeddings
using WordTokenizers
using MLDataPattern
using Random
using MLDataUtils

cd("./Downloads/emojify")
train = CSV.read("train_emoji.csv",header =["Text","Classifier","Col3","Col4"])[:,[1,2]];
first(train,6)

X = train[:,1]
Y = train[:,2];

countmap(train[!,2])

X_bal,Y_bal = oversample((X,Y),shuffle = true);

emoji_dictionary = Dict{Int64,String}(0=>"💙",    # :heart: prints a black instead of red heart depending on the font
                    1=> "🎾",
                    2=> "😄",
                    3=> "😞",
                    4=> "🍴")

N = 5
Y_ = zeros(N,length(Y_bal))
for i in 1:length(Y_bal)
    Y_[Y_bal[i]+1,i] = 1
end

const embtable = load_embeddings(GloVe) # or load_embeddings(FastText_Text) or ...

const get_word_index = Dict(word=>ii for (ii,word) in enumerate(embtable.vocab))

function get_embedding(word)
    ind = get_word_index[word]
    emb = embtable.embeddings[:,ind]
    return emb
end

set_tokenizer(poormans_tokenize)

X_ = [tokenize(lowercase(i)) for i in X_bal];

function tokenise(s)
    token_arr = []
    for c in s
        if (c in keys(get_word_index))==1
        push!(token_arr,get_embedding(c))
    else
        #print(c)
        push!(token_arr,get_embedding("unk"))
    
        end
    end
    return token_arr
end

Xs = [tokenise(a) for a in X_];

max_length = 0
for i in 1:length(Xs)
    if max_length<length(Xs[i])
        max_length = length(Xs[i])
    end
end

#Converting array{array{array,1],1},1}-Xs to array{embedded matrix}  
X1 = []
for i in 1:length(Xs)
    m = fill(0.0,(length(Xs[i][1]),max_length))
    for j =1:length(Xs[i])
        for k in 1:length(Xs[i][j])
            m[k,j] = Xs[i][j][k]
        end
    end
    push!(X1,m)
end

Y1 = []
for i in 1:size(Y_)[2]
    push!(Y1,Y_[:,i])
end

for i in 1:length(X)
    X1[i] = Flux.normalise(X1[i],dims = 2)
end

X1,Y1 = shuffleobs((X1,Y1));

X_train = X1[1:176]
Y_train = Y1[1:176]
X_val = X1[161:end]
Y_val = Y1[161:end];

batch = []  #batching of each element for Stochastic Gradient Descent as [(X1[1],Y1[1]),(X1[2],Y1[2].......)]
for i in 1:length(X_train)
    push!(batch,(X_train[i],[Y_train[i]]))
end

#Model
Scanner = Chain(LSTM(length(Xs[1][1]),128),LayerNorm(128),x->relu.(x),
        Dropout(0.5),        
        LSTM(128,128),
        LayerNorm(128),
        Dropout(0.5),
        x->relu.(x),
        Dense(128,N)
        ,softmax)

function loss(x,y)
    y_hat = Scanner.(x)
    l= crossentropy(y_hat[1][:,end],y[1])
    Flux.reset!(Scanner)
    return l
end

loss(batch[1]...)

function val_loss_(x,y)
    y_ = []
    for i in 1:length(x)
        push!(y_,Scanner.(x[i]))
        Flux.reset!(Scanner)
    end
    y_hat = y_
    l=0.0
    for i in 1:length(y_hat)
        l+= crossentropy(y_hat[i][length(x[i])][:,end],y[i])
    end
    return l/length(y_hat)
end

function accuracy(x,y)
    y_hat = []
    for i in 1:length(x)
        push!(y_hat,Scanner.(x[i]))
        Flux.reset!(Scanner)
    end
    sum=0.0
    for i in 1:length(y)
        if (argmax(y_hat[i][length(x[i])][:,end])==argmax(y[i]))
        sum+=1.0
        end
    end
    return sum/length(y)
end

opt = ADAM(0.0001,(0.9,0.999))
epochs = 100

best_acc = 0
@info("Beginning training loop...")

for i in 1:epochs
    global best_acc
    index = Random.randperm(length(batch))
    batch = batch[index]
    Flux.train!(loss, params(Scanner),batch, opt)
    loss_val = val_loss_(X_val,Y_val)
    accuracy_ = accuracy(X_val,Y_val)
    if i%10==0
        print("Epoch[$i]- Loss: $loss_val Accuracy: $accuracy_\n")
    end
    # If this is the best accuracy we've seen so far, save the model out
    if accuracy_ >best_acc
        @info("Epoch[$i]-> New best accuracy: $accuracy_ ! Saving model out to emojifier_norm.bson")
        BSON.@save joinpath(dirname(@__FILE__), "emojifier_norm.bson") Scanner max_length
        best_acc = accuracy_
    end
            
end

#Model evaluation on Training Data
BSON.@load "./emojifier_norm.bson" Scanner
accuracy(X1,Y1)

y_label = []
for i in 1:length(X1)
    push!(y_label,Scanner(X1[i]))
    Flux.reset!(Scanner)
end

y_pred = []
for i in 1:length(y_label)
    push!(y_pred,Int64(argmax(y_label[i][:,10])-1))
    end
y_pred = Int64.(y_pred);


confusmat(5,Y_bal.+1,y_pred.+1)


