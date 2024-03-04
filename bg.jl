using CSV, DataFrames, Flux, Statistics, CUDA
using Flux: train!

# Read the CSV file
df = DataFrame(CSV.File("dataset.csv"))
# Now df is a DataFrame containing the data from the CSV file

lim = size(df)[1]

model = Chain(
    Dense(3, 6, leakyrelu),
    BatchNorm(6),
    Dropout(0.1),
    Dense(6, 9, leakyrelu),
    #BatchNorm(9),
    #Dropout(0.1),
    Dense(9, 3, leakyrelu),
    BatchNorm(3),
    Dropout(0.1),
    Dense(3, 1)
) |> gpu
model = f64(model)

loss(model, x, y) = mean(Flux.Losses.logitbinarycrossentropy.(model(x), y))

function lgf(model, x, y)
    k = abs2.(model(x))
    return Flux.Losses.logitbinarycrossentropy.(k, y)
end

function x_train(df, a, sz)
    x_train = [df[a, 1], df[a, 2], df[a, 3]]

    b = a + 1

    for i in b:sz
        x_train = hcat(x_train, [df[i, 1], df[i, 2], df[i, 3]])
    end

    return x_train
end

function y_train(df, a, sz) 
    y = df[a:sz, 4]
    y = replace.(y, "b" => "0")
    y = replace.(y, "s" => "1")
    y = parse.(Float64, y)

    return y
end

function training(model, df, lim, opt)
    for epoch in 1:5
        
        a = 1
        sz = 1500

        while sz < lim
            y = y_train(df, a, sz)
            x = x_train(df, a, sz)
            data = [(x, y)] |> gpu
            train!(loss, model, data, opt)
            
            a = a + 1500
            sz = sz + 1500
        end

    end
end

opt2 = Flux.setup(Descent(0.5), model)
opt = Flux.setup(ADAM(0.1), model)

x = x_train(df, 1, 1000) |> gpu
y = y_train(df, 1, 1000) |> gpu

println("Initial loss on test data: ", loss(model, x, y))

training(model, df, lim, opt)
training(model, df, lim, opt2)

println("Loss on test data: $(loss(model, x, y))")