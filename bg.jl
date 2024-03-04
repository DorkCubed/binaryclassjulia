using CSV, DataFrames, Flux, Statistics, CUDA
using Flux: train!

# Read the CSV file
df = DataFrame(CSV.File("dataset.csv"))
# Now df is a DataFrame containing the data from the CSV file

lim = size(df)[1]

model = Chain(
    Dense(4, 10, tanh),
    BatchNorm(10),
    Dropout(0.1),
    Dense(10, 15, softplus),
    Dense(15, 3, sigmoid),
    BatchNorm(3),
    Dropout(0.1),
    Dense(3, 1)
) |> gpu
model = f64(model)

loss(model, x, y) = mean(lgf(model, x, y))

function lgf(model, x, y)
    k = Flux.Losses.logitbinarycrossentropy.(model(x), y)
    return ((sqrt.(k)) .* 100)
end

function sdf(df)
    sdf = DataFrame()
    for i in 1:size(df)[1]
        class = df[i, 4]
        if class == "s"
            sdf = vcat(sdf, [df[i, :]])
        end
    end
    return sdf
end

function bdf(df)
    bdf = DataFrame()
    for i in 1:size(df)[1]
        class = df[i, 4]
        if class == "b"
            bdf = vcat(bdf, [df[i, :]])
        end
    end
    return bdf
end

function x_train(df, a, sz)
    tempa = hypot((df[a, 1] - 0.5), (df[a, 2] - 0.5), (df[a, 3] - 0.5))
    x_train = [df[a, 1], df[a, 2], df[a, 3], tempa]

    b = a + 1

    for i in b:sz
        tempa = hypot((df[i, 1] - 0.5), (df[i, 2] - 0.5), (df[i, 3] - 0.5))
        x_train = hcat(x_train, [df[i, 1], df[i, 2], df[i, 3], tempa])
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
    for epoch in 1:2
        
        a = 1
        sz = 50

        while sz < lim
            y = y_train(df, a, sz)
            x = x_train(df, a, sz)
            data = [(x, y)] |> gpu
            train!(loss, model, data, opt)
            
            a = a + 50
            sz = sz + 50
        end

    end
end

opt = Flux.setup(Descent(0.05), model)
opt2 = Flux.setup(Adam(0.001), model)

x = x_train(df, 1, 1000) |> gpu
y = y_train(df, 1, 1000) |> gpu

println("Initial loss on test data: ", loss(model, x, y))

training(model, df, lim, opt)
training(model, df, lim, opt2)

println("Loss on test data: $(loss(model, x, y))")