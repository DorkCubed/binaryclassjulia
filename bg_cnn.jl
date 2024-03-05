using CSV, DataFrames, Flux, Statistics, CUDA, Random, ProgressMeter
using Flux: train!

# Read the CSV file
df = DataFrame(CSV.File("dataset.csv"))
# Now df is a DataFrame containing the data from the CSV file

model = Chain(
    Dense(4, 16, leakyrelu),
    BatchNorm(16),
    Dropout(0.1),
    Dense(16, 20, tanh),
    Dropout(0.2),
    Dense(20, 12, leakyrelu),
    BatchNorm(12),
    Dropout(0.3),
    Dense(12, 3, sigmoid),
    BatchNorm(3),
    Dense(3, 1)
) |> gpu
model = f64(model)

loss(model, x, y) = mean(lgf(model, x, y))

function lgf(model, x, y)
    k = Flux.Losses.logitbinarycrossentropy.(model(x), y)
    return k
end

function sdf(df)
    sdf = DataFrame()
    for i in 1:size(df)[1]
        class = df[i, 4]
        if class == "s"
            temp = DataFrame([df[i, :]])
            sdf = vcat(sdf, temp)
        end
    end
    return sdf
end

function bdf(df)
    bdf = DataFrame()
    for i in 1:size(df)[1]
        class = df[i, 4]
        if class == "b"
            temp = DataFrame([df[i, :]])
            bdf = vcat(bdf, temp)
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

function giverand(vec, batchsize, sorb)
    random_indices = rand(1:size(vec, 2), batchsize) |> gpu
    random_batch = vec[:, random_indices] |> gpu

    ans = CuArray{Float64}(undef, 0)
    ans = fill(sorb == 0 ? 0f0 : 1f0, batchsize) |> gpu

    random_batch = vcat(random_batch, (ans'))
    return random_batch
end

function training(model, df, opt)
    
    df_s = sdf(df) |> gpu
    df_b = bdf(df) |> gpu

    x_s = x_train(df_s, 1, (size(df_s)[1])) |> gpu
    x_b = x_train(df_b, 1, (size(df_b)[1])) |> gpu


    @showprogress for epoch in 1:10000
        rands = giverand(x_s, rand(200:250), 1)
        randb = giverand(x_b, rand(200:250), 0)
        rando = hcat(rands, randb) |> gpu

        indices = shuffle(1:size(rando, 2)) |> gpu
        rando = rando[:, indices]

        data = [(rando[1:4, :], rando[5, :])]  |> gpu

        train!(loss, model, data, opt) |> gpu
    end

end

opt = Flux.setup(Descent(0.05), model)
opt2 = Flux.setup(Adam(0.001), model)

x = x_train(df, 1, 1000) |> gpu
y = y_train(df, 1, 1000) |> gpu

println("Initial loss on test data: ", loss(model, x, y))

#training(model, df, opt)
training(model, df, opt2)

println("Loss on test data: $(loss(model, x, y))")

#generating(model, df, lim)