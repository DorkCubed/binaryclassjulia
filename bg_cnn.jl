using CSV, DataFrames, Flux, CUDA, Random, ProgressMeter
using Flux: train!

# Read the CSV file
df = DataFrame(CSV.File("dataset.csv"))
lim = size(df)[1]

model = Chain(
    Dense(4, 16, leakyrelu),
    BatchNorm(16),
    Dropout(0.1),
    Dense(16, 32, tanh),
    Dropout(0.2),
    Dense(32, 12, leakyrelu),
    BatchNorm(12),
    Dropout(0.3),
    Dense(12, 3, sigmoid),
    BatchNorm(3),
    Dense(3, 1, sigmoid)
) |> gpu
model = f64(model)

loss(model, x, y) = Flux.binarycrossentropy(model(x), y')

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

function training(model, x_s, x_b, opt)
    @showprogress for epoch in 1:10000
        rands = giverand(x_s, rand(200:300), 1)
        randb = giverand(x_b, rand(200:300), 0)
        rando = hcat(rands, randb) |> gpu

        indices = shuffle(1:size(rando, 2)) |> gpu
        rando = rando[:, indices]

        data = [(rando[1:4, :], rando[5, :])]  |> gpu

        train!(loss, model, data, opt) |> gpu
    end

end

function check(model, x, y)
    ans = model(x) |> gpu
    ans = round.(ans)
    ans = ans .- y'
    co = count(ans .== 0) |> gpu
    return co
end

function datacheck(model, df, lim, batchsize)
    i = 1
    accuracy = 0
    while i + batchsize <= lim
        xch = x_train(df, i, i + batchsize) |> gpu
        ych = y_train(df, i, i + batchsize) |> gpu

        co = check(model, xch, ych) / batchsize
        accuracy = accuracy + co
        accuracy = accuracy / 2

        i = i + batchsize
    end

    return accuracy
end

opt = Flux.setup(RMSProp(0.005), model)
opt2 = Flux.setup(Adam(1e-4), model)

df_s = sdf(df) |> gpu
df_b = bdf(df) |> gpu

x_s = x_train(df_s, 1, (size(df_s)[1])) |> gpu
x_b = x_train(df_b, 1, (size(df_b)[1])) |> gpu

println("Initial accuracy is ", string(Float16(datacheck(model, df, lim, 1000) * 100)), "%")

training(model, x_s, x_b, opt)
training(model, x_s, x_b, opt2)

println("Trained accuracy is ", string(Float16(datacheck(model, df, lim, 1000) * 100)), "%")