using CSV, DataFrames, Flux, CUDA, Random, ProgressMeter
using Flux: train!

# Reading the CSV file and converting it to a dataframe
println("Reading the dataframe...")
df = DataFrame(CSV.File("dataset.csv"))
lim = size(df)[1]

# Initializing the model
println("Started model Initialization...")
model = Chain(
    Dense(4, 16, leakyrelu),
    BatchNorm(16),
    Dropout(0.1),
    Dense(16, 32, tanh),
    Dropout(0.2),
    Dense(32, 12, leakyrelu),
    BatchNorm(12),
    Dense(12, 3, sigmoid),
    BatchNorm(3),
    Dense(3, 1, sigmoid)
) |> gpu
model = f64(model)

loss(model, x, y) = Flux.binarycrossentropy(model(x), y')

# The following functions - sdf and bdf - are used to separate the data into signal and background
function sdf(df)

    sdf = DataFrame()
    for i in 1:size(df)[1]
        
        class = df[i, 4]
        if class == "s"
            # Initializing a temporary dataframe
            temp = DataFrame([df[i, :]])

            # We must now add the dataframes
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

# The x_train and y_train functions are used to convert the data into a format that can be used for training
function x_train(df, a, sz)
    
    # We will add the distance of each point from the center of the normalized data
    # This is done to make the model more robust
    tempa = hypot((df[a, 1] - 0.5), (df[a, 2] - 0.5), (df[a, 3] - 0.5))

    # Initializing the x_train array
    x_train = [df[a, 1], df[a, 2], df[a, 3], tempa]
    
    b = a + 1

    for i in b:sz
        tempa = hypot((df[i, 1] - 0.5), (df[i, 2] - 0.5), (df[i, 3] - 0.5))

        # Adding the ith point to the x_train array
        x_train = hcat(x_train, [df[i, 1], df[i, 2], df[i, 3], tempa])
    end

    return x_train
end

# The y_train funcntion is much simpler - reding and parsing the classes into an array
function y_train(df, a, sz) 

    y = df[a:sz, 4]
    y = replace.(y, "b" => "0")
    y = replace.(y, "s" => "1")
    y = parse.(Float64, y)

    return y
end

# This function is used to give a random batch of data
function giverand(vec, batchsize, sorb)

    random_indices = rand(1:size(vec, 2), batchsize) |> gpu
    random_batch = vec[:, random_indices] |> gpu

    # Rather than reading the class data from the dataframe, we can directly fill the answer array
    # This is because the input matrix only contains a single class of points
    ans = CuArray{Float64}(undef, 0)
    ans = fill(sorb == 0 ? 0f0 : 1f0, batchsize) |> gpu
    random_batch = vcat(random_batch, (ans'))

    return random_batch
end

# We'll use a training function to train the model so that we can call it multiple times
function training(model, x_s, x_b, opt)

    @showprogress for epoch in 1:10000
        # Getting a random batch of signal and background data
        rands = giverand(x_s, rand(200:300), 1)
        randb = giverand(x_b, rand(200:300), 0)
        rando = hcat(rands, randb) |> gpu

        # Shuffling the data
        indices = shuffle(1:size(rando, 2)) |> gpu
        rando = rando[:, indices]

        data = [(rando[1:4, :], rando[5, :])]  |> gpu

        train!(loss, model, data, opt) |> gpu
    end

end

# This function is used to check the accuracy of the model
function check(model, x, y)

    ans = model(x) |> gpu
    ans = round.(ans)
    ans = ans .- y'

    # We now have a matrix of the differences between the predictions and the data
    # Counting the number of zeroes in this matrix will give us the number of correct predictions
    co = count(ans .== 0) |> gpu

    return co
end

# This function is used to check the accuracy of the model over the entire dataset
function datacheck(model, df, lim, batchsize)

    i = 1
    accuracy = 0

    while i + batchsize <= lim
        
        # Getting the data for the batch
        xch = x_train(df, i, i + batchsize) |> gpu
        ych = y_train(df, i, i + batchsize) |> gpu

        co = check(model, xch, ych) / batchsize

        accuracy = accuracy + co
        i = i + batchsize
    end
    # Dividing by the number of batches to get the accuracy
    accuracy = accuracy / (lim / batchsize)

    return accuracy
end

# With all the functions defined, we can now train the model!

opt = Flux.setup(RMSProp(0.005), model)
opt2 = Flux.setup(Adam(1e-4), model)

println("Started data processing...")
df_s = sdf(df) |> gpu
df_b = bdf(df) |> gpu

x_s = x_train(df_s, 1, (size(df_s)[1])) |> gpu
x_b = x_train(df_b, 1, (size(df_b)[1])) |> gpu

# Executing the data processing will take a while, have patience

println("Initial accuracy is ", string(Float16(datacheck(model, df, lim, 1000) * 100)), "%")

println("Started training...")
training(model, x_s, x_b, opt)
println("Now training with Adam...")
training(model, x_s, x_b, opt2)

println("Trained accuracy is ", string(Float16(datacheck(model, df, lim, 1000) * 100)), "%")