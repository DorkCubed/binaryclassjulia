using CSV
using DataFrames
using Plots
using Interact

# Read the CSV file into a DataFrame
df = CSV.read("dataset.csv", DataFrame)

function getyz(df, x_range)
    y = []
    z = []
    for i in 1:size(df)[1]
        class = df[i, 4]
        x = df[i, 1]
        if class == "s" && x >= x_range[1] && x <= x_range[2]
            y = vcat(y, [df[i, 2]])
            z = vcat(z, [df[i, 3]])
        end
    end
    return y, z
end

gr() # Choose the GR.jl backend for 3D plotting

@manipulate for x_min in slider(minimum(df[:, 1]):0.1:maximum(df[:, 1]), label="X min"), x_max in slider(minimum(df[:, 1]):0.1:maximum(df[:, 1]), label="X max")
    y, z = getyz(df, [x_min, x_max])
    scatter(y, z, legend=false)
end