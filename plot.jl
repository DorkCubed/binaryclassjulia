using CSV
using DataFrames
using Plots

# Read the CSV file into a DataFrame
df = CSV.read("dataset.csv", DataFrame)

function getxyz(df)
    x = []
    y = []
    z = []
    for i in 1:size(df)[1]
        class = df[i, 4]
        if class == "s"
            x = vcat(x, [df[i, 1]])
            y = vcat(y, [df[i, 2]])
            z = vcat(z, [df[i, 3]])
        end
    end
return x, y, z
end

gr() # Choose the GR.jl backend for 3D plotting
x, y, z = getxyz(df)
p = scatter(x, y, z, legend=false)

# Save the plot as an image file
savefig(p, "plot.png")