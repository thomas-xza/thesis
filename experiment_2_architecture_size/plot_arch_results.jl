using Plots

# Key-value store
data = Dict(
    32 => [0.698556, 0.695858, 0.696118, 0.692866, 0.696146, 0.695796, 0.693443, 0.695491, 0.692187],
    128 => [0.666663, 0.665968, 0.660378, 0.655696, 0.654846, 0.65424, 0.650292, 0.676448,],
    256 => [0.530119, 0.517477, 0.496647, 0.509344, 0.494437, 0.491354, 0.50228, 0.495217, 0.493534],
    512 => [0.378705, 0.328414, 0.290856, missing, missing, missing, missing, missing, missing],
    1024 => [0.511394, 0.135184, missing, missing, missing, missing, missing, missing, missing],
2048 => [0.212542, missing, missing, missing, missing, missing, missing, missing, missing]
)

# Plot
plot()
for (key, value) in data
    plot!(1:length(value), value, label=key, legend=:bottomright, marker=:circle)
end

# Customize plot
xlabel!("Training epoch quantity")
ylabel!("Average loss function output during test run")
# title!("")

# save plot
savefig("line_graph_loss_vs_epochs_different_archs.pdf")
