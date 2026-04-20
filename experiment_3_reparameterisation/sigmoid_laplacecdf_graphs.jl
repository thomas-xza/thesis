using Plots
using Distributions

##  Generated via LLM (but makes heavy use of Julia libraries regardless).

##  1. Define the functions

##  Parameterised Sigmoid

sigmoid_param(x, k) = 1 / (1 + exp(-k * x))

sigmoid(x) = 1 / (1 + exp(-x))

##  Laplace CDF with mu=0, b=1

laplace_dist = Laplace(0.0, 0.018)

laplace_dist_2 = Laplace(0.0, 1)

##  2. Create the x range

x = -6:0.01:6

##  3. Calculate values

# y_sigmoid_reparam = sigmoid_param.(x, 75)  ##  Broadcast over the array
y_sigmoid = sigmoid.(x)  ##  Broadcast over the array

y_laplace_2 = cdf.(laplace_dist_2, x)  ##  Broadcast the CDF function
# y_laplace = cdf.(laplace_dist, x)  ##  Broadcast the CDF function

##  4. Generate the Plot

p = plot(x, y_sigmoid, 
     label="Sigmoid", 
     linewidth=2, 
     legend=:bottomright)

# p = plot(x, y_sigmoid_reparam, 
#      label="Parameterised Sigmoid (75)", 
#      linewidth=2, 
#      legend=:bottomright)

plot!(x, y_laplace_2, 
      label="Laplace CDF (0, 1)", 
      linewidth=2)

# plot!(x, y_laplace, 
#       label="Laplace CDF (0, 0.018)", 
#       linewidth=2)


xlabel!("x")
ylabel!("Probability")

savefig("sigmoid_laplace_cdf.pdf")
