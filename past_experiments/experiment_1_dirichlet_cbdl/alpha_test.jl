
# using Plots, SpecialFunctions

using SpecialFunctions


function gen_predictive_dist(d ::Int64, m ::Int64, n ::Int64, y ::Int64)

    alpha = zeros(UInt64, n)

    alpha_plus_d = zeros(UInt64, n)

    permutations_n = UInt128((2^m)^n)

    x_range = zeros(UInt64, permutations_n)

    res = zeros(Float32, permutations_n)

    all_perms = Vector{UInt64}[]

    for i = UInt128(0):UInt128(permutations_n - 1)

        alpha = gen_permutation(i, m, n)

        all_perms = [all_perms..., alpha]

        # println(alpha)

        for j = 1:length(alpha)

            alpha_plus_d[j] = alpha[j] + d

        end

        # println(alpha)

        res[i+1] = (alpha[y] + d) / (sum(alpha_plus_d) + d*n)

        x_range[i+1] = i

        # @bp

        # println(i, " ", res[i+1])

    end

    return x_range, res, all_perms

end


function gen_permutation(i ::UInt128, m ::Int64, n ::Int64)

    set = zeros(UInt64, n)

    and_bits = UInt128(2^m - 1)

    a = UInt128(0)

    for j = 0:(n-1)

        a = and_bits & i

        set[j+1] = a >> (j * m)
        
        and_bits = and_bits << m

    end

    return set
    
end


function calc_entropy(all_perms :: Vector{Vector{UInt64}})

    # println(x_range)

    println(all_perms)

    all_entropy = Vector{Float64}[]

    highest_entropy = -1024

    lowest_entropy = 1024

    for (i, perm) in enumerate(all_perms)

        perm = perm .+ 1

        sum_a = sum(perm)

        b_a = 1

        for a in perm

            b_a *= factorial(a - 1)

        end

        b_a = b_a / factorial(big(sum_a - 1))

        summation = 0

        for a in perm

            summation += (a - 1) * digamma(a)

        end
        
        all_entropy = [
            all_entropy...,
            log(b_a) + (sum_a - length(perm)) * digamma(sum_a) - summation
        ]

        if all_entropy[i] > highest_entropy

            highest_entropy = all_entropy[i]

            println("new highest ", highest_entropy, perm)

        end

        if  all_entropy[i] < lowest_entropy

            lowest_entropy = all_entropy[i]

            println("new lowest ", lowest_entropy, perm)
            
        end

    end

end


# println(gen_permutation(UInt128(0x1), 8, 8))

# println(gen_permutation(UInt128(0x1_0000_0000_0000), 8, 16))

#res = gen_predictive_dist(6, 5, 20, 1)

x_range, res, all_perms = gen_predictive_dist(1, 3, 3, 1)

calc_entropy(all_perms)

# @time x_range, res = gen_predictive_dist(1, 3, 3, 1)

# println(length(res))

# println(res)

# p = bar(x_range, res, title="Dirichlet plot")

# savefig(p, "bar.pdf")

