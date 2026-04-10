using Plots
using Trapz


function main(a :: Float64 = 1.57,
              b :: Float64 = 4.71
              )

    ys_k_v = plaintext_to_tuples("model_params_batch_256_epochs_2_mcdrop_seed_")

    pxlnxs_k_v = shannon_entropy(ys_k_v)

    xs = Tuple(a:0.015625:b)

    shannon_integrands_k_v = Dict{String, Float64}()

    for (k, px_ln_xs) in pxlnxs_k_v

        # println(px_ln_xs)

        println(typeof(px_ln_xs))

        println(typeof(xs))

        shannon_integrands_k_v[k] = integrate_pdf(xs, px_ln_xs)
        
    end

    return shannon_integrands_k_v

end


function integrate_pdf(xs :: NTuple{N, Float64},
                       ys :: NTuple{N, Float64}
                       ) where N

    # p = plot(collect(x_tuple), collect(y_floats))

    # savefig(p, replace(file, ".txt" => ".pdf"))

    # println(y_floats)

    # println(zipped_x_y)

    return trapz(collect(xs), collect(ys))    

end


function shannon_entropy(ys_k_v :: Dict{String, Tuple{Vararg{Float64}}})
    
    pxlnxs_k_v = Dict{String, Tuple{Vararg{Float64}}}()

    for (k, ys) in ys_k_v

        pxlnxs_k_v[k] = map(y -> y > 0 ? -y * log(y) : 0.0, ys)

    end

    return pxlnxs_k_v

end


function plaintext_to_tuples(batch_name :: String)

    files = filter(f -> startswith(f, batch_name) && endswith(f, ".txt"), readdir())

    # integrands = Float64[]

    # xs_k_v = Dict{String, NTuple{201, Float64}}()

    ys_k_v = Dict{String, Tuple{Vararg{Float64}}}()

    for file in files

        pdf_set_name = replace(file, ".txt" => "")

        raw_content = read(file, String)

        y_vals = split(rstrip(raw_content), "\n")

        ys_k_v[pdf_set_name] = Base.front(Tuple(parse.(Float64, y_vals)))

        # println(typeof(x_tuple))

        # println(y_vals)

        # println(typeof(y_floats))

    end

    return ys_k_v

end


println(main())
