using Plots
using Trapz


function integrate_pdf(xs :: NTuple{N, Float64} , ys :: NTuple{N, Float64}) where N
    
        # p = plot(collect(x_tuple), collect(y_floats))

        # savefig(p, replace(file, ".txt" => ".pdf"))

        # println(y_floats)

        # println(zipped_x_y)

        return trapz(collect(xs), collect(ys))    

end


function integrate_pdfs_set(a :: Float64 = 1.57, b :: Float64 = 4.71)

    files = filter(f -> endswith(f, ".txt"), readdir())

    x_interval = 1.57:0.015625:4.71

    integrands = Float64[]

    for file in files

        name_only = replace(file, ".txt" => "")

        raw_content = read(file, String)

        y_vals = split(rstrip(raw_content), "\n")

        y_floats = Base.front(Tuple(parse.(Float64, y_vals)))

        # println(typeof(x_tuple))

        # println(y_vals)

        # println(typeof(y_floats))

        integrands = vcat(integrands, integrate_pdf(Tuple(x_interval), y_floats))

    end

    return integrands

end


println(integrate_pdfs_set())
