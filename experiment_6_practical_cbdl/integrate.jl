using Plots


files = filter(f -> endswith(f, ".txt"), readdir())


x_interval = 1.57:0.015625:4.71

x_tuple = Tuple(x_interval)



for file in files

    name_only = replace(file, ".txt" => "")

    raw_content = read(file, String)
    
    y_vals = split(rstrip(raw_content), "\n")

    # println(y_vals)

    y_floats = Base.front(Tuple(parse.(Float64, y_vals)))

    println(typeof(x_tuple))

    println(typeof(y_floats))

    # println(y_floats)

    zipped_x_y = Tuple(zip(x_tuple, y_floats))

    # println(zipped_x_y)

    p = plot(collect(x_tuple), collect(y_floats))
        
    savefig(p, replace(file, ".txt" => ".pdf"))
    
end


println(x_tuple)
