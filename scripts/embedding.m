function [] = embedding(algo, dataset)
    disp(sprintf("Start with %s and %s...", algo, dataset));
    % execute this function as a script:
    %   matlab -batch "embedding <algo> <dataset>"
    dataset_file = sprintf("../datasets/%s.json", dataset); 
    data = jsondecode(fileread(dataset_file));
    train_triplets = data.train_triplets;
    dim = 2;
    % matlab is 1-indexed 
    train_triplets = train_triplets + 1; 
    
    library = 'vanderMaaten';
    addpath(genpath('../lib/vanderMaaten_STE'));
    startT = cputime;
    switch algo
        case 'CKL'
            mu = 0;  % function doesn't provide a default 
            embedding = ckl_x(train_triplets, dim, mu);
        case 'CKL-K'
            mu = 0;  % function doesn't provide a default
            kernel, embedding = ckl_k(train_triplets, dim, mu);
        case 'GNMDS'
            % mu=[], such that the default is used
            embedding = gnmds_x(train_triplets, [], dim);
        case 'GNMDS-K'
            % mu=[], such that the default is use
            kernel, embedding = gnmds_k(train_triplets, [], dim);
        case 'STE'
            % lambda=[], such that the default is used
            embedding = ste_x(train_triplets, [], dim);
        case 'STE-K'
            % lambda=[], such that the default is used
            kernel, embedding = ste_k(train_triplets, [], dim);
        case 'tSTE'
            embedding = tste(train_triplets, dim);
        otherwise
            error('Unknown algorithm %s', algo)
    end
    endT = cputime;
    
    
    
    result = struct("algorithm", algo, "dataset", dataset, "cpu_time", endT - startT, "library", library, "embedding", embedding);
    if exist("kernel", "var")
        result.kernel = kernel;
    end

    result_file = sprintf("../results/matlab_%s_%s_%s.json", algo, dataset, string(datetime())); 
    disp(sprintf("Write result to %s ...", result_file))
    f = fopen(result_file, 'w');
    fprintf(f, jsonencode(result, PrettyPrint=true));
    fclose(f);
end