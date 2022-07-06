/* execute this function as a script:
    matlab -r "embedding <algo> <dataset> <result file>" */
function embedding(algo, dataset_file, result_file)

display(algo);
display(dataset_file);
display(result_file);
dataset = jsondecode(readlines(dataset_file));
train_triplets = readtable(dataset.train_triplets);
dim = dataset.num_dimensions;

/* matlab is 1-indexed */
train_triplets = train_triplets + 1; 

library = 'STE';
addpath(genpath('lib/vanderMaaten_STE'));
startT = cputime;
switch algo
    case 'CKL'
        mu = 0;  /* function doesn't provide a default */
        embedding = ckl_x(train_triplets, dim, mu);
    case 'CKL-K'
        mu = 0;  /* function doesn't provide a default */
        kernel, embedding = ckl_k(train_triplets, dim, mu);
    case 'GNMDS'
        /* mu=[], such that the default is used */
        embedding = gnmds_x(train_triplets, [], dim);
    case 'GNMDS-K'
        /* mu=[], such that the default is used */
        kernel, embedding = gnmds_k(train_triplets, [], dim);
    case 'STE'
        /* lambda=[], such that the default is used */
        embedding = ste_x(train_triplets, [], dim);
    case 'STE-K'
        /* lambda=[], such that the default is used */
        kernel, embedding = ste_k(train_triplets, [], dim);
    case 'tSTE'
        embedding = tste(train_triplets, dim);
    otherwise
        error('Unknown algorithm %s', algo)
end
endT = cputime;



result = struct("dataset", dataset_file, "cpu_time", endT - startT,
                "library", library, "embedding", embedding, "kernel", []);
if exist("kernel", "var")
    result.kernel = kernel;
end
disp("Write result to %s ...", result_file)
writelines(jsondecode(result, PrettyPrint=true), result_file);


exit; /* quit matlab after the script */
end;