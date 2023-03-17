'Usage:
  embedding.R ALGO DATASET [--seed=SEED]

Options:
  -h --help     Show this screen.
  --seed=SEED   Random number seed [default: NULL].
  --version     Show version.
' -> doc

library(docopt)
library(jsonlite)

args <- docopt(doc, version = 'Ordinal Embedding in R')

algo <- args$ALGO
dataset_name <- args$DATASET
dataset_file <- file.path('./datasets', paste(dataset_name, '.json', sep=""))
result_file <- file.path('./results', paste('R_', algo, '_', dataset_name, '_', Sys.time() ,'.json', sep=""))
margin <- 1
seed <- args$SEED 

set.seed(seed)

dataset <- fromJSON(dataset_file)
train_triplets <- dataset$train_triplets
objects <- dataset$n_objects
dims = 2
# dims <- dataset$num_dimensions

# R is 1-indexed
train_triplets <- train_triplets + 1
train_quadruplets <- train_triplets[, c(1, 2, 1, 3)]

print("Start embedding ...")
if (algo == 'SOE') {
    library <- 'loe'
    require('loe')
    timing <- system.time(
        result <- SOE(CM=train_quadruplets, N=objects, p=dims, report=1000, 
                      rnd=nrow(train_quadruplets))  # rnd: use all triplets
    )
    loss <- result$str
    embedding <- result$X
} else if (algo == 'MLDS') {
    library <- 'MLDS'
    require('MLDS')
    df <- data.frame(resp=train_triplets$response,
                     s1=train_triplets$lowest,
                     s2=train_triplets$target, 
                     s3=train_triplets$highest)
    timing <- system.time(
        estimator <- mlds(df)
    )
    loss <- logLik.mlds(estimator)[0]
    embedding <- estimator$pscale
} else {
    stop(paste("Unsupported algo", algo))
}
elapsed <- timing[3]
result <- data.frame(dataset=dataset_name, 
               library=library, 
               algorithm=algo, 
               loss=loss, 
               cpu_time=elapsed,
               embedding=I(list(embedding)))  # the I() protects the list, such that the df contains just 1 entry
print(paste("Save results to", result_file, "..."))
cat(toJSON(result, pretty=TRUE), file=result_file)