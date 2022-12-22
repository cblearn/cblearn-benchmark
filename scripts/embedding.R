library(docopt)
library(rjson)
"Usage: embedding.R [--seed SEED] ALGO DATASET RESULT

-h --help   show this
ALGO    name of the embedding algorithm
DATASET meta file of the triplet dataset
RESULT result file
--seed=SEED random number seed [default: NULL]." -> doc
args <- docopt(doc)
print(args)

algo <- args$ALGO
dataset_file <- args$DATASET
result_file <- args$RESULT
margin <- 1
seed <- args$SEED 

set.seed(seed)

dataset <- fromJSON(file=dataset_file)
train_triplets <- read.csv(dataset$train_triplets)
objects <- dataset$num_objects
dims <- dataset$num_dimensions

# R is 1-indexed
train_triplets <- train_triplets + 1

print("Start embedding ...")
if (algo == 'SOE') {
    library <- 'loe'
    require(library)
    timing <- system.time(
        result <- loe.SOE(CM=train_triplets, N=objects, p=dims, report=0, rnd=nrow(triplets))
    )
    loss <- result$str
    embedding <- result$X
} else if (algo == 'MLDS') {
    library <- 'MLDS'
    require(library)
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

result <- list(dataset=dataset_file, 
               library=library, 
               algorithm=algo, 
               loss=loss, 
               cpu_time=timing$elapsed,
               embedding=embedding)
print(result)

print(paste("Save results to", result_file, "..."))
save(toJSON(result, indent=2), file=result_file)