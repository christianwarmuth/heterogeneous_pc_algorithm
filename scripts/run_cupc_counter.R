suppressMessages(library(pcalg))
suppressMessages(library(graph))
suppressMessages(library(MASS))
suppressMessages(library(tictoc))
suppressMessages(library(igraph))
suppressMessages(library(optparse))

source("cupc_counter.R")

option_list = list(
    make_option(c("-n", "--nodes"), type = "integer", default = 60,
        help = "number of nodes [p = 60]", metavar = "nodes"),
    make_option(c("-s", "--samples"), type = "integer", default = 1000,
        help = "number of samples [n = 1000]", metavar = "samples"),
    make_option(c("-p", "--probability"), type = "double", default = 0.2,
        help = "probability of connecting a node to another node with higher topological ordering [prob = 0.2]", metavar = "probability"),
    make_option(c("-l", "--lower_bound"), type = "double", default = 0.1,
        help = "lower limit of edge weights, chosen uniformly at random [lower_bound = 0.1]", metavar = "lower_bound"),
    make_option(c("-u", "--upper_bound"), type = "double", default = 1.0,
        help = "upper limit of edge weights, chosen uniformly at random [upper_bound = 1.0]", metavar = "upper_bound"),
    make_option(c("-a", "--alpha"), type = "double", default = 0.05,
        help = "level of significance [alpha = 0.05]", metavar = "alpha"),
    make_option(c("-o", "--output_file"), type = "character", default = "",
        help = "output file for storing the measurements [output_file = '']", metavar = "output_file"))

opt_parser = OptionParser(option_list = option_list)
opt = parse_args(opt_parser)

p <- opt$nodes
n <- opt$samples
prob <- opt$probability
lB <- opt$lower_bound
uB <- opt$upper_bound
alpha <- opt$alpha
o <- opt$output_file

vars <- c(paste0(1:p))
set.seed(43)

gGtrue <- randomDAG(p, prob = prob, lB = 0.1, uB = 1, V = vars)
N1 <- runif(p, 0.5, 1)
Sigma1 <- matrix(0, p, p)
diag(Sigma1) <- N1
eMat <- mvrnorm(n, mu = rep(0, p), Sigma = Sigma1)
gmG <- list(x = rmvDAG(n, gGtrue, errMat = eMat), g = gGtrue)
dataset <- gmG$x

corrolationMatrix <- cor(dataset)
p <- ncol(dataset)
suffStat <- list(C = corrolationMatrix, n = nrow(dataset))

row <- list(p, n, prob, lB, uB, alpha)

write.table(row, file = o, sep = ",", append = TRUE, quote = FALSE, col.names = FALSE, 
    row.names = FALSE, eol = ",")

cuPC_fit <- cu_pc(suffStat, p = p, o = o, alpha = alpha)
