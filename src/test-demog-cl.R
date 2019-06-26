print("Tests:")
print("(1) age mean differences between clusters (pairwise t-test with Bonferroni correction);")
print("(2) average number of encounters between clusters (pairwise t-test with Bonferroni correction);")
print("(3) sex numerosities via chi-squared test with Bonferroni correction")

level.path <- paste('level-', commandArgs(TRUE)[2])
data.path <- file.path(commandArgs(TRUE)[1], level.path)

cl_df = read.table(file.path(data.path, 
                   paste(commandArgs(TRUE)[3],
                   '-cl-demographics.csv')), 
                   sep=',',
                   header=TRUE,
                   as.is = TRUE)

cl_df$CLUSTER <- as.factor(cl_df$CLUSTER)

print("AGE:")
pairwise.t.test(cl_df$AGE, cl_df$CLUSTER, p.adjust.method='bonferroni')
print("\n")
print("N_ENCOUNTERS")
pairwise.t.test(cl_df$N_ENCOUNTERS, cl_df$CLUSTER, p.adjust.method='bonferroni')

# tab <- table(cl_df$SEX, cl_df$CLUSTER)

pairwise.chisq.test <- function(x, g, p.adjust.method = p.adjust.methods, ...) {
  DNAME <- paste(deparse(substitute(x)), "and", deparse(substitute(g)))
  g <- factor(g)
  p.adjust.method <- match.arg(p.adjust.method)

  compare.levels <- function(i, j) {
    idx <- which(as.integer(g) == i | as.integer(g) == j)
    xij <- x[idx]
    gij <- as.character(g[idx])
    gij <- as.factor(gij)
    print(table(xij, gij))
    chisq.test(xij, gij, ...)$p.value
  }
  PVAL <- pairwise.table(compare.levels, levels(g), p.adjust.method)
  ans <- list(method = "chi-squared test", data.name = DNAME, p.value = PVAL,
              p.adjust.method = p.adjust.method)
  class(ans) <- "pairwise.htest"
  ans
}
print("\n")
print("SEX")
pairwise.chisq.test(cl_df$SEX, cl_df$CLUSTER, p.adjust.method='bonferroni')
