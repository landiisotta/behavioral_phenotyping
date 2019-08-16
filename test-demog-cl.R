require(eeptools)
require(reshape2)

df <- read.table('df_w2vemb_level4.csv', 
                 sep = ',',
                 header = TRUE, 
                 as.is = TRUE)

names(df)

df$pid <- apply(t(df$clpid), 2, function(x) strsplit(x, '-')[[1]][2])
df$cluster <- apply(t(df$clpid), 2, function(x) strsplit(x, '-')[[1]][1])
df <- subset(df, select = -c(X, cllab_aoa, clpid))
df$age <- age_calc(as.Date(df$bdate, "%d/%m/%Y"), 
                   units = 'years')

demog_df <- subset(df, select=c(pid, cluster, sex, age, n_enc))
demog_df <- unique(demog_df, by="pid")

print("Tests:")
print("(1) age mean differences between clusters (pairwise t-test with Bonferroni correction);")
print("(2) average number of encounters between clusters (pairwise t-test with Bonferroni correction);")
print("(3) sex numerosities via chi-squared test with Bonferroni correction")

# level.path <- paste('level-', commandArgs(TRUE)[2])
# data.path <- file.path(commandArgs(TRUE)[1], level.path)
# 
# cl_df = read.table(file.path(data.path, 
#                    paste(commandArgs(TRUE)[3],
#                    '-cl-demographics.csv')), 
#                    sep=',',
#                    header=TRUE,
#                    as.is = TRUE)

demog_df$cluster <- as.factor(demog_df$cluster)
  
print("AGE per cluster (M, SD):")
tapply(demog_df$age, demog_dfdf$cluster, function(x) c(mean(x), sd(x)))
a_age <- aov(demog_df$age~demog_df$cluster)
summary(a_age)
pairwise.t.test(demog_df$age, demog_df$cluster, p.adjust.method='bonferroni')
print("\n")

print("N_ENCOUNTERS:")
tapply(demog_df$n_enc, demog_df$cluster, function(x) c(mean(x), sd(x)))
a_enc <- aov(demog_df$n_enc~demog_df$cluster)
summary(a_enc)
pairwise.t.test(demog_df$n_enc, demog_df$cluster, p.adjust.method='bonferroni')

tab <- table(demog_df$sex, demog_df$cluster)
tab
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
  ans <- list(method = "chi-squared test", 
              data.name = DNAME, 
              p.value = PVAL,
              p.adjust.method = p.adjust.method)
  class(ans) <- "pairwise.htest"
  ans
}
print("\n")
print("SEX")
pairwise.chisq.test(demog_df$sex, demog_df$cluster, 
                    p.adjust.method='bonferroni')

print("AGE OF ASSESSMENT:")
tapply(df$aoa, df$cluster, function(x) c(mean(x), sd(x)))
a_aoa <- aov(df$aoa~df$cluster)
summary(a_aoa)
pairwise.t.test(df$aoa, df$cluster, p.adjust.method='bonferroni')

##################################################################
df$feat_cl <- paste(df$cluster, 
                    df$feat, 
                    sep='-')

tapply(df$score, 
       df$feat_cl, 
       function(x) c(mean(x), sd(x)))

df_rid <- subset(df, select=c(feat, score_sc))
df_rid$cl_pid <- paste(df$pid, df$cluster, sep = '-')
df_wide<-dcast(df_rid, cl_pid ~ feat, 
               value.var = 'score_sc', 
               drop = FALSE, fun.aggregate = mean)
df_wide$pid <- apply(t(df_wide$cl_pid), 2, function(x) strsplit(x, '-')[[1]][1])
df_wide$cluster <- apply(t(df_wide$cl_pid), 2, function(x) strsplit(x, '-')[[1]][2])
df_wide <- subset(df_wide, select = -cl_pid)

df_wide$cluster <- as.factor(df_wide$cluster)

# Create combinations of the variables
combinations <- combn(colnames(df_wide),2, simplify = FALSE)

# Do the t.test
results <- lapply(seq_along(names(df_wide)[:ncol(df_wide)-2]), function (n) {
  df_rid <- df[,]
  result <- t.test(df[,1], df[,2])
  return(result)})

# Rename list for legibility    
names(results) <- paste(matrix(unlist(combinations), ncol = 2, byrow = TRUE)[,1], matrix(unlist(combinations), ncol = 2, byrow = TRUE)[,2], sep = " vs. ")


for (n in names(df_wide)[1:(ncol(df_wide) - 2)]) {
  df_tmp <- drop_na(subset(df_wide, select = c(n, 'cluster')))
  tab <- table(df_tmp$cluster) > 0
  if (length(tab[tab == FALSE]) >= 1){
    df_tmp <- df_tmp[-which(df_tmp$column == as.factor(which(tab == FALSE))), ]
    print(df_tmp)}
  cat("\n", "Testing variable", n, "\n\n")
  pt <- pairwise.t.test(df_tmp[, 1], df_tmp[, 2])
  av <- aov(df_tmp[, 1] ~ df_tmp[, 2])
  print(summary(av))
  print(pt)
}




