# Post-hoc analyses:
# - Check confounders;
# - Compare variable scores;
# - Run external validation (TBD).

# LIBRARIES
require(eeptools)
require(reshape2)
require(ggplot2)
require(GGally)
require(plyr)
require(tidyr)

# FUNCTIONS
# Pairwise chi-square test function
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

# DATA PATH AND FILE
DATA_PATH <- '~/Documents/behavioral_phenotyping/data'
FILE_NAME <- 'df_w2vemb_level4.csv'
PLOT_NAME <- 'feat_dist_hc_w2v_level4.pdf'

# RUN ANALYSES
# Read table
df <- read.table(file.path(DATA_PATH, FILE_NAME),
sep = ',',
header = TRUE,
as.is = TRUE)
df <- subset(df, select = c(clpid, sex, bdate, aoa,
n_enc, feat, score_sc, score))
# Add current age column to dataframe
df$cage <- age_calc(as.Date(df$bdate, "%d/%m/%Y"),
units = 'years')
df <- df[order(df$clpid),]

# Inspect confounders
df_conf <- unique(data.frame(pid = apply(t(df$clpid), 2,
function(x) strsplit(x, '-')[[1]][2]),
cluster = as.factor(apply(t(df$clpid), 2,
function(x) strsplit(x, '-')[[1]][1])),
cage = df$cage,
sex = df$sex,
n_enc = df$n_enc), by = 'pid')
# Add to confounder df the behr length for each subject
lenbehr <- ddply(df, .(clpid), nrow)$V1
df_conf$lenbehr <- lenbehr

# Tests:
# (1) age mean differences between clusters (pairwise t-test with Bonferroni correction);
# (2) average number of encounters between clusters (pairwise t-test with Bonferroni correction);
# (3) sex counts via chi-squared test with Bonferroni correction.

print("AGE per cluster (M, SD):")
tapply(df_conf$cage, df_conf$cluster, function(x) c(mean(x), sd(x)))
pairwise.t.test(df_conf$cage, df_conf$cluster, p.adjust.method = 'bonferroni')

print("N_ENCOUNTERS per cluster (M, SD):")
tapply(df_conf$n_enc, df_conf$cluster, function(x) c(mean(x), sd(x)))
pairwise.t.test(df_conf$n_enc, df_conf$cluster, p.adjust.method = 'bonferroni')

print("SEX counts pairwise chi-square between clusters")
tab <- table(df_conf$sex, df_conf$cluster)
tab
pairwise.chisq.test(df_conf$sex, df_conf$cluster,
p.adjust.method = 'bonferroni')

print("AGE OF ASSESSMENT per cluster (M, SD):")
df$cluster <- as.factor(apply(t(df$clpid), 2,
function(x) strsplit(x, '-')[[1]][1]))
tapply(df$aoa, df$cluster, function(x) c(mean(x), sd(x)))
pairwise.t.test(df$aoa, df$cluster, p.adjust.method = 'bonferroni')

print("Length BEHR per cluster (M, SD):")
tapply(df_conf$lenbehr, df_conf$cluster, function(x) c(mean(x), sd(x)))
pairwise.t.test(df_conf$lenbehr, df_conf$cluster, p.adjust.method = 'bonferroni')

# Summary statistics feature raw scores.
# df$feat_cl <- paste(df$cluster,
# df$feat,
# sep = '-')
# print("Summary statistics feature scores foe each cluster.")
# tapply(df$score,
# df$feat_cl,
# function(x) c(mean(x), sd(x)))

##################################################################

# Multiple pairwise comparisons between groups
df_wide <- subset(df, select = c(clpid, feat, score_sc))
df_wide <- dcast(df_wide, clpid ~ feat,
value.var = 'score_sc',
drop = FALSE, fun.aggregate = mean)
df_wide$pid <- apply(t(df_wide$clpid), 2, function(x) strsplit(x, '-')[[1]][2])
df_wide$cluster <- apply(t(df_wide$clpid), 2, function(x) strsplit(x, '-')[[1]][1])
df_wide <- subset(df_wide, select = - clpid)
df_wide$cluster <- as.factor(df_wide$cluster)

print("Percentage of missing data for each cluster")
na_cl <- c()
na_count <- c()
for (cl in levels(df_wide$cluster)){
  tmp <- df_wide[df_wide$cluster==cl, 1:(ncol(df_wide)-2)]
  ttab <- table(is.na(tmp))/(nrow(tmp)*ncol(tmp))
  print(ttab)
  na_cl <- c(na_cl, rep(cl,nrow(tmp)*ncol(tmp)))
  na_count <- c(na_count, rep('notmiss', table(is.na(tmp))[1]),
                rep('miss', table(is.na(tmp))[2]))
}
na_cl <- as.factor(na_cl)
na_count <- as.factor(na_count)
pairwise.chisq.test(na_count, na_cl)
  
# Run pairwise t-test or t-test for score comparisons.
# for (n in names(df_wide)[1 : (ncol(df_wide) - 2)]) {
#     # Drop missing values.
#     df_tmp <- drop_na(subset(df_wide, select = c(n, 'cluster')))
#     check_tab <- table(df_tmp$cluster) > 1
#     cat("\n", "Testing variable", n, "\n\n")
#     if (length(check_tab[check_tab == FALSE]) >= 1) {
#         idxs <- which(df_tmp$cluster == which(check_tab == FALSE) - 1)
#         if (length(idxs) > 0) {
#             df_tmp <- df_tmp[- which(df_tmp$cluster == which(check_tab == FALSE) - 1),]}
#         try(print(t.test(df_tmp[, 1] ~ df_tmp[, 2])))} else {
#           try(print(pairwise.t.test(df_tmp[, 1], df_tmp[, 2],
#                               p.adjust.method = 'bonferroni')))
#         #pt <- pairwise.t.test(df_tmp[, 1], df_tmp[, 2],
#         #                      p.adjust.method = 'bonferroni')
#         #print(pt)}
#         }
# }

# Feature distibution plot
# pdf(file = file.path(DATA_PATH, PLOT_NAME))
ggpairs(subset(df_wide, select = c(grep('ados|psi', names(df_wide)), cluster)), label.pos = 3)
# ggpairs(subset(df_wide, select = c(grep('griffiths', names(df_wide)), cluster)),
#         columnLabels = c("gmds::GQ", "gmds::q_A",
#                          "gmds::q_B", "gmds::q_C",
#                          "gmds::q_D", "gmds::q_E",
#                          "gmds::q_F", "cluster"))
# ggpairs(subset(df_wide, select = c(grep('wechsler', names(df_wide)), cluster)))
# ggpairs(subset(df_wide, select = c(grep('vineland', names(df_wide)), cluster)))
# ggpairs(subset(df_wide, select = c(grep('srs', names(df_wide)), cluster)))
# ggpairs(subset(df_wide, select = c(grep('psi', names(df_wide)), cluster)))
# ggpairs(subset(df_wide, select = c(grep('leiter', names(df_wide)), cluster)))
# dev.off()



