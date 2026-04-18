
# Jackson & Turner (2017) Confidence Intervals for Small k
# ==========================================================

# Load metafor
library(metafor)

# Small sample data
k <- 5
yi <- c(0.5, 0.3, 0.7, 0.4, 0.6)
vi <- c(0.1, 0.15, 0.08, 0.12, 0.09)

# Standard DL
rma_dl <- rma.uni(yi=yi, vi=vi, method="DL")
print("DL:")
print(confint(rma_dl))

# Hartung-Knapp adjustment
rma_hk <- rma.uni(yi=yi, vi=vi, method="DL", test="knha")
print("Hartung-Knapp:")
print(confint(rma_hk))

# Jackson (2014) CI - if implemented
# This would require custom code or different package

# Jackson et al. (2017) between-study variance CI
# Uses Q-profile method

# Profile likelihood CI for tau2
confint(rma_dl, parm="tau2")

# Compare CI widths
dl_width <- rma_dl$ci.ub - rma_dl$ci.lb
hk_width <- rma_hk$ci.ub - rma_hk$ci.lb

cat("\n=== CI Width Comparison ===\n")
cat("DL width:", dl_width, "\n")
cat("HK width:", hk_width, "\n")
cat("HK/DL ratio:", hk_width/dl_width, "\n")
