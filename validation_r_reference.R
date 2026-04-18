
# Generate Reference Values for Validation
# ==========================================
# This script generates reference values using R metafor package
# for comparison with Python implementation

library(metafor)

# Tolerance for numerical comparisons
TOLERANCE <- 1e-4

# Function to format result
format_result <- function(result) {
  list(
    effect = as.numeric(result$beta),
    se = result$se,
    ci_lower = result$ci.lb,
    ci_upper = result$ci.ub,
    tau2 = result$tau2,
    i2 = result$I2
  )
}

# =============================================================================
# Test Case 1: BCG Vaccine
# =============================================================================
cat("\n=== BCG Vaccine ===\n")

bcg_yi <- c(-0.59, -0.17, -0.03, -0.47, -0.88, -0.29,
            -0.21, -0.67, -0.31, -0.39, -0.36, -0.25, -0.24)
bcg_vi <- c(0.029, 0.022, 0.009, 0.025, 0.058, 0.018,
            0.049, 0.036, 0.053, 0.068, 0.027, 0.021, 0.053)

# DerSimonian-Laird
rma_dl <- rma.uni(yi=bcg_yi, vi=bcg_vi, method="DL")
cat("DL:", format_result(rma_dl)$effect, "\n")

# REML
rma_reml <- rma.uni(yi=bcg_yi, vi=bcg_vi, method="REML")
cat("REML:", format_result(rma_reml)$effect, "\n")

# Paule-Mandel
rma_pm <- rma.uni(yi=bcg_yi, vi=bcg_vi, method="PM")
cat("PM:", format_result(rma_pm)$effect, "\n")

# Hartung-Knapp
rma_hk <- rma.uni(yi=bcg_yi, vi=bcg_vi, method="DL", test="knha")
cat("HK:", format_result(rma_hk)$effect, "\n")

# =============================================================================
# Test Case 2: Magnesium MI
# =============================================================================
cat("\n=== Magnesium MI ===\n")

mag_yi <- c(-2.02, -0.58, -0.46, -1.55, -1.21, -1.15, -0.82,
            -0.50, -0.19, -0.34, -0.12, -0.04, -0.13, -0.01, 0.03, -0.03)
mag_vi <- c(0.136, 0.124, 0.342, 0.421, 0.335, 0.142, 0.332,
            0.189, 0.102, 0.398, 0.192, 0.206, 0.208, 0.113, 0.089, 0.110)

rma_dl <- rma.uni(yi=mag_yi, vi=mag_vi, method="DL")
cat("DL:", format_result(rma_dl)$effect, "\n")

rma_reml <- rma.uni(yi=mag_yi, vi=mag_vi, method="REML")
cat("REML:", format_result(rma_reml)$effect, "\n")

rma_pm <- rma.uni(yi=mag_yi, vi=mag_vi, method="PM")
cat("PM:", format_result(rma_pm)$effect, "\n")

rma_hk <- rma.uni(yi=mag_yi, vi=mag_vi, method="DL", test="knha")
cat("HK:", format_result(rma_hk)$effect, "\n")

# =============================================================================
# Test Case 3: Very Small (3 studies)
# =============================================================================
cat("\n=== Very Small (3 studies) ===\n")

small_yi <- c(0.7, 0.3, 0.5)
small_vi <- c(0.05, 0.08, 0.06)

rma_dl <- rma.uni(yi=small_yi, vi=small_vi, method="DL")
cat("DL:", format_result(rma_dl)$effect, "\n")

rma_reml <- rma.uni(yi=small_yi, vi=small_vi, method="REML")
cat("REML:", format_result(rma_reml)$effect, "\n")

rma_pm <- rma.uni(yi=small_yi, vi=small_vi, method="PM")
cat("PM:", format_result(rma_pm)$effect, "\n")

rma_hk <- rma.uni(yi=small_yi, vi=small_vi, method="DL", test="knha")
cat("HK:", format_result(rma_hk)$effect, "\n")

cat("\n=== Validation Complete ===\n")
