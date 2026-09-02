# ============================================================
# EEG STATISTICAL ANALYSIS V1
# Subject-aware mixed-effects logistic regression
# ============================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(lme4)
  library(lmerTest)
  library(broom.mixed)
})

PROJECT_ROOT <- "C:/Users/Ali/Desktop/BrainPerturbationProject"

INPUT_FILE <- file.path(
  PROJECT_ROOT,
  "features",
  "ml_ready_v2",
  "ml_ready_dataset_v2.csv"
)

OUTPUT_DIR <- file.path(
  PROJECT_ROOT,
  "features",
  "statistical_analysis_r_v1"
)

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat("\n============================================================\n")
cat("EEG STATISTICAL ANALYSIS V1\n")
cat("============================================================\n")

cat("Reading:\n", INPUT_FILE, "\n\n")

df <- read_csv(INPUT_FILE, show_col_types = FALSE)

cat("Rows:", nrow(df), "\n")
cat("Columns:", ncol(df), "\n")
cat("Subjects:", length(unique(df[["subject"]])), "\n\n")

# ------------------------------------------------------------
# FEATURE SET
# ------------------------------------------------------------

features <- c(
  "delta_abs",
  "theta_abs",
  "alpha_abs",
  "beta_abs",
  "gamma_abs",
  "delta_rel",
  "theta_rel",
  "alpha_rel",
  "beta_rel",
  "gamma_rel",
  "theta_alpha_ratio",
  "theta_beta_ratio",
  "alpha_beta_ratio",
  "delta_theta_ratio",
  "theta_alpha_beta_ratio",
  "delta_frontal",
  "theta_frontal",
  "alpha_frontal",
  "beta_frontal",
  "gamma_frontal",
  "delta_central",
  "theta_central",
  "alpha_central",
  "beta_central",
  "gamma_central",
  "delta_parietal",
  "theta_parietal",
  "alpha_parietal",
  "beta_parietal",
  "gamma_parietal",
  "delta_occipital",
  "theta_occipital",
  "alpha_occipital",
  "beta_occipital",
  "gamma_occipital",
  "delta_temporal",
  "theta_temporal",
  "alpha_temporal",
  "beta_temporal",
  "gamma_temporal",
  "theta_alpha_frontal_ratio",
  "alpha_beta_frontal_ratio",
  "theta_alpha_central_ratio",
  "alpha_beta_central_ratio",
  "theta_alpha_parietal_ratio",
  "alpha_beta_parietal_ratio",
  "theta_alpha_occipital_ratio",
  "alpha_beta_occipital_ratio",
  "theta_alpha_temporal_ratio",
  "alpha_beta_temporal_ratio",
  "delta_frontoparietal_diff",
  "delta_frontoparietal_ratio",
  "theta_frontoparietal_diff",
  "theta_frontoparietal_ratio",
  "alpha_frontoparietal_diff",
  "alpha_frontoparietal_ratio",
  "beta_frontoparietal_diff",
  "beta_frontoparietal_ratio",
  "gamma_frontoparietal_diff",
  "gamma_frontoparietal_ratio"
)

features <- features[features %in% names(df)]

cat("Features available:", length(features), "\n\n")

# ------------------------------------------------------------
# BASIC PREPARATION
# ------------------------------------------------------------

df <- df %>%
  mutate(
    subject = factor(subject),
    target_remember = as.numeric(target_remember),
    target_correct = as.numeric(target_correct)
  )

# ------------------------------------------------------------
# ANALYSIS FUNCTION
# ------------------------------------------------------------

run_target_analysis <- function(data, target_name, feature_names) {

  cat("\n============================================================\n")
  cat("TARGET:", toupper(target_name), "\n")
  cat("============================================================\n")

  results <- list()

  for (feature in feature_names) {

    cat("Analyzing:", feature, "\n")

    d <- data %>%
      select(subject, all_of(target_name), all_of(feature)) %>%
      rename(
        y = all_of(target_name),
        x = all_of(feature)
      ) %>%
      filter(
        !is.na(subject),
        !is.na(y),
        !is.na(x),
        is.finite(x),
        is.finite(y)
      )

    if (nrow(d) < 100) {
      next
    }

    if (length(unique(d$y)) < 2) {
      next
    }

    if (sd(d$x) == 0) {
      next
    }

    # Standardize predictor for comparable effect sizes
    d <- d %>%
      mutate(x_z = as.numeric(scale(x)))

    # Mixed-effects logistic regression
    model <- tryCatch(
      glmer(
        y ~ x_z + (1 | subject),
        data = d,
        family = binomial(link = "logit"),
        control = glmerControl(
          optimizer = "bobyqa",
          optCtrl = list(maxfun = 100000)
        )
      ),
      error = function(e) NULL
    )

    if (is.null(model)) {
      results[[length(results) + 1]] <- data.frame(
        target = target_name,
        feature = feature,
        n = nrow(d),
        subjects = length(unique(d$subject)),
        beta = NA_real_,
        se = NA_real_,
        odds_ratio = NA_real_,
        ci_low = NA_real_,
        ci_high = NA_real_,
        p_value = NA_real_,
        singular = NA,
        converged = FALSE
      )
      next
    }

    coefs <- summary(model)$coefficients

    if (!("x_z" %in% rownames(coefs))) {
      next
    }

    beta <- coefs["x_z", "Estimate"]
    se <- coefs["x_z", "Std. Error"]
    p_value <- coefs["x_z", "Pr(>|z|)"]

    odds_ratio <- exp(beta)

    ci_low <- exp(beta - 1.96 * se)
    ci_high <- exp(beta + 1.96 * se)

    results[[length(results) + 1]] <- data.frame(
      target = target_name,
      feature = feature,
      n = nrow(d),
      subjects = length(unique(d$subject)),
      beta = beta,
      se = se,
      odds_ratio = odds_ratio,
      ci_low = ci_low,
      ci_high = ci_high,
      p_value = p_value,
      singular = isSingular(model, tol = 1e-4),
      converged = TRUE
    )
  }

  bind_rows(results)
}

# ------------------------------------------------------------
# RUN BOTH TARGETS
# ------------------------------------------------------------

remember_results <- run_target_analysis(
  df,
  "target_remember",
  features
)

correct_results <- run_target_analysis(
  df,
  "target_correct",
  features
)

results <- bind_rows(
  remember_results,
  correct_results
)

# ------------------------------------------------------------
# FDR CORRECTION
# ------------------------------------------------------------

results <- results %>%
  group_by(target) %>%
  mutate(
    p_fdr = p.adjust(p_value, method = "BH")
  ) %>%
  ungroup()

# ------------------------------------------------------------
# EFFECT INTERPRETATION
# ------------------------------------------------------------

results <- results %>%
  mutate(
    effect_direction = case_when(
      beta > 0 ~ "positive",
      beta < 0 ~ "negative",
      TRUE ~ "zero"
    ),
    effect_magnitude = case_when(
      abs(beta) < 0.2 ~ "very_small",
      abs(beta) < 0.5 ~ "small",
      abs(beta) < 0.8 ~ "moderate",
      abs(beta) < 1.2 ~ "large",
      TRUE ~ "very_large"
    ),
    fdr_significant = !is.na(p_fdr) & p_fdr < 0.05
  )

# ------------------------------------------------------------
# SORT
# ------------------------------------------------------------

results <- results %>%
  arrange(target, p_fdr)

# ------------------------------------------------------------
# SAVE MAIN RESULTS
# ------------------------------------------------------------

write_csv(
  results,
  file.path(
    OUTPUT_DIR,
    "r_mixed_effects_statistical_results_v1.csv"
  )
)

# ------------------------------------------------------------
# TOP RESULTS
# ------------------------------------------------------------

top_results <- results %>%
  filter(!is.na(p_fdr)) %>%
  arrange(p_fdr) %>%
  slice_head(n = 30)

write_csv(
  top_results,
  file.path(
    OUTPUT_DIR,
    "r_mixed_effects_top_results_v1.csv"
  )
)

# ------------------------------------------------------------
# TARGET SUMMARY
# ------------------------------------------------------------

target_summary <- results %>%
  group_by(target) %>%
  summarise(
    features_analyzed = n(),
    fdr_significant = sum(fdr_significant, na.rm = TRUE),
    min_p_fdr = min(p_fdr, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(
  target_summary,
  file.path(
    OUTPUT_DIR,
    "r_mixed_effects_target_summary_v1.csv"
  )
)

# ------------------------------------------------------------
# QC
# ------------------------------------------------------------

qc <- data.frame(
  rows = nrow(df),
  columns = ncol(df),
  subjects = length(unique(df[["subject"]])),
  features_analyzed = length(features),
  result_rows = nrow(results),
  fdr_significant_rows = sum(results$fdr_significant, na.rm = TRUE),
  nan_numeric_cells = sum(
    sapply(results, function(x)
      if (is.numeric(x)) sum(is.nan(x)) else 0
    )
  ),
  inf_numeric_cells = sum(
    sapply(results, function(x)
      if (is.numeric(x)) sum(is.infinite(x)) else 0
    )
  ),
  duplicate_target_feature = sum(
    duplicated(results[c("target", "feature")])
  )
)

write_csv(
  qc,
  file.path(
    OUTPUT_DIR,
    "r_mixed_effects_qc_v1.csv"
  )
)

# ------------------------------------------------------------
# CONSOLE SUMMARY
# ------------------------------------------------------------

cat("\n============================================================\n")
cat("FINAL R STATISTICAL ANALYSIS QC\n")
cat("============================================================\n")

cat("Rows:", nrow(df), "\n")
cat("Subjects:", length(unique(df[["subject"]])), "\n")
cat("Features analyzed:", length(features), "\n")
cat("Result rows:", nrow(results), "\n")
cat(
  "FDR significant:",
  sum(results$fdr_significant, na.rm = TRUE),
  "\n"
)

cat("\n============================================================\n")
cat("TOP SCIENTIFIC RESULTS\n")
cat("============================================================\n")

print(
  results %>%
    filter(fdr_significant) %>%
    select(
      target,
      feature,
      beta,
      odds_ratio,
      ci_low,
      ci_high,
      p_fdr,
      effect_magnitude
    ) %>%
    arrange(p_fdr) %>%
    slice_head(n = 30)
)

cat("\n============================================================\n")
cat("R STATISTICAL ANALYSIS V1 COMPLETE\n")
cat("============================================================\n")

cat("Saved:\n")
cat(
  file.path(
    OUTPUT_DIR,
    "r_mixed_effects_statistical_results_v1.csv"
  ),
  "\n"
)

cat(
  file.path(
    OUTPUT_DIR,
    "r_mixed_effects_top_results_v1.csv"
  ),
  "\n"
)

cat(
  file.path(
    OUTPUT_DIR,
    "r_mixed_effects_target_summary_v1.csv"
  ),
  "\n"
)

cat(
  file.path(
    OUTPUT_DIR,
    "r_mixed_effects_qc_v1.csv"
  ),
  "\n"
)

cat("\nSTATUS: PASS - R MIXED-EFFECTS STATISTICAL ANALYSIS CREATED\n")