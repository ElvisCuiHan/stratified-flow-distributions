#!/usr/bin/env Rscript
# Read frozen evidence only; no refitting, method selection or new observations.
suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})
args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args)) args[1] else "outputs/conference_revision_20261019/figures"
if (dir.exists(out)) stop("Use a new output directory to preserve previous exports.")
dir.create(out, recursive = TRUE)
inputs <- character()
read_evidence <- function(path) {
  inputs <<- unique(c(inputs, path))
  read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
}
plot_family <- "Times"
theme_set(theme_classic(base_size = 9, base_family = plot_family) + theme(
  plot.title = element_text(size = 10, face = "bold", hjust = 0),
  axis.title = element_text(size = 9), axis.text = element_text(size = 8, colour = "black"),
  axis.line = element_line(linewidth = .3), axis.ticks = element_line(linewidth = .3),
  legend.position = "bottom", legend.title = element_blank(),
  legend.text = element_text(size = 8), legend.key.width = grid::unit(12, "pt"),
  strip.background = element_blank(), strip.text = element_text(size = 9, face = "bold", hjust = 0),
  panel.spacing = grid::unit(10, "pt"), plot.margin = margin(5, 7, 5, 5)
))
blue <- "#0072B2"
orange <- "#D55E00"
grey <- "#707070"
summarize <- function(d, groups, value, expected = 25L) {
  parts <- split(d, interaction(d[groups], drop = TRUE, lex.order = TRUE))
  do.call(rbind, lapply(parts, function(z) {
    x <- z[[value]]
    stopifnot(length(x) == expected, all(is.finite(x)))
    se <- sd(x) / sqrt(length(x))
    cbind(z[1, groups, drop = FALSE], n = length(x), mean = mean(x),
          lower = mean(x) - 1.96 * se, upper = mean(x) + 1.96 * se)
  }))
}
plots <- list()
save_plot <- function(p, name, width, height) {
  ggsave(file.path(out, paste0(name, ".pdf")), p, device = cairo_pdf,
         width = width, height = height, units = "in", family = plot_family)
  ggsave(file.path(out, paste0(name, ".png")), p, device = ragg::agg_png,
         width = width, height = height, units = "in", dpi = 300, bg = "white")
  plots[[name]] <<- list(plot = p, width = width, height = height)
}
write_table <- function(d, name) write.csv(d, file.path(out, name), row.names = FALSE)

mechanism <- read_evidence("evidence/sfd_mechanism_v1/contrasts.csv")
specs <- list(c("mass", "upper_mass", "a  Mass change"),
              c("position", "upper_position", "b  Position change"),
              c("width", "upper_width", "c  Width change"),
              c("width", "upper_mass", "d  Mass under width change"))
mechanism <- do.call(rbind, lapply(specs, function(s) {
  z <- subset(mechanism, scenario == s[1] & metric == s[2]); z$panel <- s[3]; z
}))
mechanism$method <- factor(mechanism$method,
  c("SFD fixed kernels", "SFD joint", "SFD oracle kernels"),
  c("Fixed kernels", "Joint fit", "Oracle kernels"))
ms <- summarize(mechanism, c("panel", "method"), "estimated_contrast")
truth <- unique(mechanism[c("panel", "true_contrast")])
p <- ggplot(ms, aes(method, mean)) +
  geom_hline(data = truth, aes(yintercept = true_contrast), linewidth = .35, linetype = 2) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = .12, linewidth = .4) +
  geom_point(aes(colour = method), size = 1.8) +
  scale_colour_manual(values = c(grey, blue, orange)) +
  facet_wrap(~panel, scales = "free_y", ncol = 2) +
  labs(x = NULL, y = "Estimated contrast") + theme(legend.position = "none")
save_plot(p, "mechanism_separation", 7.0, 3.6)
write_table(ms, "mechanism_plot_data.csv")

geometry <- read_evidence("evidence/sfd_geometry_stress_v1/contrasts.csv")
recovery <- read_evidence("evidence/sfd_geometry_stress_v1/recovery.csv")
specs <- list(c("mass", "upper_mass", "a  Mass change"),
              c("position", "upper_position", "b  Position change"),
              c("width", "upper_width", "c  Width change"),
              c("position", "upper_mass", "d  Mass under position change"),
              c("width", "upper_mass", "e  Mass under width change"))
gd <- do.call(rbind, lapply(specs, function(s) {
  z <- subset(geometry, scenario == s[1] & metric == s[2]); z$panel <- s[3]; z
}))
gtruth <- unique(gd[c("panel", "truth")])
gs <- summarize(gd, c("panel", "condition", "method"), "estimate")
ra <- subset(recovery, scenario == "reference")
ra$panel <- "f  Angle RMSE (degrees)"
rs <- summarize(ra, c("panel", "condition", "method"), "angle_rmse_degrees")
gs <- rbind(gs, rs)
gs$method <- factor(gs$method, c("SFD supplied", "SFD rotated", "SFD learned angles"),
                    c("Supplied", "Rotated", "Learned angles"))
gs$condition <- factor(gs$condition, c("nominal", "overlap", "heavy_noise"),
                      c("Gaussian", "Overlap", "Student-t"))
pd <- position_dodge(width = .5)
p <- ggplot(gs, aes(condition, mean, colour = method, shape = method)) +
  geom_hline(data = gtruth, aes(yintercept = truth), inherit.aes = FALSE,
             linewidth = .3, linetype = 2) +
  geom_errorbar(aes(ymin = lower, ymax = upper), position = pd, width = .1, linewidth = .35) +
  geom_point(position = pd, size = 1.7) + scale_colour_manual(values = c(grey, orange, blue)) +
  scale_shape_manual(values = c(16, 17, 15)) +
  facet_wrap(~panel, scales = "free_y", ncol = 3) +
  labs(x = NULL, y = "Estimated contrast (a-e)")
save_plot(p, "geometry_validation", 7.0, 4.0)
write_table(gs, "geometry_plot_data.csv")

pbmc <- read_evidence("evidence/pbmc_inductive_marker_holdout_v2/scores.csv")
marker <- read_evidence("evidence/pbmc_inductive_marker_holdout_v2/marker_prediction.csv")
ds <- summarize(pbmc, "method", "heldout_log_score")
ds$method <- factor(ds$method, rev(c("Gaussian", "GMM-2", "GMM-4", "GMM-8", "KDE", "MST SFD")))
p1 <- ggplot(ds, aes(mean, method)) +
  geom_errorbar(aes(xmin = lower, xmax = upper), orientation = "y", width = .12, linewidth = .4) +
  geom_point(size = 1.8) + labs(title = "a  Density prediction", x = "Held-out log density", y = NULL)
qs <- summarize(marker, "readout", "log_loss_delta_vs_density")
qs$readout <- factor(qs$readout, rev(c("PCs", "PCs + density", "PCs + density + entropy",
   "PCs + density + SFD", "PCs + density + GMM memberships")),
   rev(c("PCs", "+ Density", "+ Entropy", "+ SFD tube odds", "+ GMM memberships")))
p2 <- ggplot(qs, aes(mean, readout)) + geom_vline(xintercept = 0, linewidth = .3, linetype = 2) +
  geom_errorbar(aes(xmin = lower, xmax = upper), orientation = "y", width = .12, linewidth = .4) +
  geom_point(size = 1.8) + labs(title = "b  Held-out marker proxy",
    x = "Log-loss change vs PCs + density", y = NULL)
save_plot(p1 + p2 + plot_layout(widths = c(.9, 1.1)), "pbmc_inductive_validation", 7.0, 2.5)
write_table(ds, "pbmc_density_plot_data.csv")
write_table(qs, "pbmc_marker_plot_data.csv")

if ("--development" %in% args) {
dev_plots <- list(); ablation_plots <- list()
for (task in c("md", "ecg")) {
  old <- read_evidence(paste0("outputs/application_pilots_20261018/", task, "/scores.csv"))
  new <- read_evidence(paste0("outputs/occupancy_development_20261019/", task, "/scores.csv"))
  d <- rbind(data.frame(method = old$method, fold = old$rep, value = old$heldout_log_score),
             data.frame(method = new$method, fold = new$fold, value = new$heldout_log_score))
  order <- unique(d$method)
  stopifnot(!anyDuplicated(d[c("method", "fold")]),
            nrow(d) == length(order) * ifelse(task == "md", 3, 4))
  labels <- setNames(order, order)
  labels[c("MST tube", "MST SFD", "Spline SFD")] <-
    c("Previous MST tube", "Previous MST SFD", "Previous spline SFD")
  d$y <- match(d$method, rev(order)) + (d$fold - mean(unique(d$fold))) * .08
  means <- aggregate(value ~ method, d, mean)
  means$y <- match(means$method, rev(order))
  dev_plots[[task]] <- ggplot(d, aes(value, y)) +
    geom_point(size = 1.5, shape = 1) +
    geom_point(data = means, aes(value, y), shape = 124, size = 4) +
    scale_y_continuous(breaks = seq_along(order), labels = labels[rev(order)]) +
    labs(title = ifelse(task == "md", "a  Alanine dipeptide: 3 reused folds", "b  ECG: 4 reused folds"),
         x = ifelse(task == "md", "Log density (rad^-2)", "Log density (mV^-2)"), y = NULL)
  write_table(d, paste0(task, "_development_plot_data.csv"))
  a <- read_evidence(paste0("outputs/occupancy_development_20261019/", task, "/ablations.csv"))
  a$method <- factor(a$method, c("Uniform occupancy", "Learned branches", "Learned occupancy"),
                     c("Uniform", "Branch mass", "Occupancy"))
  ablation_plots[[task]] <- ggplot(a, aes(method, heldout_log_score, group = fold)) +
    geom_line(linewidth = .4, colour = grey) + geom_point(size = 1.7, shape = 1) +
    labs(title = ifelse(task == "md", "a  Alanine dipeptide", "b  ECG"),
         x = NULL, y = ifelse(task == "md", "Log density (rad^-2)", "Log density (mV^-2)"))
}
save_plot(dev_plots$md + dev_plots$ecg, "application_development", 9.2, 4.0)
save_plot(ablation_plots$md + ablation_plots$ecg, "matched_width_ablation", 7.0, 2.8)
}

cairo_pdf(file.path(out, "evidence_review.pdf"), width = 9.2, height = 5.0, family = plot_family)
for (p in plots) print(p$plot)
dev.off()
writeLines(capture.output(sessionInfo()), file.path(out, "sessionInfo.txt"))
captions <- c(
  "Mechanism: four prespecified upper-branch contrasts; 25 independent simulation repetitions. Points and 1.96 Monte Carlo SE bars summarize means, not parameter confidence intervals. Dashed lines show generating contrasts. Fixed and oracle kernel variants fit masses only; their other parameters are held fixed, not estimated.",
  "Geometry: 25 simulation repetitions per condition; all three SFD geometry variants retained. Panels a-e show mean contrasts with 1.96 Monte Carlo SE; panel f shows reference-group angle RMSE in degrees. Student-t generating noise has matched transverse SD.",
  "PBMC: training-fitted representations and excluded marker genes, 25 overlapping cell splits from one dataset. Bars show split Monte Carlo variability, not independent-subject uncertainty. Lower marker log loss is better; GMM memberships are retained even though they improve more than scalar SFD odds.",
  "Applications: every previous control and every new occupancy-development method is retained. Hollow points show individual reused trajectory/record folds, ticks descriptive means; no confidence intervals. These are development scores, not untouched confirmatory tests. Methods differ in their geometry, residual updates, scale search and parameter counts.",
  "Ablation: within each reused fold, the three variants have the same skeleton, width, residual kernel and observations. Width was selected on validation data by the occupancy variant. Lines connect folds, not time points; no sampling uncertainty claim. This comparison does not establish superiority to GMM or identify physical states."
)
if (!("--development" %in% args)) captions <- captions[1:3]
writeLines(captions, file.path(out, "captions.txt"))
hash <- function(paths) as.list(setNames(vapply(paths, function(p) digest::digest(file = p, algo = "sha256"), ""), paths))
manifest <- list(operation = "R read-only scientific evidence export", input_sha256 = hash(inputs),
  script_sha256 = hash("scripts/plot_sfd_evidence.R"),
  evidence_stage = c("frozen simulations", "single-dataset split diagnostic", "reused-data development"),
  output_sha256 = hash(list.files(out, full.names = TRUE)))
jsonlite::write_json(manifest, file.path(out, "export_manifest.json"), auto_unbox = TRUE, pretty = TRUE)
message("Exported ", length(plots), " figures and the review PDF to ", out)
