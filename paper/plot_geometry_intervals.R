#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: Rscript plot_sfd_geometry_intervals.R SUMMARY_CSV NEW_OUT")
suppressPackageStartupMessages(library(ggplot2))
out <- args[[2]]
if (dir.exists(out)) stop("Refusing to overwrite a figure directory")
d <- read.csv(args[[1]], stringsAsFactors = FALSE)
stopifnot(nrow(d) == 378L, all(d$total == 25L), all(d$valid + d$failures == d$total))
keep <- d$kind == "contrast" &
  ((d$scenario == "mass" & d$target == "mass") |
   (d$scenario == "position" & d$target == "mean_position") |
   (d$scenario == "width" & d$target == "width"))
pdat <- d[keep, ]
stopifnot(nrow(pdat) == 54L)
pdat$condition <- factor(pdat$condition, levels = c("nominal", "overlap", "heavy_noise"),
                         labels = c("Gaussian", "Overlapping arms", "Student-t noise"))
pdat$method <- factor(pdat$method, levels = c("SFD supplied", "SFD rotated", "SFD learned angles"),
                      labels = c("Supplied", "Rotated", "Learned"))
pdat$target <- factor(pdat$target, levels = c("mass", "mean_position", "width"),
                      labels = c("Mass contrast", "Position contrast", "Width contrast"))
pdat$covariance <- factor(pdat$covariance, levels = c("information", "sandwich"),
                          labels = c("Information", "Sandwich"))
pdat$count <- paste0(pdat$covered, "/", pdat$total)
pdat$count_y <- ifelse(pdat$covariance == "Information", 1.08, 1.20)
dodge <- position_dodge(width = .65)
p <- ggplot(pdat, aes(x = method, y = coverage_all, colour = covariance, shape = covariance)) +
  geom_hline(yintercept = .95, colour = "#777777", linetype = "dashed", linewidth = .3) +
  geom_errorbar(aes(ymin = coverage_mc_lower, ymax = coverage_mc_upper),
                width = .13, linewidth = .4, position = dodge) +
  geom_point(size = 2, position = dodge) +
  geom_text(aes(y = count_y, label = count), position = dodge,
            family = "Times", size = 2.8, show.legend = FALSE) +
  facet_grid(condition ~ target) +
  scale_colour_manual(values = c("Information" = "#176B87", "Sandwich" = "#C76D32")) +
  scale_shape_manual(values = c("Information" = 16, "Sandwich" = 17)) +
  scale_y_continuous(limits = c(0, 1.27), breaks = c(0, .5, .95), expand = expansion(mult = 0)) +
  labs(x = NULL, y = "Coverage of nominal 95% parameter intervals", colour = NULL, shape = NULL) +
  theme_classic(base_family = "Times", base_size = 10) +
  theme(strip.background = element_blank(), strip.text = element_text(face = "bold", size = 10),
        axis.text = element_text(colour = "black", size = 9), panel.spacing = grid::unit(10, "pt"),
        legend.position = "bottom", plot.margin = margin(7, 7, 4, 7))
dir.create(out, recursive = TRUE)
ggsave(file.path(out, "geometry_interval_coverage.pdf"), p, width = 8.1, height = 7.6, device = cairo_pdf)
ggsave(file.path(out, "geometry_interval_coverage.png"), p, width = 8.1, height = 7.6, dpi = 300, bg = "white")
write.csv(pdat, file.path(out, "figure_input.csv"), row.names = FALSE, na = "")
writeLines(capture.output(sessionInfo()), file.path(out, "R_sessionInfo.txt"))
writeLines(paste(
  "Post-hoc coverage of nominal 95% Wald intervals in the saved two-arm geometry study.",
  "Panels show the three predeclared intended contrasts under each generating condition.",
  "All directions, occupancy and widths active in a fit enter its complete covariance.",
  "Points and counts use all 25 repetitions; numerical failures count as noncoverage.",
  "Bars are exact binomial 95% Monte Carlo intervals for coverage, not parameter intervals.",
  "The dashed line marks nominal coverage. Information and sandwich calculations use the same saved fit.",
  "Under rotated geometry or Student-t noise, coverage concerns the generating target, not an estimated pseudo-true target.",
  "Sandwich covariance changes estimated variability, not misspecification bias.",
  "All cross-effects and failure records remain in the complete summary tables."
), file.path(out, "caption.txt"))
