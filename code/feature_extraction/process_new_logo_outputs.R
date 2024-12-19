rm(list = ls())

setwd("~/Dropbox/1_proj/logos/")

source("code/extract_features/process_new_logo_outputs_functions.R")


name = "koinos"

colors <- read.csv(paste("code/extract_features/new_logo_outputs/", name, "_colors_labeled.csv", sep=""))
hulls <- read.csv(paste("code/extract_features/new_logo_outputs/", name, "_hull_labeled.csv", sep=""))
fonts <- read.csv(paste("code/extract_features/new_logo_outputs/", name, "_fonts.csv", sep=""))
global_feats <- read.csv(paste("code/extract_features/new_logo_outputs/", name, "_summary_feats.csv", sep=""))
marks <- read.csv(paste("code/extract_features/new_logo_outputs/", name, "_marks_labeled.csv", sep=""))
mark_feats <- read.csv(paste("code/extract_features/new_logo_outputs/", name, "_mark_features.csv", sep=""))


if(!exists("marks")){
  marks = NA
  mark_feats = NA
}

feature_list = create.logo.features(name, colors, hulls, fonts, global_feats, marks, mark_feats)

new_y_bin = feature_list$bin_feats
new_y_mult = feature_list$mult_feats

old_y_bin = read.csv(file="data/y_bin_all_py2.csv", row.names=1)
old_y_mult = read.csv(file="data/y_mult_ncolors_py2.csv", row.names=1)

new_y_bin = new_y_bin[,match(colnames(old_y_bin), colnames(new_y_bin))]

all(colnames(new_y_bin) == colnames(old_y_bin))
all(colnames(new_y_mult) == colnames(old_y_mult))


write.csv(new_y_bin, file=paste("code/extract_features/new_logo_outputs/",name,"_y_bin.csv", sep=""), row.names = TRUE)
write.csv(new_y_mult, file=paste("code/extract_features/new_logo_outputs/",name,"_y_mult.csv", sep=""), row.names = TRUE)
