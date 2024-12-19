setwd("~/Dropbox/1_proj/logos/")
setwd("/media/ryan/hdd/Dropbox/1_proj/logos/")

new <- read.csv("data/new_sources/all_combined.csv")
load("data/cb_full_w_cats.RData")
cb <- df; rm(df)

new$name <- as.character(new$name)
cb$id <- as.character(cb$id)

new$cb <- NA
new$method <- NA

matched_c2n <- pmatch(cb$id, new$name, duplicates.ok = FALSE)
new$cb[na.omit(matched_c2n)] <- cb$id[!is.na(matched_c2n)]
new$method[na.omit(matched_c2n)] <- 2

matched_n2c <- pmatch(new$name, cb$id, duplicates.ok = FALSE)
new$cb[!is.na(matched_n2c)] <- cb$id[na.omit(matched_n2c)]
new$method[!is.na(matched_n2c)] <- 1

new$cb[new$name %in% cb$id] <- new$name[new$name %in% cb$id]
new$method[new$name %in% cb$id] <- 0

