setwd("~/Dropbox/1_proj/logos/")
setwd("/media/ryan/hdd/Dropbox/1_proj/logos/")

load("data/cb_full_w_cats.RData")
cb <- df; rm(df)

comb <- read.csv("data/new_sources/all_combined.csv")

comb$desc <- ""
comb$desc[comb$cb != ""] <- cb$desc[na.omit(match(comb$cb, cb$id))]

write.csv(comb, "data/all_combined_desc.csv")
