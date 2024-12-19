try(setwd("~/Dropbox/1_proj/logos/"), silent=TRUE)
try(setwd("/media/ryan/hdd/Dropbox/1_proj/logos/"), silent = TRUE)

setwd("data/new_sources/")
curlogos <- list.files("logos/")
curlogos_names <- sapply(curlogos, function(x) strsplit(x, split="[.]")[[1]][1])
names(curlogos_names) <- NULL

fdat <- read.csv("all_combined_FINAL.csv")
fdat$logo <- fdat$name %in% curlogos_names

fdat$name[fdat$logo == FALSE]

fdat$logo_file <- curlogos[match(fdat$name, curlogos_names)]
fdat$logo_url <- paste("http://www.columbia.edu/~rtd2118/logos/", fdat$logo_file, sep="")

write.csv(fdat, file="all_combined_FINAL_with_URLS.csv")
