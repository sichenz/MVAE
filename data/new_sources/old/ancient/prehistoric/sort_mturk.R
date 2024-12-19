mturk <- read.csv("/media/ryan/hdd/Dropbox/1_proj/logos/data/new_sources/mturk/final_results.csv")
mturk$desc <- as.character(mturk$desc)
charcounts <- sapply(mturk$desc, nchar)
names(charcounts) <- c()

write.csv(mturk[order(charcounts, decreasing=FALSE),-4], file="/media/ryan/hdd/Dropbox/1_proj/logos/data/new_sources/mturk/mturk_sorted.csv")

write.csv(mturk[charcounts < 300,-4], file="/media/ryan/hdd/Dropbox/1_proj/logos/data/new_sources/mturk/mturk_short_descs.csv")
