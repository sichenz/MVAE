
try(setwd("~/Dropbox (CBS)/logos/"), silent=TRUE)
try(setwd("~/Dropbox/1_proj/logos/"), silent=TRUE)
try(setwd("/media/ryan/hdd/Dropbox/1_proj/logos/"), silent=TRUE)


df <- read.csv("data/new_sources/final_cleaned_new100.csv")
bp <- read.csv("data/new_sources/bp_factors.csv")
new <- read.csv("data/new_sources/mturk/get_more_wd_combined_batch_results.csv")

colnames(new)
colnames(df)

df$name <- as.character(df$name)
df$brand_identity <- as.character(df$brand_identity)
df <- df[df$name %in% new$Input.name,]
df <- df[order(df$name),]

new$Input.name <- as.character(new$Input.name)
new$Answer.desc <- as.character(new$Answer.desc)
new <- new[new$Input.name %in% df$name,]
new <- new[order(new$Input.name),]

all(df$name == new$Input.name)

new_df <- df
for(r in 1:nrow(new_df)){
  new_df$brand_identity[r] <- paste(df$brand_identity[r], new$Answer.desc[r], collapse=" ")
}


write.csv(new_df[1:10,], file="data/new_sources/mturk/get_full_bp_inputs_1-10.csv", row.names=FALSE)
write.csv(new_df[11:100,], file="data/new_sources/mturk/get_full_bp_inputs_11-100.csv", row.names=FALSE)
write.csv(new_df[101:200,], file="data/new_sources/mturk/get_full_bp_inputs_101-200.csv", row.names=FALSE)
write.csv(new_df[201:300,], file="data/new_sources/mturk/get_full_bp_inputs_201-300.csv", row.names=FALSE)
write.csv(new_df[301:400,], file="data/new_sources/mturk/get_full_bp_inputs_301-400.csv", row.names=FALSE)
write.csv(new_df[401:500,], file="data/new_sources/mturk/get_full_bp_inputs_401-500.csv", row.names=FALSE)
write.csv(new_df[501:619,], file="data/new_sources/mturk/get_full_bp_inputs_501-619.csv", row.names=FALSE)