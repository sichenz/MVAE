old_bp <- read.csv("data/new_sources/bp_merged.csv")

new_bp <- read.csv("data/new_sources/newbp_1-10.csv")
for(name in c("11-100","101-200","201-300","301-400","401-500","501-End")){
  file <- paste("data/new_sources/newbp_",name,".csv", sep="")
  cur_bp <- read.csv(file)
  new_bp <- rbind(new_bp, cur_bp)
}

new_bp <- new_bp[new_bp$AssignmentStatus == "Approved",]
dim(new_bp)

new_names <- new_bp[,"Input.name"]
new_bp <- cbind(new_names, new_bp[,sapply(strsplit(colnames(new_bp), "[.]"), function(x) x[1]=="Answer")])
new_bp <- new_bp[, !colnames(new_bp) %in% c("Answer.boring","Answer.ugly","Answer.dishonest")]
colnames(new_bp) <- colnames(old_bp)

bp <- rbind(old_bp, new_bp)

write.csv(bp, file="data/new_sources/combined_old_new_bp.csv")
