bpfull <- read.csv("data/new_sources/bp_merged.csv")

# Summarize by factors: ---------------------------------------------------

down.to.earth <- c("down.to.earth","family.oriented","small.town")
honest <- c("honest","sincere","real")
wholesome <- c("wholesome","original")
cheerful <- c("cheerful","sentimental","friendly")
daring <- c("daring","trendy","exciting")
spirited <- c("spirited","cool","young")
imaginative <- c("imaginative","unique")
up.to.date <- c("up.to.date","independent","contemporary")
reliable <- c("reliable","hard.working","secure")
intelligent <- c("intelligent","technical","corporate")
successful <- c("successful","leader","confident")
upper.class <- c("upper.class","glamorous","good.looking")
charming <- c("charming","feminine","smooth")
outdoorsy <- c("outdoorsy","masculine","western")
tough <- c("tough","rugged")

sincerity <- c(down.to.earth, honest, wholesome, cheerful)
excitement <- c(daring, spirited, imaginative, up.to.date)
competence <- c(reliable, intelligent, successful)
sophistication <- c(upper.class, charming)
ruggedness <- c(outdoorsy, tough)

all_factors <- c(sincerity, excitement, competence, sophistication, ruggedness)


compute.mean.factors <- function(bp_data){
  scores <- rep(NA,5)
  names(scores) <- c("sincerity","excitement","competence","sophistication","ruggedness")
  
  scores[1] <- mean(as.matrix(bp_data[,colnames(bp_data) %in% sincerity]))
  scores[2] <- mean(as.matrix(bp_data[,colnames(bp_data) %in% excitement]))
  scores[3] <- mean(as.matrix(bp_data[,colnames(bp_data) %in% competence]))
  scores[4] <- mean(as.matrix(bp_data[,colnames(bp_data) %in% sophistication]))
  scores[5] <- mean(as.matrix(bp_data[,colnames(bp_data) %in% ruggedness]))
  
  scores
}

compute.mean.subfactors <- function(bp_data){
  scores <- rep(NA,15)
  names(scores) <- c("down.to.earth","honest","wholesome","cheerful","daring",
                     "spirited","imaginative","up.to.date","reliable","intelligent",
                     "successful","upper.class","charming","outdoorsy","tough")
  
  scores[1] <- mean(as.matrix(bp_data[,colnames(bp_data) %in% down.to.earth]))
  scores[2] <- mean(as.matrix(bp_data[,colnames(bp_data) %in% honest]))
  scores[3] <- mean(as.matrix(bp_data[,colnames(bp_data) %in% wholesome]))
  scores[4] <- mean(as.matrix(bp_data[,colnames(bp_data) %in% cheerful]))
  scores[5] <- mean(as.matrix(bp_data[,colnames(bp_data) %in% daring]))
  scores[6] <- mean(as.matrix(bp_data[,colnames(bp_data) %in% spirited]))
  scores[7] <- mean(as.matrix(bp_data[,colnames(bp_data) %in% imaginative]))
  scores[8] <- mean(as.matrix(bp_data[,colnames(bp_data) %in% up.to.date]))
  scores[9] <- mean(as.matrix(bp_data[,colnames(bp_data) %in% reliable]))
  scores[10] <- mean(as.matrix(bp_data[,colnames(bp_data) %in% intelligent]))
  scores[11] <- mean(as.matrix(bp_data[,colnames(bp_data) %in% successful]))
  scores[12] <- mean(as.matrix(bp_data[,colnames(bp_data) %in% upper.class]))
  scores[13] <- mean(as.matrix(bp_data[,colnames(bp_data) %in% charming]))
  scores[14] <- mean(as.matrix(bp_data[,colnames(bp_data) %in% outdoorsy]))
  scores[15] <- mean(as.matrix(bp_data[,colnames(bp_data) %in% tough]))
  
  scores
}


factor_scores <- do.call(rbind, by(bpfull[,-1], bpfull$name, compute.mean.factors))
subfactor_scores <- do.call(rbind, by(bpfull[,-1], bpfull$name, compute.mean.subfactors))

# Sample and compute ------------------------------------------------------

compute.subsample.factors <- function(bp_data, size){
  samp <- sample(1:nrow(bp_data), size, replace=T)
  compute.mean.factors(bp_data[samp,])
}

compute.subsample.subfactors <- function(bp_data, size){
  samp <- sample(1:nrow(bp_data), size, replace=T)
  compute.mean.subfactors(bp_data[samp,])
}

factor_mae <- matrix(NA, length(5:20), 2)
subfactor_mae <- matrix(NA, length(5:20), 2)
index <- 1
for(size in 5:20){
  print(size)
  nboot <- 100
  factor_boot_out <- array(NA, dim=c(dim(factor_scores),nboot))
  subfactor_boot_out <- array(NA, dim=c(dim(subfactor_scores),nboot))
  for(i in 1:nboot){
    factor_boot_out[,,i] <- do.call(rbind, by(bpfull[,-1], bpfull$name, compute.subsample.factors, size=size)) - factor_scores
    subfactor_boot_out[,,i] <- do.call(rbind, by(bpfull[,-1], bpfull$name, compute.subsample.subfactors, size=size)) - subfactor_scores
  }
  
  factor_mae[index,] <- c(size, mean(apply(factor_boot_out, c(2,3), function(x) mean(abs(x), na.rm=T))))
  subfactor_mae[index,] <- c(size, mean(apply(subfactor_boot_out, c(2,3), function(x) mean(abs(x), na.rm=T))))
  index <- index+1
}

tab <- cbind(factor_mae,subfactor_mae[,-1])
colnames(tab) <- c("nboot","factor_mae","subfactor_mae")

plot(tab[,"nboot"],tab[,"factor_mae"], type="l", col=2, lwd=2, xlab="Nboot", ylab="MAE", main="Red = Factors, Blue = Subfactors")
lines(tab[,"nboot"],tab[,"subfactor_mae"], type="l", col=4, lwd=2)

