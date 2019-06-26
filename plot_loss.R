# Title     : plot_loss
# Objective : Read log line by line and plot
# Created by: bijan
# Created on: 24.06.19

###################Load Stuff####################
require(ggplot2)



con <- file("runs/icnet_isic18/76660/run_2019_06_25_15_25_17.log", "r")
lines <- c()

while(TRUE) {
  line = readLines(con, 1)
  if(length(line) == 0) break
  else if(grepl("*INFO Iter*", line)) {
    lines <- c(lines, line)
  }
}
close(con)
print(lines)


split <- strsplit(lines, split=" ")

extract_iter <- function(line) {
  return (as.integer(gsub(".*\\[(.*)","\\1", gsub("(.*)\\/.*", "\\1", line))))
}

extract_iter(split[[4]][5])

dates <- c()
iter <- c()
loss <- c()
time_req <- c()

for (i in 1:length(split)) {
  dates <- c(dates,as.Date(split[[i]][1], "%Y-%m-%d"))
  iter <- c(iter, extract_iter(split[[i]][5]))
  loss <- c(loss,as.numeric(split[[i]][8]))
  time_req <- c(time_req,as.numeric(split[[i]][11]))
}

df <- data.frame(date=dates,
                 iter=iter,
                 loss=loss,
                 time_req=time_req)

names(df) <- c("date", "iter", "loss", "time_req")

ggplot(data = df) + geom_line(aes(x=df$iter, y=df$loss)) + 
  ylim(0,2) + ylab("Loss") + xlab("Iteration") +
  geom_smooth(aes(x=df$iter, y=df$loss))


###################Load Validation####################
con <- file("./runs/icnet_isic18/76660/run_2019_06_25_15_25_17.log", "r")

lines <- c()

while(TRUE) {
  line = readLines(con, 1)
  if(length(line) == 0) break
  else if(grepl("*Mean IoU*", line)) {
    lines <- c(lines, line)
  }
}
close(con)
print(lines)
split <- strsplit(lines, split=" ")

iou <- c()
for (i in 1:length(split)) {
  iou <- c(iou,split[[i]][8])
}
iou <- as.numeric(iou)
df <- data.frame(iou=iou)
df$epoch <- as.numeric(rownames(df))


ggplot(data = df) + geom_line(aes(x=df$epoch, y=df$iou)) + 
  ylim(0,1) + ylab("Intersection over Union") + xlab("Epoch") 

