source("~/Settings/startup.R")
train = read.csv("labels.train")
dev = read.csv("labels.dev")
test = read.csv("labels.test")
freq = c(table(train), table(dev), table(test))
marker = names(freq)
split = c(rep("train", 8), rep("dev", 8), rep("test", 8))
df = data.frame(freq, marker, split)
ggplot(df, aes(x=marker, y=freq, fill=marker)) +
  geom_bar(stat="identity", position="dodge") +
  facet_wrap(~split, scale="free")