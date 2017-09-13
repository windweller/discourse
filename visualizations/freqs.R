source("~/Settings/startup.R")
train = read.csv("../data/books/discourse_task_v1/labels.train")
dev = read.csv("../data/books/discourse_task_v1/labels.dev")
test = read.csv("../data/books/discourse_task_v1/labels.test")
freq = c(table(train), table(dev), table(test))
marker = names(freq)
split = c(rep("train", 8), rep("dev", 8), rep("test", 8))
df = data.frame(freq, marker, split)
df %>% mutate(split = factor(split, levels=c("train", "dev", "test"))) %>%
ggplot(., aes(x=marker, y=freq)) +
  geom_bar(stat="identity", position="dodge") +
  facet_wrap(~split, scale="free") +
  ylab("frequency") + 
  xlab("discourse marker") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("discourse_senteval_task.png", width=7, height=3)