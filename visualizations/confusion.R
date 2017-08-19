source("~/Settings/startup.R")
df = read.csv("books_LSTM_1_1024_all_tempmax_e1_confusion_test.csv")
confusions = table(df) %>% as.data.frame() %>%
  group_by(labels) %>%
  mutate(prop_gold = Freq / sum(Freq)) %>%
  ungroup()
diagonal = confusions %>% filter(labels==preds)
# my_levels = (0:13)[diagonal$labels[order(diagonal$prop_gold)]]
my_levels = c(11, 4, 7, 9,
              10, 8, 5, 6,
              13, 3, 12, 2,
              1, 0)
my_labels = my_levels
# my_labels = c("because", "although", "but", "when",
#               "for example", "before", "after", "however",
#               "so", "still", "though", "meanwhile",
#               "while", "if")[my_levels+1]
confusions %>% 
  mutate(labels = factor(labels, levels=my_levels, labels=my_labels),
         preds = factor(preds, levels=my_levels, labels=my_labels)) %>%
  ggplot(., aes(x=preds, y=labels, fill=log(prop_gold+0.001))) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  geom_tile() +
  ylab("true label") +
  xlab("model classification")
ggsave("zoomed_confusion.png", width=8, height=6)

confusions %>% 
  mutate(labels = factor(labels, levels=my_levels, labels=my_labels),
         preds = factor(preds, levels=my_levels, labels=my_labels)) %>%
  ggplot(., aes(x=preds, y=labels, fill=prop_gold)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  geom_tile() +
  ylab("true label") +
  xlab("model classification")
ggsave("confusion.png", width=8, height=6)

intuitive_levels = c(
  "if", "because",
  "however", "still", "though", "but", "although", "while",
  "when", "before", "after",
  "so", "for example", "meanwhile")
confusions %>% 
  mutate(labels = factor(labels, levels=my_levels, labels=my_labels),
         preds = factor(preds, levels=my_levels, labels=my_labels)) %>%
  mutate(labels = factor(labels, levels=intuitive_levels),
         preds = factor(preds, levels=intuitive_levels)) %>%
  # ggplot(., aes(x=preds, y=labels, fill=prop_gold)) +
  ggplot(., aes(x=preds, y=labels, fill=log(prop_gold+0.001))) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  geom_tile() +
  ylab("true label") +
  xlab("model classification")
ggsave("confusion_intuition.png", width=8, height=6)

intuitive_levels = c(
  "if", "because",
  "however", "still", "though", "but", "although", "while",
  "when", "before", "after",
  "so", "for example", "meanwhile")
confusions %>% 
  mutate(labels = factor(labels, levels=my_levels, labels=my_labels),
         preds = factor(preds, levels=my_levels, labels=my_labels)) %>%
  mutate(labels = factor(labels, levels=intuitive_levels),
         preds = factor(preds, levels=intuitive_levels)) %>%
  # ggplot(., aes(x=preds, y=labels, fill=prop_gold)) +
  ggplot(., aes(x=preds, y=labels, fill=log(prop_gold+0.001))) +
  geom_tile() +
  ylab("true label") +
  xlab("model classification") +
  theme_black +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  theme(legend.position = "none")
ggsave("confusion_intuition_black.png", width=7, height=6)

df %>%
  mutate(correct = labels==preds) %>%
  group_by(labels) %>%
  summarise(mean(correct)) %>%
  ungroup %>% as.data.frame %>%
  mutate(labels = factor(labels, levels=my_levels, labels=my_labels)) %>%
  ggplot(., aes(x=labels, y=`mean(correct)`)) +
  geom_bar(stat="identity") +
  theme_black +
  geom_hline(yintercept = 1/length(unique(df$labels)), colour="gray") +
  geom_hline(yintercept = 1, colour="gray") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  xlab("discourse marker") +
  ylab("classification recall") +
  ylim(0, 1)
ggsave("recall.png", width=9, height=6)