df = read.csv("data/lengths.csv")
names(df) = c("len1", "len2", "connective", "version", "source")
df = df %>% mutate(file = paste(source, version)) %>%
  gather("var", "val", 1:2) %>%
  mutate(file = factor(file, levels=c(
    "ptb test",
    "ptb valid",
    "ptb train",
    "wikitext-103 test",
    "wikitext-103 valid",
    "wikitext-103 train"
  )))
df %>% filter(source =="ptb") %>%
  ggplot(., aes(x=val, colour=connective, fill=connective)) +
  geom_histogram(alpha=1/4) +
  # geom_vline(xintercept = 35) +
  facet_grid(file ~ var, scale="free")
df %>% filter(source =="wikitext-103") %>%
  ggplot(., aes(x=val, colour=connective, fill=connective)) +
  geom_histogram(alpha=1/3) +
  geom_vline(xintercept = 35) +
  facet_grid(file ~ var, scale="free") +
  xlim(0, 100)

df %>% filter(source == "ptb" & version=="train") %>%
  summarise(mean(val <= 35))

df %>% filter(source == "wikitext-103" & version=="train") %>%
  summarise(mean(val <= 35))