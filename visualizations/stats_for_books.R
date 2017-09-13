source("~/Settings/startup.R")
df = read.csv("stats_for_books.csv") %>%
  group_by(marker) %>%
  mutate(freq = file1 + file2,
         percent = freq/sum(freq)) %>%
  ungroup() %>% as.data.frame()
clean = df %>% filter(clean)

df %>% 
  mutate(group=paste(marker, location)) %>%
  ggplot(., aes(x=group, y=freq, fill=clean)) +
  geom_bar(stat="identity", position="dodge") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

df %>% 
  filter(location=="internal") %>%
  ggplot(., aes(x=marker, y=freq, fill=clean)) +
  geom_bar(stat="identity", position="dodge") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("sentence-internal examples")

markers_w_no_clean_locations = (df %>% group_by(marker) %>%
  summarise(any_clean = clean[1] || clean[2]) %>%
  filter(!any_clean) %>%
  ungroup() %>% as.data.frame())$marker

df %>% filter(clean) %>%
  mutate(group=paste(marker, location)) %>%
  ggplot(., aes(x=group, y=freq, fill=location)) +
  geom_bar(stat="identity", position="dodge") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

df %>% group_by(marker, location) %>%
  summarise(clean = clean[1]) %>%
  as.data.frame()

internal = df%>% filter(location=="internal")
initial = df %>% filter(location=="initial")

sum(df$freq)/10^6
sum(clean$freq)/10^6
sum(internal$freq)/10^6
length(unique(clean$marker))
unique(clean$marker)
markers_w_no_clean_locations

## aim for 2-3 million
thresh = 300000
sub = df %>% group_by(marker) %>%
  filter(clean) %>%
  summarise(freq = sum(freq)) %>%
  mutate(freq = ifelse(freq > thresh, thresh, freq))
sub %>%
  # filter(freq > 10000) %>%
  ggplot(., aes(x=marker, y=freq)) +
  geom_bar(stat="identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
sum(sub$freq)

sub = df %>% group_by(marker) %>%
  filter(clean) %>%
  summarise(freq = sum(freq)) 
sub %>%
  # filter(freq > 50000) %>%
  ggplot(., aes(x=marker, y=freq)) +
  geom_bar(stat="identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
sum(sub$freq)

markers = sub$marker
for (m in markers) {
  initial = (df %>% filter(marker==m & location=="initial"))$clean
  internal = (df %>% filter(marker==m & location=="internal"))$clean
  if (internal) {print(m)}
}


df %>% group_by(marker) %>%
  filter(location=="internal") %>%
  summarise(freq = sum(freq)) %>%
  filter(freq > 50000) %>%
  ggplot(., aes(x=marker, y=freq)) +
  geom_bar(stat="identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


df %>% group_by(marker) %>%
  summarise(freq = sum(freq)) %>%
  filter(freq > 50000) %>%
  ggplot(., aes(x=marker, y=freq)) +
  geom_bar(stat="identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))



(df %>% filter(marker=="but" & location=="internal"))$freq
sum((df %>% filter(marker=="for example"))$freq)

df %>% 
  filter(marker %in% c(
    "but", "because", "so", "before",
    "still", "for example", "if", "when")) %>%
  mutate(marker = factor(marker, levels=c(
    "but", "so", "if", "when", "because",
    "before", "still", "for example"
  ))) %>%
  group_by(marker) %>%
  summarise(freq = sum(freq)) %>%
  ggplot(., aes(x=marker, y=freq)) +
  geom_bar(stat="identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("simple8.png", width=5, height=3)

df %>% 
  group_by(marker) %>%
  filter(!(marker %in% c("and", "as", "also"))) %>%
  summarise(freq = sum(freq)) %>%
  mutate(marker = factor(marker, levels=marker[order(freq)])) %>%
  ggplot(., aes(x=marker, y=freq)) +
  geom_bar(stat="identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("all.png", width=5, height=3)

