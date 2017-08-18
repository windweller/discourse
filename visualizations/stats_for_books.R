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

internal = df%>% filter(location=="internal")
initial = df %>% filter(location=="initial")

sum(df$freq)/10^6
sum(clean$freq)/10^6
sum(internal$freq)/10^6
length(unique(clean$marker))
unique(clean$marker)
markers_w_no_clean_locations
