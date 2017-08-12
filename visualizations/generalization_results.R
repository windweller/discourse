df = read.csv("generalization_results.csv")
source("~/Settings/startup.R")

tasks = df %>% group_by(task) %>% summarise(performance = mean(performance))

df = df %>% mutate(task = factor(task, levels = tasks$task[order(tasks$performance)]))

df %>% 
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  mutate(group = paste(ssplit, layers, split)) %>%
  # mutate(split = factor(ifelse(is.na(split), "unknown", char(split)))) %>%
  ggplot(., aes(x=task, y=performance, shape=marker_set, colour=marker_set)) +
  # geom_bar(stat="identity", position="dodge") +
  geom_point(alpha=0.7) +
  facet_wrap(~ group) +
  scale_shape_manual(values=1:10) +
  # ylim(0,100) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("which_marker_set.png", width=10, height=6)

df %>% 
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  filter(ssplit=="dep") %>%
  filter(layers==2) %>%
  filter(split=="rand") %>%
  mutate(group = paste(ssplit, layers, split)) %>%
  # mutate(split = factor(ifelse(is.na(split), "unknown", char(split)))) %>%
  ggplot(., aes(x=task, y=performance, shape=marker_set, colour=marker_set)) +
  # geom_bar(stat="identity", position="dodge") +
  geom_point(alpha=0.7) +
  facet_wrap(~ group) +
  # ylim(0,100) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("which_marker_set_zoom1.png", width=5, height=4)


df %>% 
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  filter(ssplit=="string") %>%
  filter(layers==1) %>%
  filter(split=="rand") %>%
  mutate(group = paste(ssplit, layers, split)) %>%
  # mutate(split = factor(ifelse(is.na(split), "unknown", char(split)))) %>%
  ggplot(., aes(x=task, y=performance, fill=marker_set, colour=marker_set)) +
  # geom_bar(stat="identity", position="dodge") +
  geom_point(alpha=0.7) +
  facet_wrap(~ group) +
  # ylim(0,100) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("which_marker_set_zoom2.png", width=5, height=4)


df %>% 
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  filter(ssplit=="dep") %>%
  filter(layers==1) %>%
  filter(split=="rand") %>%
  mutate(group = paste(ssplit, layers, split)) %>%
  # mutate(split = factor(ifelse(is.na(split), "unknown", char(split)))) %>%
  ggplot(., aes(x=task, y=performance, fill=marker_set, colour=marker_set)) +
  # geom_bar(stat="identity", position="dodge") +
  geom_point(alpha=0.7) +
  facet_wrap(~ group) +
  # ylim(0,100) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("which_marker_set_zoom3.png", width=5, height=4)


df %>% 
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  mutate(group = paste(marker_set, layers, split)) %>%
  # mutate(split = factor(ifelse(is.na(split), "unknown", char(split)))) %>%
  ggplot(., aes(x=task, y=performance, fill=ssplit, colour=ssplit)) +
  # geom_bar(stat="identity", position="dodge") +
  geom_point(alpha=0.7) +
  facet_wrap(~ group) +
  # ylim(0,100) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("which_extraction_ssplit.png", width=10, height=6)


df %>% 
  filter(marker_set %in% c("but/because", "five")) %>%
  filter(split=="rand") %>%
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  mutate(group = paste(marker_set, layers, split)) %>%
  # mutate(split = factor(ifelse(is.na(split), "unknown", char(split)))) %>%
  ggplot(., aes(x=task, y=performance, fill=ssplit, colour=ssplit)) +
  # geom_bar(stat="identity", position="dodge") +
  geom_point(alpha=0.7) +
  facet_wrap(~ group) +
  # ylim(0,100) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("which_extraction_ssplit_zoom.png", width=10, height=6)


df %>% 
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  mutate(group = paste(marker_set, ssplit, split)) %>%
  mutate(layers = factor(layers)) %>%
  # mutate(split = factor(ifelse(is.na(split), "unknown", char(split)))) %>%
  ggplot(., aes(x=task, y=performance, fill=layers, colour=layers)) +
  # geom_bar(stat="identity", position="dodge") +
  geom_point(alpha=0.7) +
  facet_wrap(~ group) +
  # ylim(0,100) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("which_num_layers.png", width=10, height=6)

df %>% 
  filter(marker_set=="but/because") %>%
  filter(ssplit=="string") %>%
  filter(split=="rand") %>%
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  mutate(group = paste(marker_set, ssplit, split)) %>%
  mutate(layers = factor(layers)) %>%
  # mutate(split = factor(ifelse(is.na(split), "unknown", char(split)))) %>%
  ggplot(., aes(x=task, y=performance, fill=layers, colour=layers)) +
  # geom_bar(stat="identity", position="dodge") +
  geom_point(alpha=0.7) +
  facet_wrap(~ group) +
  # ylim(0,100) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("which_num_layers_zoom1.png", width=5, height=4)

df %>% 
  filter(marker_set %in% c("but/because", "five")) %>%
  filter(ssplit=="string") %>%
  filter(split=="rand") %>%
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  mutate(group = paste(marker_set, ssplit, split)) %>%
  mutate(layers = factor(layers)) %>%
  # mutate(split = factor(ifelse(is.na(split), "unknown", char(split)))) %>%
  ggplot(., aes(x=task, y=performance, fill=layers, colour=layers)) +
  # geom_bar(stat="identity", position="dodge") +
  geom_point(alpha=0.7) +
  facet_wrap(~ group) +
  # ylim(0,100) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("which_num_layers_zoom2.png", width=8, height=4)


new_df = df %>% 
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  mutate(group = paste(marker_set, ssplit, split, layers))
new_df %>%
  ggplot(., aes(x=task, y=performance, colour=group, shape=group)) +
  # geom_bar(stat="identity", position="dodge") +
  geom_point(alpha=0.7) +
  scale_shape_manual(values=1:20) +
  # ylim(0,100) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("all_generalization_experiments.png", width=12, height=8)

