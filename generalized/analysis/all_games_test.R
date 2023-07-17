# Train
df_1 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/block_game/combined_compressed.csv")
df_1$Rewards = (df_1$Rewards - min(df_1$Rewards)) / (max(df_1$Rewards) - min(df_1$Rewards))

df_2 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/chicken_game/combined_compressed.csv")
df_2$Rewards = (df_2$Rewards - min(df_2$Rewards)) / (max(df_2$Rewards) - min(df_2$Rewards))

df_3 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/coordination_game/combined_compressed.csv")
df_3$Rewards = (df_3$Rewards - min(df_3$Rewards)) / (max(df_3$Rewards) - min(df_3$Rewards))

df_4 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/pennies_game/combined_compressed.csv")
df_4$Rewards = (df_4$Rewards - min(df_4$Rewards)) / (max(df_4$Rewards) - min(df_4$Rewards))

df_5 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/prionsers_dilemma_game/combined_compressed.csv")
df_5$Rewards = (df_5$Rewards - min(df_5$Rewards)) / (max(df_5$Rewards) - min(df_5$Rewards))

final_df <- rbind(df_1, df_2, df_3, df_4, df_5)

mean(final_df[final_df$Agent == "Algaater",]$Rewards)
sd(final_df[final_df$Agent == "Algaater",]$Rewards)
mean(final_df[final_df$Agent == "BBL",]$Rewards)
sd(final_df[final_df$Agent == "BBL",]$Rewards)
mean(final_df[final_df$Agent == "EEE",]$Rewards)
sd(final_df[final_df$Agent == "EEE",]$Rewards)
mean(final_df[final_df$Agent == "S++",]$Rewards)
sd(final_df[final_df$Agent == "S++",]$Rewards)

model <- lm(final_df$Rewards ~ final_df$Agent)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "final_df$Agent", conf.level=0.95)
print(tukey)

# Test
df_1 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/block_game/combined_compressed_test.csv")
df_1$Rewards = (df_1$Rewards - min(df_1$Rewards)) / (max(df_1$Rewards) - min(df_1$Rewards))

df_2 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/chicken_game/combined_compressed_test.csv")
df_2$Rewards = (df_2$Rewards - min(df_2$Rewards)) / (max(df_2$Rewards) - min(df_2$Rewards))

df_3 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/coordination_game/combined_compressed_test.csv")
df_3$Rewards = (df_3$Rewards - min(df_3$Rewards)) / (max(df_3$Rewards) - min(df_3$Rewards))

df_4 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/pennies_game/combined_compressed_test.csv")
df_4$Rewards = (df_4$Rewards - min(df_4$Rewards)) / (max(df_4$Rewards) - min(df_4$Rewards))

df_5 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/prionsers_dilemma_game/combined_compressed_test.csv")
df_5$Rewards = (df_5$Rewards - min(df_5$Rewards)) / (max(df_5$Rewards) - min(df_5$Rewards))

final_df <- rbind(df_1, df_2, df_3, df_4, df_5)

mean(final_df[final_df$Agent == "Algaater",]$Rewards)
sd(final_df[final_df$Agent == "Algaater",]$Rewards)
mean(final_df[final_df$Agent == "BBL",]$Rewards)
sd(final_df[final_df$Agent == "BBL",]$Rewards)
mean(final_df[final_df$Agent == "EEE",]$Rewards)
sd(final_df[final_df$Agent == "EEE",]$Rewards)
mean(final_df[final_df$Agent == "S++",]$Rewards)
sd(final_df[final_df$Agent == "S++",]$Rewards)

model <- lm(final_df$Rewards ~ final_df$Agent)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "final_df$Agent", conf.level=0.95)
print(tukey)

# Changers
df_1 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/block_game/combined_compressed_test_changers.csv")
df_1$Rewards = (df_1$Rewards - min(df_1$Rewards)) / (max(df_1$Rewards) - min(df_1$Rewards))

df_2 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/chicken_game/combined_compressed_test_changers.csv")
df_2$Rewards = (df_2$Rewards - min(df_2$Rewards)) / (max(df_2$Rewards) - min(df_2$Rewards))

df_3 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/coordination_game/combined_compressed_test_changers.csv")
df_3$Rewards = (df_3$Rewards - min(df_3$Rewards)) / (max(df_3$Rewards) - min(df_3$Rewards))

df_4 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/pennies_game/combined_compressed_test_changers.csv")
df_4$Rewards = (df_4$Rewards - min(df_4$Rewards)) / (max(df_4$Rewards) - min(df_4$Rewards))

df_5 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/prionsers_dilemma_game/combined_compressed_test_changers.csv")
df_5$Rewards = (df_5$Rewards - min(df_5$Rewards)) / (max(df_5$Rewards) - min(df_5$Rewards))

final_df <- rbind(df_1, df_2, df_3, df_4, df_5)

mean(final_df[final_df$Agent == "Algaater",]$Rewards)
sd(final_df[final_df$Agent == "Algaater",]$Rewards)
mean(final_df[final_df$Agent == "BBL",]$Rewards)
sd(final_df[final_df$Agent == "BBL",]$Rewards)
mean(final_df[final_df$Agent == "EEE",]$Rewards)
sd(final_df[final_df$Agent == "EEE",]$Rewards)
mean(final_df[final_df$Agent == "S++",]$Rewards)
sd(final_df[final_df$Agent == "S++",]$Rewards)

model <- lm(final_df$Rewards ~ final_df$Agent)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "final_df$Agent", conf.level=0.95)
print(tukey)

# Self-play
df_1 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/block_game/combined_self_play.csv")
df_1$Rewards = (df_1$Rewards - min(df_1$Rewards)) / (max(df_1$Rewards) - min(df_1$Rewards))

df_2 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/chicken_game/combined_self_play.csv")
df_2$Rewards = (df_2$Rewards - min(df_2$Rewards)) / (max(df_2$Rewards) - min(df_2$Rewards))

df_3 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/coordination_game/combined_self_play.csv")
df_3$Rewards = (df_3$Rewards - min(df_3$Rewards)) / (max(df_3$Rewards) - min(df_3$Rewards))

df_4 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/pennies_game/combined_self_play.csv")
df_4$Rewards = (df_4$Rewards - min(df_4$Rewards)) / (max(df_4$Rewards) - min(df_4$Rewards))

df_5 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/prionsers_dilemma_game/combined_self_play.csv")
df_5$Rewards = (df_5$Rewards - min(df_5$Rewards)) / (max(df_5$Rewards) - min(df_5$Rewards))

final_df <- rbind(df_1, df_2, df_3, df_4, df_5)

mean(final_df[final_df$Agent == "Algaater",]$Rewards)
sd(final_df[final_df$Agent == "Algaater",]$Rewards)
mean(final_df[final_df$Agent == "BBL",]$Rewards)
sd(final_df[final_df$Agent == "BBL",]$Rewards)
mean(final_df[final_df$Agent == "EEE",]$Rewards)
sd(final_df[final_df$Agent == "EEE",]$Rewards)
mean(final_df[final_df$Agent == "S++",]$Rewards)
sd(final_df[final_df$Agent == "S++",]$Rewards)

model <- lm(final_df$Rewards ~ final_df$Agent)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "final_df$Agent", conf.level=0.95)
print(tukey)

# Smart
df_1 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/block_game/combined_compressed_smart.csv")
df_1$Rewards = (df_1$Rewards - min(df_1$Rewards)) / (max(df_1$Rewards) - min(df_1$Rewards))

df_2 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/chicken_game/combined_compressed_smart.csv")
df_2$Rewards = (df_2$Rewards - min(df_2$Rewards)) / (max(df_2$Rewards) - min(df_2$Rewards))

df_3 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/coordination_game/combined_compressed_smart.csv")
df_3$Rewards = (df_3$Rewards - min(df_3$Rewards)) / (max(df_3$Rewards) - min(df_3$Rewards))

df_4 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/pennies_game/combined_compressed_smart.csv")
df_4$Rewards = (df_4$Rewards - min(df_4$Rewards)) / (max(df_4$Rewards) - min(df_4$Rewards))

df_5 <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/prionsers_dilemma_game/combined_compressed_smart.csv")
df_5$Rewards = (df_5$Rewards - min(df_5$Rewards)) / (max(df_5$Rewards) - min(df_5$Rewards))

final_df <- rbind(df_1, df_2, df_3, df_4, df_5)

mean(final_df[final_df$Agent == "Algaater",]$Rewards)
sd(final_df[final_df$Agent == "Algaater",]$Rewards)
mean(final_df[final_df$Agent == "BBL",]$Rewards)
sd(final_df[final_df$Agent == "BBL",]$Rewards)
mean(final_df[final_df$Agent == "EEE",]$Rewards)
sd(final_df[final_df$Agent == "EEE",]$Rewards)
mean(final_df[final_df$Agent == "S++",]$Rewards)
sd(final_df[final_df$Agent == "S++",]$Rewards)

model <- lm(final_df$Rewards ~ final_df$Agent)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "final_df$Agent", conf.level=0.95)
print(tukey)

