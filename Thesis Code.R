#install.packages("Hmisc")
#install.packages("ADNIMERGE_0.0.1.tar.gz", repos = NULL, type = "source")
#install.packages(corrplot)

#library(Hmisc) 
#library(ADNIMERGE) 
library(ggplot2) 
library(dplyr)
library(corrplot)

# Load Data 
adnimerge <- read.delim("ADNIMERGE.csv", sep = ',', stringsAsFactors = FALSE)
vitals <- read.delim("VITALS.csv", sep = ',', stringsAsFactors = FALSE)
postproc <- read.delim("Biospecimen_Results/ADMC_BA_POSTPROC_06_28_18.csv", 
                       sep = ',', stringsAsFactors = FALSE)

adni_dict <- read.delim("ADNIMERGE_DICT.csv", sep =',')
postproc_dict <- read.delim("Biospecimen_Results/ADMC_BA_POSTPROC_06_28_18_DICT.csv", 
                            sep = ',', stringsAsFactors = FALSE)

# Initial number of participants in datasets
count(distinct(adnimerge, RID)) #2231
count(distinct(postproc, RID)) #1671
count(distinct(vitals, RID)) #3282

#z <- adnimerge[which(adnimerge$RID %in% postproc$RID),]
#count(distinct(z,RID))
#group_by(z, DX_bl) %>% summarise(n_distinct(RID))

# Discard irrelevant variables 
vitals2 <- vitals %>% select(RID, VISCODE, VSWEIGHT, VSWTUNIT, VSHEIGHT, VSHTUNIT)
postproc2 <- postproc[ , -c(1,5:9,34:57)] 
adnimerge2 <- adnimerge[ , c(1,3,8:11,15,19:21,60,103:105,109:110,112)] %>%
  mutate(DX_bl = ifelse(DX_bl == "LMCI"|DX_bl == "EMCI", 'MCI', DX_bl)) %>%
  rename("GENDER" = PTGENDER, "EDUCATION" = PTEDUCAT) %>% 
  mutate(GENDER = as.factor(GENDER)) %>%
  arrange(RID, VISCODE)

# Process info for BMI meausure 
bmi <- vitals2 %>% 
  select(RID, VISCODE,VSHEIGHT, VSHTUNIT, VSWEIGHT, VSWTUNIT) %>%
  mutate(VSHEIGHT = ifelse(VSHTUNIT == 1, VSHEIGHT * 2.54, VSHEIGHT)) %>%
  mutate(VSWEIGHT = ifelse(VSWTUNIT == 1, VSWEIGHT / 2.205, VSWEIGHT)) %>%
  filter(VSHEIGHT != -4 & !is.na(VSHEIGHT)) %>%
  filter(VISCODE %in% c("bl", "sc", "v01")) %>%
  arrange(RID)


#Check if there is just one time of measurement per participant
counts <- group_by(bmi, RID) %>% count()
counts[counts$n > 1,] 

# Check if heights are correct
too_tall <- bmi[bmi$VSHEIGHT > 200,] 
too_tall
too_tall_ids <- too_tall[3:5,1]

mutate_cond <- function(.data, condition, ..., envir = parent.frame()) {
  condition <- eval(substitute(condition), .data, envir)
  .data[condition, ] <- .data[condition, ] %>% mutate(...)
  .data
}
# Source: https://stackoverflow.com/questions/34096162/dplyr-mutate-replace-several-columns-on-a-subset-of-rows
  
bmi <- bmi %>% mutate_cond(RID %in% too_tall_ids, VSHEIGHT = VSHEIGHT/2.54) 
bmi[bmi$RID %in% too_tall_ids,]

bmi <- bmi %>% 
  mutate(VSHEIGHT = VSHEIGHT/100) %>%
  mutate(BMI = VSWEIGHT/(VSHEIGHT*VSHEIGHT)) %>% 
  mutate(VISCODE = "bl") %>%
  select(-c(VSHTUNIT,VSWTUNIT))


# Remove patients with SMC and include only those with data available for three years
group_by(adnimerge2, DX_bl) %>% summarise(n_distinct(RID))   
  
adnimerge2 <- adnimerge2 %>% 
  filter(DX_bl != 'SMC')  %>% 
  filter(Years_bl <= 3.2) %>% 
  group_by(RID) %>% 
  mutate(max_data = (max(M))) %>%
  filter(max_data == 36) %>%
  select(-c(max_data, Years_bl)) %>%
  ungroup(RID)

#postproc2 <- filter(postproc2, SUBJECT_FLAG != 1)  #107
#postproc_dict %>% filter(FLDNAME == 'SUBJECT_FLAG') %>% select(TEXT) 
#postproc_dict %>% filter(FLDNAME == "TLCA_CDCA_LOGTRANSFORMFLAG") %>% select(TEXT) 

# Merge data 
total <- adnimerge2 %>%
  left_join(bmi, c("RID", "VISCODE")) %>%
  left_join(postproc2, by = c("RID", "VISCODE")) %>% 
  arrange(RID, VISCODE)
  
count(distinct(total,RID)) #765

# Only participants with BA info available 
final <- total[which(total$RID %in% postproc2$RID),]

vitalna <- total %>% 
  filter(VISCODE == "bl") %>% 
  sum(is.na(bmi)) #%>%
  #count(is.na(VSHEIGHT)) 


#`%not in%` <- function (x, table) is.na(match(x, table, nomatch=NA_integer_))

# Final number of participants:
count(distinct(final,RID)) #738

# Distribution of Baseline Diagnosis:
bl_distribution <- final %>% 
  group_by(DX_bl) %>% 
  summarise(n_distinct(RID))
bl_distribution

# Distribution of Final Diagnosis: 
m36_distribution <- final %>% 
  filter(VISCODE == "m36") %>% 
  group_by(DX) %>% 
  summarise(n_distinct(RID))
m36_distribution

MCI_convert <- filter(final, DX == "Dementia" & DX_bl == "MCI")
count(distinct(MCI_convert,RID))

# Diagnosis of MCI patients at m36
MCI_patients <- final %>% filter(DX_bl == 'MCI') 
#count(distinct(MCI,RID))

MCI_DX <- MCI_patients %>% filter(VISCODE == "m36") %>% select(DX)
table(MCI_DX) #10 empty values 

# Remove NAs for now
#MCI_patients <- MCI_patients %>% group_by(RID) %>% filter(DX != "")

# Check if CSF measures are available at M36 
CSF_check <- final %>% filter(DX_bl == 'MCI') %>% filter(VISCODE== 'm36')

dim(CSF_check[CSF_check$ABETA == '',])[1]  #490 missing values


# Create Conversion Variable
MCI_patients <- group_by(MCI_patients, RID) %>% 
  mutate(Convert = ifelse(DX == "Dementia", 1, 0)) %>%
  ungroup()

MCI_patients$Convert <- as.factor(MCI_patients$Convert)

check <- select(MCI_patients, RID,VISCODE, DX, Convert)


# MCI Baseline subset
MCI_bl <- MCI_patients %>% 
  filter(VISCODE == "bl") %>%
  select(RID, VISCODE, AGE, GENDER, EDUCATION, APOE4, CA:Convert)




# Look at correlation of mean values of each feature # 
#MCI$GENDER <- as.factor(MCI$GENDER)

predictors <- filter(MCI_patients, VISCODE == 'bl') %>%
  select(AGE, GENDER, EDUCATION, APOE4,DX,CA:Convert)
  
predictors$DX <- as.factor(predictors$DX)
lapply(predictors[,c(2,5)],levels)

predictors[] <- lapply(predictors,as.integer)
cor(predictors)

correlations <- cor(predictors)
correlation_plot <- corrplot(correlations, method = "circle", 
                             title = "Variable Correlations", tl.col = "black", 
                             order = "FPC", line = -2)
correlation_plot


#MCI_patients[MCI_patients$VISCODE== 'm36',]$DX == "Dementia"

#MCI_patients[VISCODE ==  'bl']$Final_DX <- 'Dementia'




