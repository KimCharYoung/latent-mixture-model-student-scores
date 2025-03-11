#####Run libraries#####
library(rstan)
library(ggplot2)

#####2" Configure R to use multiple cores when compiling Stan models####
dotR <- file.path(Sys.getenv("HOME"), ".R")
if (!file.exists(dotR)) dir.create(dotR)
M <- file.path(dotR, "Makevars")
if (!file.exists(M)) file.create(M)
cat("\nCXX14FLAGS=-O3 -march=native -mtune=native",
    "\nCXX14=g++ -std=c++1y", file = M, append = TRUE)

# Open the Makevars file and fix it
makevars_path <- file.path(Sys.getenv("HOME"), ".R", "Makevars")
# Read the current content
current_content <- readLines(makevars_path)
# Remove any lines with -march=native
new_content <- current_content[!grepl("-march=native", current_content)]
# Write the fixed content back
writeLines(new_content, makevars_path)
# Add a proper newline at the end if missing
if (length(new_content) > 0 && substr(new_content[length(new_content)], nchar(new_content[length(new_content)]), nchar(new_content[length(new_content)])) != "\n") {
  write("", file = makevars_path, append = TRUE)
}

## Define Stan Model
stan_model <- "
data {
  int<lower=0> p;         // Number of students
  int<lower=0> n;         // Number of questions
  int<lower=0, upper=n> k[p];  // Scores for each student
}

parameters {
  real<lower=0.5, upper=1> phi;  // Rate for knowledge group
  vector<lower=0, upper=1>[p] z_prob;  // Probability of being in knowledge group
}

transformed parameters {
  vector<lower=0, upper=1>[p] theta;  // Individual probabilities
  
  for (i in 1:p) {
    // Weighted average of the two rates
    theta[i] = (1 - z_prob[i]) * 0.5 + z_prob[i] * phi;
  }
}

model {
  // Prior for group membership probability
  z_prob ~ beta(1, 1);  // Uniform prior on z_prob
  
  // Prior for knowledge group success rate is implicitly uniform from bounds
  
  // Likelihood
  for (i in 1:p) {
    k[i] ~ binomial(n, theta[i]);
  }
}

generated quantities {
  // Classify students - group assignment with 0.5 threshold
  int<lower=0, upper=1> z[p];
  
  for (i in 1:p) {
    z[i] = (z_prob[i] > 0.5);
  }
}
"

## Run the Model
# Set rstan options to avoid warnings
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# Run the model
exam_data <- list(
  p = 15,                 # Number of students
  n = 40,                 # Number of questions
  k = c(21, 17, 21, 18, 22, 31, 31, 34, 34, 35, 35, 36, 39, 36, 35)  # Scores
)

fit <- stan(
  model_code = stan_model,  # Now explicitly defined
  data = exam_data,
  chains = 4,
  iter = 2000,
  warmup = 1000
)


# Print summary of key parameters
print(fit, pars = c("phi", "z_prob"))


# Extract the probability samples
z_probs <- extract(fit, pars = "z_prob")$z_prob
prob_knowledge <- colMeans(z_probs)

# Create a data frame with student information
student_data <- data.frame(
  Student = 1:15,
  Score = exam_data$k,
  Prob_Knowledge_Group = prob_knowledge
)

# View the results
print(student_data, row.names = FALSE)


# Create a visualization
ggplot(student_data, aes(x = reorder(factor(Student), Score), y = Score, color = Prob_Knowledge_Group)) +
  geom_point(size = 5) +
  scale_color_gradient(low = "red", high = "blue", 
                       name = "Prob. Knowledge\nGroup") +
  labs(title = "Student Scores with Group Classification",
       subtitle = "Blue = Knowledge Group, Red = Guessing Group",
       x = "Student ID (ordered by score)", 
       y = "Score (out of 40)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



# Extract and summarize phi
phi_samples <- extract(fit, pars = "phi")$phi
mean_phi <- mean(phi_samples)
cat("Estimated knowledge group success rate (phi):", round(mean_phi, 3), "\n")

# Plot histogram of phi
hist(phi_samples, main = "Posterior Distribution of Knowledge Group Success Rate", 
     xlab = "phi", col = "lightblue", breaks = 30)


# Add a column indicating likely group assignment
student_data$Group <- ifelse(student_data$Prob_Knowledge_Group > 0.5, 
                             "Knowledge", "Guessing")

# Print the categorized results
student_data <- student_data[order(student_data$Score), ]  # Sort by score
print(student_data, row.names = FALSE)

# Calculate mean scores by group
aggregate(Score ~ Group, data = student_data, FUN = mean)