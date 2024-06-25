import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Parameters for the Beta distribution
alpha, beta_param = 6, 33

# Generate a range of values between 0 and 1
x = np.linspace(0, 1, 1000)

# Compute the PDF for the Beta distribution
pdf = beta.pdf(x, alpha, beta_param)

# Compute the 95% confidence interval
ci_lower = beta.ppf(0.025, alpha, beta_param)
ci_upper = beta.ppf(0.975, alpha, beta_param)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(x, pdf, label='Beta(6, 33) PDF')
plt.fill_between(x, pdf, where=(x >= ci_lower) & (x <= ci_upper), color='skyblue', alpha=0.4, label='95% CI')
plt.axvline(ci_lower, color='grey', linestyle='--', label='2.5% Percentile')
plt.axvline(ci_upper, color='grey', linestyle='--', label='97.5% Percentile')
plt.title('Beta(6, 33) Distribution with 95% Confidence Interval')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
