 A big thank you to Gerard Taylor for his original code! I've learned a lot from his work, which laid a solid foundation for this project. Building on his implementation, I've made some enhancements and improvements to address specific issues and optimize the performance. Below are the main changes I've introduced:
Efficient Data Splitting
The process of randomly grouping samples into training, testing, and validation sets has been optimized. The original implementation used random_shuffle to shuffle the dataArray, which could be computationally intensive. I've replaced it with a more efficient approach to improve performance.

Bug Fix in Weight Initialization
There was an issue with the weight initialization during training. Specifically, the function generator_random_number did not utilize random seeds。 I've fixed this by ensuring that random seeds are correctly applied, improving randomness in the beginning of training at each time.

Enhanced Data Normalization
The original code supported training on two different datasets, but the normalization function encountered difficulties when handling multiple datasets. To address this, I've defined two separate normalization functions, allowing for independent and accurate normalization of each dataset.
