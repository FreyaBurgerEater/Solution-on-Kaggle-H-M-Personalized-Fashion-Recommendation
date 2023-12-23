#Project Overview

This project is an implementation inspired by the strategy employed by the Silver Medal-winning team in a previous competition. The competition focused on H&M Personalized Fashion Recommendations on Kaggle, and the goal was to predict the items each customer would purchase in the 7 days following the training data period.

#Data Description

Official Data Page: [H&M Personalized Fashion Recommendations | Kaggle](link_to_kaggle)

The dataset includes transaction data, customer metadata, and item metadata (including item descriptions, text, and images).

- `images/`: Folder containing images for each `article_id`. Note that not all `article_id` values have corresponding images.
- `articles.csv`: Detailed metadata for each item with `article_id`.
- `customers.csv`: Metadata for each customer with `customer_id`.
- `sample_submission.csv`: Sample submission file in the correct format.
- `transactions_train.csv`: Training data including customer purchases on each date, along with other information. Duplicate rows correspond to multiple purchases of the same item.

#Approach

The project adopted a strategy similar to that of the Silver Medal-winning team. The machine configuration included a GPU (NVIDIA 3090), CPU (Intel i9-10900k), 128GB RAM, and an Ubuntu 18.04 operating system.

#Solution Strategy

The project primarily utilized the tabular data part of the dataset and excluded image and text data.
The chosen model for this project was Catboost.

Key Strategies:

Data Preprocessing Optimization: To enhance efficiency, all tabular data was initially converted to pickle format, reducing data loading time significantly.
Feature Engineering: The team focused on constructing meaningful features through a four-step process:
Creating a user-item matrix and training user embeddings using the LightFM library.
Performing one-hot encoding for item attributes and aggregating them by users.
Generating candidate items and deriving ranking features.
Generating static and dynamic features for users, items, and user-item pairs.
Model Selection: Catboost was chosen as the final model for training.
Training and Validation: The dataset was split into training and validation sets, with the last week as the validation set. The Catboost model achieved a CV MAP of 0.324.
