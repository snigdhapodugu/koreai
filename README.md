### Hello! Welcome to koreai! 

To use this tool:
1. Clone the repository onto your local computer.
2. Open the terminal.
3. Enter 'streamlit run model.py'.

Learn more about our project [here](https://devpost.com/software/koreai).

## Inspiration
Korean color analysis has been very popular in Western media (TikTok, Instagram, YouTube) for the past few years. However, the service can be costly and inaccessible, so we wanted to create a tool that allows you to receive an analysis at your convenience. 

## What does this tool do?
The tool extracts the RGB values of your eyes, lips, cheeks, and hair using computer vision. Using these features, the ML model predicts your undertone, season, and colors that may suit you. You are also given the opportunity to explore how your custom colors look on you, just like a Korean color analysis!

## How we built it
We began by creating a training dataset using a list of 100 celebrities who had already been labeled with their seasonal color types and undertones by stylists and experts online. For each celebrity, we sourced high-quality, front-facing images and used MediaPipe’s Face Mesh to extract facial RGB values corresponding to the eyes, lips, cheeks, and hairline. The final dataset paired these color features with the known season and undertone labels, forming the training set for our predictive model. 

We used the same facial feature tracking via MediaPipe to extract RGB values from users in real-time. For each feature, we calculated the average RGB value across a 5-second span to ensure stability. These values were then fed into a kNN classifier that was trained on our celebrity dataset. The model predicted both the user’s undertone and seasonal color match and achieved a 45% accuracy for undertone and 35% for seasonal type.

## Challenges
One of the main challenges we faced was finding a publicly available dataset that included both seasonal color types, undertone labels, and corresponding RGB values. While a few datasets existed, they were often not racially diverse and lacked detailed information on seasonal palettes. To address this, we created our own dataset of 100 celebrities. We sourced high-quality, front-facing images, then wrote a function to extract RGB values for the hair, lips, cheeks, and eyes. We manually gathered seasonal and undertone labels from expert analyses found in fashion articles and magazines.

It was also a challenge to use computer vision and accurately identify facial features like the lips, cheeks, hairline, and eyes, especially in images with background noise or dim lighting. With experimentation and tuning, we were able to build a robust pipeline using MediaPipe Face Mesh that could handle real-time face tracking even with movement, background distractions, and without requiring a plain white background.

We initially struggled to decide on a specific model. Some models, such as logistic regression and random forest classifiers, performed poorly and overfit to the training data due to the small dataset size and the complexity of color prediction. We found that a k-Nearest Neighbors (kNN) classifier worked best, since color similarity is represented by spatial proximity in RGB values which is similar to how kNN uses distance metrics to make predictions.

Lastly, we noticed that lighting significantly impacted the accuracy of color extraction. For better predictions, users should use our tool in well-lit environments, preferably white lighting with minimal shadows or color distortions.

## Accomplishments
One accomplishment was not only creating our own dataset, but also creating a program to extract the RGB values from celebrity’s faces as part of the dataset. The biggest accomplishment was being able to implement this in real-time through computer vision, where we identified the RGB colors of users’ facial features. We created a color analysis algorithm by locating existing categorizations of seasonal color patterns and formulated a way to mathematically transform RGB coordinates to map a skin color to a set of compatible clothing colors.

## What we learned
To start with, we needed to learn a fair bit about color theory and why certain colors look good with other colors. We also learned about the different moods (seasons) a color can invoke. More importantly, we learned about the importance of diversity in making machine learning algorithms. In order to create a product that is suitable for use by everyone, we needed a dataset consisting of individuals from various different backgrounds.

## What's next for koreai?
We plan to expand our dataset to include at least 500 individuals. Increasing the number of data points will help improve the accuracy of our model, leading to more reliable seasonal and undertone predictions, as well as more refined color palette recommendations for users.

We also want to explore new ways to detect undertones using real-time image analysis. Undertones are typically identified using two methods: first, checking whether your veins appear more blue or green; second, comparing your skin against a plain white background to see if it leans more yellow, pink, or neutral. We want to develop a feature that allows users to perform either test through their camera in real time, and then incorporate the results into the model to enhance undertone detection, and subsequently the seasonal detection.
